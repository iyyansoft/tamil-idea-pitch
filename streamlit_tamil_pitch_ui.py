import json
import os
import tempfile
import traceback
from difflib import SequenceMatcher
from pathlib import Path

import joblib
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from pydub import AudioSegment
from pydub.silence import split_on_silence

from speech_pipeline.inference_features import build_model_input
from semantic_eval.final_evaluator import TamilIdeaPitchEvaluator

from semantic_eval.tamil_transcript_refiner import (
    load_reference_texts,
    build_dataset_terms,
    refine_chunk_text,
    refine_full_transcript,
    analyze_chunk_with_dataset,
    format_raw_full_transcript,
    format_raw_transcript_by_chunks,
    format_refined_transcript_by_chunks,
    format_numbered_sentences,
)


MODEL_PATH = "models/confidence_model.pkl"
KB_PATH = "data/knowledge_base_tamil_reasoning.json"
TEXT_DATASET_PATH = "output/tamil_feature_dataset_labeled.csv"

SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg")

TARGET_CHUNK_MS = 15000
MIN_SILENCE_LEN = 700
SILENCE_THRESH_OFFSET_DB = 16
KEEP_SILENCE_MS = 250


def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return str(obj)


def _sequence_similarity(a: str, b: str) -> float:
    a = (a or "").strip()
    b = (b or "").strip()

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def normalize_score(value):
    """
    Converts score to 0-1 range.
    Supports 0-1, 0-10, and 0-100 score formats.
    """
    try:
        value = float(value)
    except Exception:
        return 0.0

    if value > 10:
        value = value / 100.0
    elif value > 1:
        value = value / 10.0

    return max(0.0, min(1.0, value))


def load_audio_file(audio_path: str) -> AudioSegment:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if not audio_path.lower().endswith(SUPPORTED_EXTENSIONS):
        raise ValueError(f"Unsupported audio format: {audio_path}")

    return AudioSegment.from_file(audio_path).set_channels(1).set_frame_rate(16000)


def split_long_chunk(chunk: AudioSegment, target_ms: int = TARGET_CHUNK_MS):
    """
    Splits long chunks into smaller fixed-size chunks.
    This prevents the last/long chunk from being skipped.
    """

    parts = []

    for start in range(0, len(chunk), target_ms):
        end = min(start + target_ms, len(chunk))
        part = chunk[start:end]

        if len(part) > 300:
            parts.append(part)

    return parts


def adaptive_split_audio(audio: AudioSegment):
    """
    Silence-aware splitting + fixed splitting fallback.
    Keeps the primary website behavior stable while processing full audio.
    """

    silence_thresh = (
        audio.dBFS - SILENCE_THRESH_OFFSET_DB
        if audio.dBFS != float("-inf")
        else -40
    )

    pieces = split_on_silence(
        audio,
        min_silence_len=MIN_SILENCE_LEN,
        silence_thresh=silence_thresh,
        keep_silence=KEEP_SILENCE_MS,
    )

    if not pieces:
        return split_long_chunk(audio, TARGET_CHUNK_MS)

    final_chunks = []
    buffer = AudioSegment.silent(duration=0)

    for piece in pieces:
        if len(piece) > TARGET_CHUNK_MS:
            if len(buffer) > 0:
                final_chunks.extend(split_long_chunk(buffer, TARGET_CHUNK_MS))
                buffer = AudioSegment.silent(duration=0)

            final_chunks.extend(split_long_chunk(piece, TARGET_CHUNK_MS))
            continue

        if len(buffer) + len(piece) <= TARGET_CHUNK_MS:
            buffer += piece
        else:
            if len(buffer) > 0:
                final_chunks.extend(split_long_chunk(buffer, TARGET_CHUNK_MS))
            buffer = piece

    if len(buffer) > 0:
        final_chunks.extend(split_long_chunk(buffer, TARGET_CHUNK_MS))

    final_chunks = [chunk for chunk in final_chunks if len(chunk) > 300]

    return final_chunks


def predict_confidence(confidence_model, text: str):
    if not text.strip():
        return None

    feature_df = build_model_input(text, confidence_model)
    expected_features = getattr(confidence_model, "n_features_in_", None)

    if expected_features is not None and feature_df.shape[1] != expected_features:
        raise ValueError(
            f"Feature mismatch: model expects {expected_features} features, "
            f"but got {feature_df.shape[1]}. Current columns: {list(feature_df.columns)}"
        )

    prediction = confidence_model.predict(feature_df)[0]

    probability = None
    if hasattr(confidence_model, "predict_proba"):
        probability = float(max(confidence_model.predict_proba(feature_df)[0]))

    return prediction, probability


def confidence_reliability(
    text: str,
    prev_text: str | None,
    stt_quality: float,
    probability: float | None,
) -> float:
    length = len(text.split())
    length_score = min(length / 18.0, 1.0)

    repetition_penalty = _sequence_similarity(text, prev_text) if prev_text else 0.0
    model_prob = probability if probability is not None else 0.5

    score = (
        0.45 * stt_quality
        + 0.35 * model_prob
        + 0.20 * length_score
        - 0.25 * repetition_penalty
    )

    return round(max(0.0, min(1.0, score)), 4)


def aggregate_asr_quality(chunk_results):
    vals = [
        c["confidence_reliability"]
        for c in chunk_results
        if c.get("refined_text")
    ]

    if not vals:
        return 0.0

    return round(sum(vals) / len(vals), 4)


def calculate_audio_pitch_confidence(chunk_results, asr_quality):
    """
    Calculates how confidently the idea was pitched through audio using:
    ASR quality, chunk confidence, Tamil dataset coverage, technical strength,
    successful chunk ratio, and uncertainty penalty.
    """

    if not chunk_results:
        return {
            "audio_pitch_confidence_rate": 0.0,
            "audio_pitch_confidence_label": "No Audio Confidence",
            "audio_pitch_confidence_explanation": "No chunks were available for confidence analysis.",
            "audio_pitch_confidence_breakdown": {
                "asr_quality": 0.0,
                "avg_chunk_confidence": 0.0,
                "avg_dataset_coverage": 0.0,
                "technical_strength": 0.0,
                "successful_chunk_ratio": 0.0,
                "uncertainty_penalty": 0.0,
            },
        }

    total_chunks = len(chunk_results)

    successful_chunks = [
        row for row in chunk_results
        if str(row.get("raw_text", "")).strip()
    ]

    successful_chunk_ratio = len(successful_chunks) / total_chunks if total_chunks else 0.0

    confidence_values = [
        float(row.get("confidence_reliability", 0.0))
        for row in chunk_results
    ]

    dataset_coverage_values = [
        float(row.get("dataset_coverage", 0.0))
        for row in chunk_results
    ]

    technical_counts = [
        float(row.get("technical_word_count", 0.0))
        for row in chunk_results
    ]

    uncertain_counts = [
        float(row.get("uncertain_word_count", 0.0))
        for row in chunk_results
    ]

    avg_chunk_confidence = (
        sum(confidence_values) / len(confidence_values)
        if confidence_values else 0.0
    )

    avg_dataset_coverage = (
        sum(dataset_coverage_values) / len(dataset_coverage_values)
        if dataset_coverage_values else 0.0
    )

    avg_technical_words = (
        sum(technical_counts) / len(technical_counts)
        if technical_counts else 0.0
    )

    avg_uncertain_words = (
        sum(uncertain_counts) / len(uncertain_counts)
        if uncertain_counts else 0.0
    )

    technical_strength = min(avg_technical_words / 4.0, 1.0)
    uncertainty_penalty = min(avg_uncertain_words / 3.0, 1.0)

    confidence_rate = (
        0.25 * float(asr_quality)
        + 0.25 * avg_chunk_confidence
        + 0.20 * avg_dataset_coverage
        + 0.15 * technical_strength
        + 0.15 * successful_chunk_ratio
        - 0.15 * uncertainty_penalty
    )

    confidence_rate = round(max(0.0, min(1.0, confidence_rate)), 4)

    if confidence_rate >= 0.80:
        label = "Very Confident Pitch"
        explanation = (
            "The audio pitch is clear, technically meaningful, strongly aligned with the Tamil dataset, "
            "and has high transcription reliability."
        )

    elif confidence_rate >= 0.65:
        label = "Confident Pitch"
        explanation = (
            "The audio pitch is mostly clear and meaningful, with good dataset coverage and reliable speech content."
        )

    elif confidence_rate >= 0.45:
        label = "Moderately Confident Pitch"
        explanation = (
            "The pitch has useful content, but it may need clearer explanation, stronger technical terms, "
            "or improved speech clarity."
        )

    elif confidence_rate >= 0.25:
        label = "Low Confidence Pitch"
        explanation = (
            "The pitch has weak clarity or limited dataset/technical coverage. The idea may need to be spoken "
            "more clearly with stronger reasoning."
        )

    else:
        label = "Very Low Confidence Pitch"
        explanation = (
            "The system could not confidently understand the pitch. This may be due to unclear audio, weak speech, "
            "low transcription success, or insufficient meaningful content."
        )

    return {
        "audio_pitch_confidence_rate": confidence_rate,
        "audio_pitch_confidence_label": label,
        "audio_pitch_confidence_explanation": explanation,
        "audio_pitch_confidence_breakdown": {
            "asr_quality": round(float(asr_quality), 4),
            "avg_chunk_confidence": round(avg_chunk_confidence, 4),
            "avg_dataset_coverage": round(avg_dataset_coverage, 4),
            "technical_strength": round(technical_strength, 4),
            "successful_chunk_ratio": round(successful_chunk_ratio, 4),
            "uncertainty_penalty": round(uncertainty_penalty, 4),
        },
    }


def generate_audio_based_feedback(chunk_results, audio_pitch_confidence, kb_result=None):
    """
    Generates dynamic feedback from actual audio/chunk analysis.
    """

    feedback = []

    if not chunk_results:
        return ["No valid audio chunks were available for feedback generation."]

    total_chunks = len(chunk_results)

    successful_chunks = [
        row for row in chunk_results
        if str(row.get("raw_text", "")).strip()
    ]

    weak_chunks = [
        row for row in chunk_results
        if row.get("analysis_status") in ["Weak", "Needs Review"]
    ]

    repeated_chunks = [
        row for row in chunk_results
        if float(row.get("repetition_score", 0.0)) > 0.70
    ]

    uncertain_chunks = [
        row for row in chunk_results
        if int(row.get("uncertain_word_count", 0)) > 0
    ]

    technical_chunks = [
        row for row in chunk_results
        if int(row.get("technical_word_count", 0)) > 0
    ]

    avg_stt_quality = sum(
        float(row.get("stt_quality", 0.0))
        for row in chunk_results
    ) / total_chunks

    avg_confidence = sum(
        float(row.get("confidence_reliability", 0.0))
        for row in chunk_results
    ) / total_chunks

    avg_dataset_coverage = sum(
        float(row.get("dataset_coverage", 0.0))
        for row in chunk_results
    ) / total_chunks

    successful_ratio = len(successful_chunks) / total_chunks

    pitch_rate = audio_pitch_confidence.get("audio_pitch_confidence_rate", 0.0)
    pitch_label = audio_pitch_confidence.get("audio_pitch_confidence_label", "Not Available")

    if pitch_rate >= 0.80:
        feedback.append(
            f"Your audio pitch sounds highly confident. The system classified it as '{pitch_label}' because most chunks were clear, meaningful, and aligned with the Tamil dataset."
        )
    elif pitch_rate >= 0.65:
        feedback.append(
            f"Your audio pitch is confident overall. The system classified it as '{pitch_label}', but a few areas can still be improved for stronger delivery."
        )
    elif pitch_rate >= 0.45:
        feedback.append(
            "Your audio pitch is moderately confident. The idea is understandable, but the delivery needs clearer explanation, stronger technical terms, or better speech clarity."
        )
    elif pitch_rate >= 0.25:
        feedback.append(
            "Your audio pitch has low confidence. The speech contains some useful content, but many chunks need clearer pronunciation, stronger reasoning, or better idea structure."
        )
    else:
        feedback.append(
            "The system could not confidently evaluate your audio pitch. This may be due to unclear speech, poor audio quality, low transcription success, or limited meaningful content."
        )

    if successful_ratio < 0.50:
        feedback.append(
            f"Only {len(successful_chunks)} out of {total_chunks} chunks produced recognizable text. Try speaking louder, closer to the microphone, and reduce background noise."
        )
    elif successful_ratio < 0.80:
        feedback.append(
            f"{len(successful_chunks)} out of {total_chunks} chunks were successfully transcribed. Some parts of the audio were unclear, so improving speech clarity will improve the evaluation."
        )
    else:
        feedback.append(
            f"Most chunks were successfully transcribed: {len(successful_chunks)} out of {total_chunks}. This means the audio was mostly understandable."
        )

    if avg_stt_quality < 0.45:
        feedback.append(
            "The average STT quality is low. The audio may contain unclear pronunciation, low volume, background noise, or long pauses."
        )
    elif avg_stt_quality < 0.65:
        feedback.append(
            "The STT quality is acceptable, but improving pronunciation and reducing noise can make the transcript more accurate."
        )
    else:
        feedback.append(
            "The STT quality is good, so the spoken content was reasonably clear for transcription."
        )

    if avg_dataset_coverage < 0.30:
        feedback.append(
            "The pitch has low alignment with the Tamil analyzed dataset. Add more relevant domain terms and explain the idea using clearer technical or project-related vocabulary."
        )
    elif avg_dataset_coverage < 0.60:
        feedback.append(
            "The pitch has moderate dataset alignment. It contains some relevant Tamil dataset terms, but adding more specific technical concepts will make it stronger."
        )
    else:
        feedback.append(
            "The pitch aligns well with the Tamil analyzed dataset, meaning the spoken content contains relevant idea-related terms."
        )

    if len(technical_chunks) == 0:
        feedback.append(
            "The pitch does not contain many technical words. Add details about components, working method, implementation, tools, or technology used."
        )
    elif len(technical_chunks) < total_chunks / 2:
        feedback.append(
            "Some chunks contain technical words, but the technical explanation is not consistent throughout the pitch. Try explaining the working process more clearly."
        )
    else:
        feedback.append(
            "The pitch includes technical terms across multiple chunks, which improves the strength of the idea presentation."
        )

    if uncertain_chunks:
        chunk_numbers = [str(row.get("chunk")) for row in uncertain_chunks[:5]]
        feedback.append(
            f"Uncertain wording was detected in chunk(s): {', '.join(chunk_numbers)}. Avoid vague phrases and speak with more direct explanation."
        )

    if repeated_chunks:
        chunk_numbers = [str(row.get("chunk")) for row in repeated_chunks[:5]]
        feedback.append(
            f"Repeated content was detected around chunk(s): {', '.join(chunk_numbers)}. Try reducing repeated statements and add new points such as benefits, implementation, cost, or real-world use."
        )

    if weak_chunks:
        chunk_numbers = [str(row.get("chunk")) for row in weak_chunks[:6]]
        feedback.append(
            f"Chunk(s) {', '.join(chunk_numbers)} need review because their clarity score or dataset match was low."
        )
    else:
        feedback.append(
            "No major weak chunks were detected. The pitch is structurally consistent across the audio."
        )

    if kb_result:
        try:
            scores = kb_result.get("scores", {})

            technical_score = normalize_score(scores.get("technical_correctness", 0.0))
            real_life_score = normalize_score(scores.get("real_life_probability", 0.0))
            theoretical_score = normalize_score(scores.get("theoretical_correctness", 0.0))

            if technical_score < 0.50:
                feedback.append(
                    "The technical correctness score is low. Explain how the idea works step-by-step and mention the required components or technology."
                )

            if real_life_score < 0.50:
                feedback.append(
                    "The real-life probability score is low. Add practical details such as where it can be used, who benefits, cost, limitations, and implementation feasibility."
                )

            if theoretical_score < 0.50:
                feedback.append(
                    "The theoretical correctness score is low. Improve the reasoning behind the idea and explain why the proposed method is valid."
                )

        except Exception:
            pass

    if avg_confidence >= 0.70 and avg_dataset_coverage >= 0.60:
        feedback.append(
            "Overall, your idea pitch is strong. To make it even better, add a short conclusion explaining impact, novelty, and practical implementation."
        )
    elif avg_confidence >= 0.50:
        feedback.append(
            "Overall, the pitch is understandable. Improve it by speaking more clearly, reducing vague words, and adding stronger technical and real-world explanation."
        )
    else:
        feedback.append(
            "Overall, the pitch needs improvement. Focus on clearer speech, better structure, stronger technical explanation, and more dataset-relevant terms."
        )

    return feedback


@st.cache_resource(show_spinner=False)
def load_cached_models(model_path: str, kb_path: str, text_dataset_path: str):
    confidence_model = joblib.load(model_path)
    evaluator = TamilIdeaPitchEvaluator(kb_path)

    reference_texts = load_reference_texts(
        text_dataset_path=text_dataset_path,
        kb_path=kb_path,
    )

    dataset_terms = build_dataset_terms(reference_texts)

    return confidence_model, evaluator, dataset_terms


def save_uploaded_audio(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def safe_transcribe_chunk(chunk, prefer_torch=False):
    """
    Keeps the STT interface and adds a final Streamlit-side retry.
    """

    from speech_pipeline.tamil_stt_free import transcribe_audiosegment_free

    try:
        transcribed = transcribe_audiosegment_free(
            chunk,
            prefer_torch=prefer_torch,
        )
    except TypeError:
        transcribed = transcribe_audiosegment_free(chunk)

    if not isinstance(transcribed, dict):
        transcribed = {
            "text": "",
            "stt_engine": "unknown",
            "language_code": "ta-IN",
            "whisper_quality": 0.0,
            "error": "STT function did not return dictionary",
            "retry_attempt": None,
        }

    if not transcribed.get("text", "").strip():
        try:
            louder_chunk = chunk.apply_gain(8)

            try:
                retry_transcribed = transcribe_audiosegment_free(
                    louder_chunk,
                    prefer_torch=prefer_torch,
                )
            except TypeError:
                retry_transcribed = transcribe_audiosegment_free(louder_chunk)

            if isinstance(retry_transcribed, dict) and retry_transcribed.get("text", "").strip():
                retry_transcribed["error"] = "Recovered after Streamlit-level volume retry"
                retry_transcribed["retry_attempt"] = retry_transcribed.get(
                    "retry_attempt",
                    "streamlit_plus_8db",
                )
                transcribed = retry_transcribed

        except Exception as retry_error:
            transcribed["error"] = (
                f"{transcribed.get('error')} | "
                f"Streamlit retry failed: {retry_error}"
            )

    transcribed.setdefault("text", "")
    transcribed.setdefault("stt_engine", "unknown")
    transcribed.setdefault("language_code", "ta-IN")
    transcribed.setdefault("whisper_quality", 0.0)
    transcribed.setdefault("error", None)
    transcribed.setdefault("retry_attempt", None)

    return transcribed


def run_pipeline(
    audio_path: str,
    model_path: str,
    kb_path: str,
    text_dataset_path: str,
    prefer_torch: bool = False,
):
    confidence_model, evaluator, dataset_terms = load_cached_models(
        model_path,
        kb_path,
        text_dataset_path,
    )

    audio = load_audio_file(audio_path)
    chunks = adaptive_split_audio(audio)

    if not chunks:
        raise ValueError("No valid audio chunks were created from the uploaded audio.")

    raw_text_list = []
    refined_text_list = []
    chunk_results = []

    previous_refined_text = None
    total_chunks = len(chunks)

    progress_bar = st.progress(0)
    status = st.empty()

    for i, chunk in enumerate(chunks, start=1):
        status.write(f"Processing chunk {i}/{total_chunks}...")

        transcribed = safe_transcribe_chunk(
            chunk,
            prefer_torch=prefer_torch,
        )

        raw_text = transcribed.get("text", "").strip()
        refined_text = refine_chunk_text(raw_text)

        raw_text_list.append(raw_text)
        refined_text_list.append(refined_text)

        row = {
            "chunk": i,
            "total_chunks": total_chunks,
            "duration_sec": round(len(chunk) / 1000.0, 2),
            "raw_text": raw_text,
            "refined_text": refined_text,
            "stt_engine": transcribed.get("stt_engine"),
            "language_code": transcribed.get("language_code"),
            "stt_quality": transcribed.get("whisper_quality", 0.70),
            "stt_error": transcribed.get("error"),
            "stt_retry_attempt": transcribed.get("retry_attempt"),
            "prediction": None,
            "probability": None,
            "confidence_reliability": 0.0,
        }

        if refined_text.strip():
            pred_result = predict_confidence(confidence_model, refined_text)

            if pred_result is not None:
                row["prediction"], row["probability"] = pred_result

        row["confidence_reliability"] = confidence_reliability(
            text=refined_text,
            prev_text=previous_refined_text,
            stt_quality=row["stt_quality"],
            probability=row["probability"],
        )

        dataset_analysis = analyze_chunk_with_dataset(
            raw_text=raw_text,
            refined_text=refined_text,
            previous_refined_text=previous_refined_text,
            stt_quality=row["stt_quality"],
            model_probability=row["probability"],
            dataset_terms=dataset_terms,
        )

        row.update(dataset_analysis)

        previous_refined_text = refined_text
        chunk_results.append(row)

        progress_bar.progress(i / total_chunks)

    status.empty()
    progress_bar.empty()

    raw_full_transcript = " ".join(raw_text_list).strip()
    refined_full_input = " ".join(refined_text_list).strip()

    refined_full_transcript = refine_full_transcript(refined_full_input)

    neat_raw_full_transcript = format_raw_full_transcript(raw_full_transcript)

    raw_transcript_by_chunks = format_raw_transcript_by_chunks(chunk_results)
    refined_transcript_by_chunks = format_refined_transcript_by_chunks(chunk_results)
    numbered_refined_sentences = format_numbered_sentences(refined_full_transcript)

    asr_quality = aggregate_asr_quality(chunk_results)

    audio_pitch_confidence = calculate_audio_pitch_confidence(
        chunk_results=chunk_results,
        asr_quality=asr_quality,
    )

    kb_result = (
        evaluator.evaluate(refined_full_transcript, asr_quality=asr_quality)
        if refined_full_transcript
        else None
    )

    audio_based_feedback = generate_audio_based_feedback(
        chunk_results=chunk_results,
        audio_pitch_confidence=audio_pitch_confidence,
        kb_result=kb_result,
    )

    return {
        "raw_full_transcript": raw_full_transcript,
        "neat_raw_full_transcript": neat_raw_full_transcript,
        "refined_full_transcript": refined_full_transcript,
        "raw_transcript_by_chunks": raw_transcript_by_chunks,
        "refined_transcript_by_chunks": refined_transcript_by_chunks,
        "numbered_refined_sentences": numbered_refined_sentences,
        "asr_quality": asr_quality,
        "audio_pitch_confidence": audio_pitch_confidence,
        "audio_based_feedback": audio_based_feedback,
        "chunks": chunk_results,
        "kb_result": kb_result,
    }




def score_to_100(value):
    """
    Converts a score into 0-100.
    Supports 0-1, 0-10, and 0-100 formats.
    """
    try:
        value = float(value)
    except Exception:
        return 0.0

    if value <= 1:
        value = value * 100
    elif value <= 10:
        value = value * 10

    return round(max(0.0, min(100.0, value)), 2)


def score_to_10(value):
    """
    Converts a score into 0-10.
    Supports 0-1, 0-10, and 0-100 formats.
    """
    try:
        value = float(value)
    except Exception:
        return 0.0

    if value <= 1:
        value = value * 10
    elif value > 10:
        value = value / 10

    return round(max(0.0, min(10.0, value)), 2)


def format_score_out_of_total(value, total=10):
    return f"{score_to_10(value):.2f} / {total}"


def make_unique_key(*parts):
    cleaned = []
    for part in parts:
        part = str(part)
        part = part.replace(" ", "_").replace("/", "_").replace("-", "_")
        part = "".join(ch for ch in part if ch.isalnum() or ch == "_")
        cleaned.append(part[:50])
    return "_".join(cleaned)


def plot_interactive_bar_chart(title, labels, values, key_prefix, y_axis_mode="unit"):
    """
    Interactive bar chart with unique Streamlit key.
    y_axis_mode='unit' shows raw 0-1 values.
    y_axis_mode='percent' shows 0-100 values.
    """
    if y_axis_mode in ["score10", "percent"]:
        plot_values = [score_to_10(v) for v in values]
        y_title = "Score out of 10"
        y_range = [0, 10]
        text_values = [f"{v:.2f}" for v in plot_values]
        hover = "<b>%{x}</b><br>Score: %{y:.2f}/10<extra></extra>"
    else:
        plot_values = [normalize_score(v) for v in values]
        y_title = "Metric value"
        y_range = [0, 1]
        text_values = [f"{v:.4f}" for v in plot_values]
        hover = "<b>%{x}</b><br>Value: %{y:.4f}<extra></extra>"

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=plot_values,
                text=text_values,
                textposition="auto",
                hovertemplate=hover,
            )
        ]
    )

    fig.update_layout(
        title=title,
        yaxis_title=y_title,
        xaxis_title="Metric",
        yaxis=dict(range=y_range),
        height=430,
        margin=dict(l=20, r=20, t=60, b=90),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=make_unique_key("bar", key_prefix, title),
    )


def display_match_metric_visuals(match, title, key_prefix):
    """
    Visualizes debug metric details like similarity, dense_similarity,
    sparse_similarity, and coverage using a modern hoverable table and bar graph.
    Pie charts are intentionally removed.
    """
    if not match:
        st.info("No match metrics available.")
        return

    metric_keys = [
        "similarity",
        "dense_similarity",
        "sparse_similarity",
        "coverage",
    ]

    rows = []
    labels = []
    values = []

    for key in metric_keys:
        if key in match:
            raw_value = float(match.get(key, 0.0))
            labels.append(key)
            values.append(raw_value)
            rows.append(
                {
                    "Metric": key,
                    "Raw Value": round(raw_value, 4),
                    "Score / 10": f"{score_to_10(raw_value):.2f} / 10",
                    "Level": score_interpretation(raw_value),
                }
            )

    if not rows:
        st.info("No metric values found for this match.")
        return

    st.markdown(f"**{title} — Metric Table**")

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["<b>Metric</b>", "<b>Raw Value</b>", "<b>Score</b>", "<b>Level</b>"],
                    fill_color="#1f2937",
                    font=dict(color="white", size=14),
                    align="left",
                    height=34,
                ),
                cells=dict(
                    values=[
                        [row["Metric"] for row in rows],
                        [row["Raw Value"] for row in rows],
                        [row["Score / 10"] for row in rows],
                        [row["Level"] for row in rows],
                    ],
                    fill_color=["#111827", "#0f172a", "#111827", "#0f172a"],
                    font=dict(color="white", size=13),
                    align="left",
                    height=30,
                ),
            )
        ]
    )

    fig.update_layout(
        title=f"{title} — Hoverable Metric Table",
        height=250,
        margin=dict(l=10, r=10, t=55, b=10),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=make_unique_key("match_metric_table", key_prefix, title),
    )

    plot_interactive_bar_chart(
        title=f"{title} — Metric Bar Graph",
        labels=labels,
        values=values,
        key_prefix=key_prefix,
        y_axis_mode="unit",
    )


def display_all_match_metric_visuals(kb_result, prefix="kb"):
    if not kb_result:
        return

    matches_by_metric = kb_result.get("matches", {})

    if not matches_by_metric:
        st.info("No match details available for metric visualization.")
        return

    st.subheader("🧩 Debug Metric Graphs for Top Matches")

    for metric_index, (metric_name, matches) in enumerate(matches_by_metric.items(), start=1):
        with st.expander(f"{metric_name} - Match Metric Graphs", expanded=False):
            if not matches:
                st.write("No matches available.")
                continue

            for idx, match in enumerate(matches, start=1):
                st.markdown(f"### Match {idx}")

                display_match_metric_visuals(
                    match=match,
                    title=f"{metric_name} Match {idx}",
                    key_prefix=f"{prefix}_{metric_index}_{idx}",
                )

                matched_text = (
                    match.get("tamil")
                    or match.get("english")
                    or match.get("reasoning")
                    or ""
                )

                if matched_text:
                    with st.expander("Matched Knowledge-base Text", expanded=False):
                        st.write(matched_text)

                st.divider()



def score_interpretation(value):
    score = score_to_10(value)
    if score >= 8:
        return "Strong"
    if score >= 6:
        return "Good"
    if score >= 4:
        return "Moderate"
    if score >= 2:
        return "Weak"
    return "Very Weak"


def display_kb_score_table(kb_result, prefix="kb"):
    """Modern interactive Plotly table for knowledge-base scores, shown out of 10."""
    if not kb_result:
        return

    scores = kb_result.get("scores", {})

    rows = [
        ("Technical Correctness", scores.get("technical_correctness", 0.0)),
        ("Real-life Probability", scores.get("real_life_probability", 0.0)),
        ("Theoretical Correctness", scores.get("theoretical_correctness", 0.0)),
        ("Final Score", kb_result.get("final_score", 0.0)),
    ]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["<b>Metric</b>", "<b>Score</b>", "<b>Level</b>"],
                    fill_color="#1f2937",
                    font=dict(color="white", size=14),
                    align="left",
                    height=34,
                ),
                cells=dict(
                    values=[
                        [r[0] for r in rows],
                        [format_score_out_of_total(r[1]) for r in rows],
                        [score_interpretation(r[1]) for r in rows],
                    ],
                    fill_color=["#111827", "#0f172a", "#111827"],
                    font=dict(color="white", size=13),
                    align="left",
                    height=30,
                ),
            )
        ]
    )

    fig.update_layout(
        title="Knowledge-base Evaluation Table",
        height=270,
        margin=dict(l=10, r=10, t=55, b=10),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=make_unique_key("kb_table", prefix),
    )

def display_kb_score_visuals(kb_result, prefix="kb"):
    """
    Knowledge-base evaluation visualization.
    Scores are shown out of 10 to match the evaluator scale.
    """
    if not kb_result:
        st.info("No knowledge-base result available for visualization.")
        return

    scores = kb_result.get("scores", {})

    labels = [
        "Technical Correctness",
        "Real-life Probability",
        "Theoretical Correctness",
        "Final Score",
    ]

    values = [
        scores.get("technical_correctness", 0.0),
        scores.get("real_life_probability", 0.0),
        scores.get("theoretical_correctness", 0.0),
        kb_result.get("final_score", 0.0),
    ]

    display_kb_score_table(kb_result, prefix=prefix)

    plot_interactive_bar_chart(
        title="Knowledge-base Evaluation Scores",
        labels=labels,
        values=values,
        key_prefix=f"{prefix}_kb_scores_bar",
        y_axis_mode="score10",
    )


def display_chunk_cards(chunk_results, prefix="single"):
    st.subheader("Transcript for Each Chunk")

    chunk_view = st.radio(
        "Choose chunk transcript",
        ["Raw chunk transcript", "Refined chunk transcript"],
        horizontal=True,
        key=f"{prefix}_chunk_card_view_radio",
    )

    for row in chunk_results:
        chunk_no = row["chunk"]
        total_chunks = row["total_chunks"]
        duration = row["duration_sec"]

        if chunk_view == "Raw chunk transcript":
            text = row.get("raw_text", "")
        else:
            text = row.get("refined_text", "")

        text = str(text or "").strip()

        if not text:
            text = "[No clear speech detected in this chunk]"

        with st.expander(
            f"Chunk {chunk_no}/{total_chunks} | Duration: {duration}s",
            expanded=False,
        ):
            st.write(text)


def display_results(result, show_debug: bool, prefix="single"):
    raw_full_transcript = result.get("raw_full_transcript", "")
    neat_raw_full_transcript = result.get("neat_raw_full_transcript", "")
    refined_full_transcript = result.get("refined_full_transcript", "")
    asr_quality = result["asr_quality"]
    chunk_results = result["chunks"]
    kb_result = result["kb_result"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Chunks Processed", len(chunk_results))
    c2.metric("ASR Quality", f"{asr_quality:.4f}")
    c3.metric("Transcript Words", len(refined_full_transcript.split()) if refined_full_transcript else 0)

    audio_pitch_confidence = result.get("audio_pitch_confidence", {})

    confidence_rate = audio_pitch_confidence.get("audio_pitch_confidence_rate", 0.0)
    confidence_label = audio_pitch_confidence.get(
        "audio_pitch_confidence_label",
        "Not Available",
    )
    confidence_explanation = audio_pitch_confidence.get(
        "audio_pitch_confidence_explanation",
        "Audio pitch confidence was not calculated.",
    )

    st.subheader("🎯 Audio Pitch Confidence Rate")

    p1, p2 = st.columns(2)

    p1.metric(
        "Audio Pitch Confidence",
        f"{confidence_rate * 100:.2f}%",
    )

    p2.metric(
        "Pitch Confidence Level",
        confidence_label,
    )

    st.info(confidence_explanation)

    with st.expander("Audio Pitch Confidence Breakdown", expanded=False):
        st.json(
            audio_pitch_confidence.get(
                "audio_pitch_confidence_breakdown",
                {},
            )
        )

    st.subheader("🗣️ Audio-Based Feedback")

    audio_based_feedback = result.get("audio_based_feedback", [])

    if audio_based_feedback:
        for index, item in enumerate(audio_based_feedback, start=1):
            st.write(f"{index}. {item}")
    else:
        st.write("No audio-based feedback was generated.")

    st.subheader("Complete Transcript View")

    transcript_view = st.radio(
        "Choose transcript view",
        [
            "Complete Raw Transcript",
            "Complete Refined Transcript",
            "Numbered Sentences / Key Points",
            "Raw Transcript by Chunks",
            "Refined Transcript by Chunks",
        ],
        horizontal=True,
        key=f"{prefix}_complete_transcript_view_radio",
    )

    if transcript_view == "Complete Raw Transcript":
        st.text_area(
            "Complete raw transcript from full audio",
            neat_raw_full_transcript or "[No raw transcript produced]",
            height=650,
            key=f"{prefix}_complete_raw_transcript_area",
        )

    elif transcript_view == "Complete Refined Transcript":
        st.text_area(
            "Complete refined Tamil transcript",
            refined_full_transcript or "[No refined transcript produced]",
            height=650,
            key=f"{prefix}_complete_refined_transcript_area",
        )

    elif transcript_view == "Numbered Sentences / Key Points":
        st.text_area(
            "Numbered sentences from refined transcript",
            result.get("numbered_refined_sentences", "") or "[No numbered sentences produced]",
            height=650,
            key=f"{prefix}_numbered_sentences_area",
        )

    elif transcript_view == "Raw Transcript by Chunks":
        st.text_area(
            "Raw transcript for every chunk",
            result.get("raw_transcript_by_chunks", "") or "[No raw chunk transcript produced]",
            height=750,
            key=f"{prefix}_raw_transcript_by_chunks_area",
        )

    elif transcript_view == "Refined Transcript by Chunks":
        st.text_area(
            "Refined transcript for every chunk",
            result.get("refined_transcript_by_chunks", "") or "[No refined chunk transcript produced]",
            height=750,
            key=f"{prefix}_refined_transcript_by_chunks_area",
        )

    display_chunk_cards(chunk_results, prefix=prefix)

    if show_debug:
        st.subheader("STT Status Check")

        st.dataframe(
            [
                {
                    "chunk": f"{row['chunk']}/{row['total_chunks']}",
                    "duration_sec": row["duration_sec"],
                    "raw_text_found": bool(str(row.get("raw_text", "")).strip()),
                    "raw_words": len(str(row.get("raw_text", "")).split()),
                    "stt_engine": row.get("stt_engine"),
                    "stt_quality": row.get("stt_quality"),
                    "retry_attempt": row.get("stt_retry_attempt"),
                    "stt_error": row.get("stt_error"),
                }
                for row in chunk_results
            ],
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("Debug: Raw full transcript without formatting", expanded=False):
            st.text_area(
                "Raw full transcript",
                raw_full_transcript or "[No raw transcript produced]",
                height=300,
                key=f"{prefix}_debug_raw_full_transcript_area",
            )

    st.subheader("Chunk-level Analysis")

    st.dataframe(
        [
            {
                "chunk": f"{row['chunk']}/{row['total_chunks']}",
                "duration_sec": row["duration_sec"],
                "status": row["analysis_status"],
                "clarity_score": row["clarity_score"],
                "audio_pitching_signal": row.get("confidence_reliability"),
                "confidence_reliability": row["confidence_reliability"],
                "prediction": row["prediction"],
                "model_probability": row["probability"],
                "stt_engine": row["stt_engine"],
                "stt_quality": row["stt_quality"],
                "dataset_coverage": row["dataset_coverage"],
                "technical_words": row["technical_word_count"],
                "uncertain_words": row["uncertain_word_count"],
                "repetition_score": row["repetition_score"],
                "issues": row["analysis_issues"],
                "refined_text": row["refined_text"],
            }
            for row in chunk_results
        ],
        use_container_width=True,
        hide_index=True,
    )

    if show_debug:
        st.subheader("Detailed Chunk Debug")

        st.dataframe(
            [
                {
                    "chunk": f"{row['chunk']}/{row['total_chunks']}",
                    "duration_sec": row["duration_sec"],
                    "raw_text": row["raw_text"],
                    "refined_text": row["refined_text"],
                    "matched_dataset_terms": ", ".join(row.get("dataset_matched_terms", [])),
                    "stt_error": row.get("stt_error"),
                    "retry_attempt": row.get("stt_retry_attempt"),
                }
                for row in chunk_results
            ],
            use_container_width=True,
            hide_index=True,
        )

    if kb_result:
        st.subheader("Knowledge-base Evaluation")

        scores = kb_result["scores"]
        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Technical Correctness", format_score_out_of_total(scores.get('technical_correctness', 0.0)))
        m2.metric("Real-life Probability", format_score_out_of_total(scores.get('real_life_probability', 0.0)))
        m3.metric("Theoretical Correctness", format_score_out_of_total(scores.get('theoretical_correctness', 0.0)))
        m4.metric("Final Score", format_score_out_of_total(kb_result.get("final_score", 0.0)))

        display_kb_score_visuals(kb_result, prefix=prefix)

        st.subheader("Explanations")
        for key, value in kb_result["explanations"].items():
            with st.expander(key, expanded=True):
                st.write(value)

        st.subheader("Knowledge-base Feedback")
        for item in kb_result["feedback"]:
            st.write(f"- {item}")

        display_all_match_metric_visuals(kb_result, prefix=prefix)

        if show_debug and "debug" in kb_result:
            st.subheader("Debug Details")
            st.json(kb_result["debug"])

    export_payload = {
        "raw_full_transcript": result.get("raw_full_transcript", ""),
        "neat_raw_full_transcript": result.get("neat_raw_full_transcript", ""),
        "refined_full_transcript": result.get("refined_full_transcript", ""),
        "numbered_refined_sentences": result.get("numbered_refined_sentences", ""),
        "raw_transcript_by_chunks": result.get("raw_transcript_by_chunks", ""),
        "refined_transcript_by_chunks": result.get("refined_transcript_by_chunks", ""),
        "asr_quality": asr_quality,
        "audio_pitch_confidence": result.get("audio_pitch_confidence", {}),
        "audio_based_feedback": result.get("audio_based_feedback", []),
        "chunks": chunk_results,
        "kb_result": kb_result,
    }

    st.download_button(
        "Download evaluation JSON",
        data=json.dumps(
            export_payload,
            ensure_ascii=False,
            indent=2,
            default=convert_numpy,
        ),
        file_name=f"{prefix}_tamil_pitch_evaluation.json",
        mime="application/json",
        use_container_width=True,
        key=f"{prefix}_download_evaluation_json_button",
    )

def get_safe_score(kb_result, key, default=0.0):
    try:
        return normalize_score(kb_result["scores"].get(key, default))
    except Exception:
        return default


def calculate_favourability_score(result):
    """
    Calculates a final score to decide which idea is more favourable.
    Keeps the chart meaningful without flooding every metric as /100 text.
    """

    kb_result = result.get("kb_result")

    if not kb_result:
        technical = 0.0
        real_life = 0.0
        theoretical = 0.0
        kb_final = 0.0
    else:
        technical = get_safe_score(kb_result, "technical_correctness")
        real_life = get_safe_score(kb_result, "real_life_probability")
        theoretical = get_safe_score(kb_result, "theoretical_correctness")
        kb_final = normalize_score(kb_result.get("final_score", 0.0))

    audio_pitch_confidence = normalize_score(
        result.get("audio_pitch_confidence", {}).get("audio_pitch_confidence_rate", 0.0)
    )

    asr_quality = normalize_score(result.get("asr_quality", 0.0))

    chunk_results = result.get("chunks", [])

    avg_dataset_coverage = 0.0
    avg_clarity = 0.0
    technical_strength = 0.0
    successful_chunk_ratio = 0.0

    if chunk_results:
        successful_chunks = [
            row for row in chunk_results
            if str(row.get("raw_text", "")).strip()
        ]

        successful_chunk_ratio = len(successful_chunks) / len(chunk_results)

        avg_dataset_coverage = sum(
            float(row.get("dataset_coverage", 0.0))
            for row in chunk_results
        ) / len(chunk_results)

        avg_clarity = sum(
            float(row.get("clarity_score", 0.0))
            for row in chunk_results
        ) / len(chunk_results)

        avg_technical_words = sum(
            float(row.get("technical_word_count", 0.0))
            for row in chunk_results
        ) / len(chunk_results)

        technical_strength = min(avg_technical_words / 4.0, 1.0)

    favourability_score = (
        0.24 * technical
        + 0.20 * real_life
        + 0.12 * theoretical
        + 0.14 * kb_final
        + 0.10 * audio_pitch_confidence
        + 0.07 * asr_quality
        + 0.05 * avg_dataset_coverage
        + 0.04 * avg_clarity
        + 0.02 * technical_strength
        + 0.02 * successful_chunk_ratio
    )

    return {
        "technical_correctness": round(technical, 4),
        "real_life_probability": round(real_life, 4),
        "theoretical_correctness": round(theoretical, 4),
        "knowledge_base_final_score": round(kb_final, 4),
        "audio_pitch_confidence": round(audio_pitch_confidence, 4),
        "asr_quality": round(asr_quality, 4),
        "dataset_coverage": round(avg_dataset_coverage, 4),
        "chunk_clarity": round(avg_clarity, 4),
        "technical_strength": round(technical_strength, 4),
        "successful_chunk_ratio": round(successful_chunk_ratio, 4),
        "favourability_score": round(favourability_score, 4),
    }


def decide_better_idea(score_1, score_2):
    diff = score_1["favourability_score"] - score_2["favourability_score"]

    if abs(diff) < 0.05:
        return {
            "winner": "Both ideas are very close",
            "winner_key": "tie",
            "reason": (
                "Both ideas have similar favourability scores. Choose based on novelty, cost, "
                "implementation difficulty, required resources, and expected real-world impact."
            ),
        }

    if diff > 0:
        return {
            "winner": "Idea 1 is more favourable",
            "winner_key": "idea_1",
            "reason": (
                "Idea 1 has a stronger overall score across technical correctness, real-life feasibility, "
                "knowledge-base score, audio pitch confidence, dataset coverage, and chunk clarity."
            ),
        }

    return {
        "winner": "Idea 2 is more favourable",
        "winner_key": "idea_2",
        "reason": (
            "Idea 2 has a stronger overall score across technical correctness, real-life feasibility, "
            "knowledge-base score, audio pitch confidence, dataset coverage, and chunk clarity."
        ),
    }


def generate_recommendation_text(result_1, result_2, score_1, score_2, decision):
    if decision["winner_key"] == "tie":
        return (
            "Both ideas are close. The system cannot strongly prefer one idea from the current audio and knowledge-base analysis. "
            "Choose the idea that is cheaper, easier to implement, more novel, and has better real-world usefulness."
        )

    selected = "idea_1" if decision["winner_key"] == "idea_1" else "idea_2"
    other = "idea_2" if selected == "idea_1" else "idea_1"

    selected_score = score_1 if selected == "idea_1" else score_2
    other_score = score_2 if selected == "idea_1" else score_1

    selected_name = "Idea 1" if selected == "idea_1" else "Idea 2"
    other_name = "Idea 2" if selected == "idea_1" else "Idea 1"

    strengths = []

    metric_names = {
        "technical_correctness": "technical correctness",
        "real_life_probability": "real-life feasibility",
        "theoretical_correctness": "theoretical correctness",
        "knowledge_base_final_score": "knowledge-base score",
        "audio_pitch_confidence": "audio pitch confidence",
        "asr_quality": "speech clarity",
        "dataset_coverage": "Tamil dataset alignment",
        "chunk_clarity": "chunk clarity",
        "technical_strength": "technical vocabulary strength",
        "successful_chunk_ratio": "successful transcription ratio",
    }

    for key, label in metric_names.items():
        if selected_score.get(key, 0.0) > other_score.get(key, 0.0):
            strengths.append(label)

    top_strengths = strengths[:4]
    strength_text = ", ".join(top_strengths) if top_strengths else "overall evaluation balance"

    return (
        f"{selected_name} is recommended because it performs better in {strength_text}. "
        f"It has a favourability score of {format_score_out_of_total(selected_score['favourability_score'])}, "
        f"while {other_name} has {format_score_out_of_total(other_score['favourability_score'])}. "
        f"To improve {other_name}, strengthen the technical explanation, real-life feasibility, dataset-related terms, and audio delivery clarity."
    )


def generate_dynamic_comparison_feedback(result_1, result_2, score_1, score_2, decision):
    feedback = []

    diff = round(abs(score_1["favourability_score"] - score_2["favourability_score"]), 4)

    feedback.append(
        f"The favourability score difference is {score_to_10(diff):.2f} out of 10. Result: {decision['winner']}."
    )

    metrics = [
        ("technical_correctness", "technical correctness"),
        ("real_life_probability", "real-life feasibility"),
        ("theoretical_correctness", "theoretical correctness"),
        ("knowledge_base_final_score", "knowledge-base final score"),
        ("audio_pitch_confidence", "audio pitch confidence"),
        ("asr_quality", "speech/transcription clarity"),
        ("dataset_coverage", "Tamil dataset alignment"),
        ("chunk_clarity", "chunk clarity"),
        ("technical_strength", "technical vocabulary strength"),
        ("successful_chunk_ratio", "successful transcription ratio"),
    ]

    for key, label in metrics:
        value_1 = score_1.get(key, 0.0)
        value_2 = score_2.get(key, 0.0)
        gap = abs(value_1 - value_2)

        if gap < 0.03:
            feedback.append(f"Both ideas are almost equal in {label}.")
        elif value_1 > value_2:
            feedback.append(f"Idea 1 is stronger in {label} by {score_to_10(gap):.2f} out of 10.")
        else:
            feedback.append(f"Idea 2 is stronger in {label} by {score_to_10(gap):.2f} out of 10.")

    idea_1_words = len(result_1.get("refined_full_transcript", "").split())
    idea_2_words = len(result_2.get("refined_full_transcript", "").split())

    if idea_1_words > idea_2_words:
        feedback.append(
            f"Idea 1 has a more detailed refined transcript with {idea_1_words} words compared to Idea 2 with {idea_2_words} words."
        )
    elif idea_2_words > idea_1_words:
        feedback.append(
            f"Idea 2 has a more detailed refined transcript with {idea_2_words} words compared to Idea 1 with {idea_1_words} words."
        )
    else:
        feedback.append("Both ideas have the same refined transcript length.")

    idea_1_feedback = result_1.get("audio_based_feedback", [])
    idea_2_feedback = result_2.get("audio_based_feedback", [])

    if idea_1_feedback:
        feedback.append(f"Idea 1 audio analysis note: {idea_1_feedback[0]}")

    if idea_2_feedback:
        feedback.append(f"Idea 2 audio analysis note: {idea_2_feedback[0]}")

    feedback.append(generate_recommendation_text(result_1, result_2, score_1, score_2, decision))

    return feedback



def display_favourability_contribution_table(score_1, score_2):
    """
    Shows Idea 1 and Idea 2 favourability contribution as a modern interactive table.
    Pie charts are intentionally removed.
    """
    contribution_rows = [
        ("Technical", score_1["technical_correctness"] * 0.24, score_2["technical_correctness"] * 0.24),
        ("Real-life", score_1["real_life_probability"] * 0.20, score_2["real_life_probability"] * 0.20),
        ("Theory", score_1["theoretical_correctness"] * 0.12, score_2["theoretical_correctness"] * 0.12),
        ("KB Final", score_1["knowledge_base_final_score"] * 0.14, score_2["knowledge_base_final_score"] * 0.14),
        ("Audio Confidence", score_1["audio_pitch_confidence"] * 0.10, score_2["audio_pitch_confidence"] * 0.10),
        ("ASR", score_1["asr_quality"] * 0.07, score_2["asr_quality"] * 0.07),
        ("Dataset", score_1["dataset_coverage"] * 0.05, score_2["dataset_coverage"] * 0.05),
        ("Chunk Clarity", score_1["chunk_clarity"] * 0.04, score_2["chunk_clarity"] * 0.04),
        ("Technical Strength", score_1["technical_strength"] * 0.02, score_2["technical_strength"] * 0.02),
        ("Successful Chunks", score_1["successful_chunk_ratio"] * 0.02, score_2["successful_chunk_ratio"] * 0.02),
    ]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[
                        "<b>Contribution Factor</b>",
                        "<b>Idea 1 Contribution</b>",
                        "<b>Idea 2 Contribution</b>",
                        "<b>Stronger</b>",
                    ],
                    fill_color="#1f2937",
                    font=dict(color="white", size=14),
                    align="left",
                    height=34,
                ),
                cells=dict(
                    values=[
                        [row[0] for row in contribution_rows],
                        [f"{score_to_10(row[1]):.2f} / 10" for row in contribution_rows],
                        [f"{score_to_10(row[2]):.2f} / 10" for row in contribution_rows],
                        [
                            "Idea 1" if row[1] > row[2] else "Idea 2" if row[2] > row[1] else "Equal"
                            for row in contribution_rows
                        ],
                    ],
                    fill_color=["#111827", "#0f172a", "#111827", "#0f172a"],
                    font=dict(color="white", size=13),
                    align="left",
                    height=30,
                ),
            )
        ]
    )

    fig.update_layout(
        title="Favourability Contribution Table",
        height=430,
        margin=dict(l=10, r=10, t=55, b=10),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="favourability_contribution_table_unique",
    )


def display_comparison_visuals(score_1, score_2):
    """
    Keeps the grouped comparison chart style and fixes duplicate Plotly IDs using a key.
    """
    st.subheader("📊 Visual Comparison")

    labels = [
        "Technical Correctness",
        "Real-life Probability",
        "Theoretical Correctness",
        "KB Final Score",
        "Audio Pitch Confidence",
        "ASR Quality",
        "Dataset Coverage",
        "Chunk Clarity",
        "Technical Strength",
        "Successful Chunks",
        "Favourability Score",
    ]

    idea_1_values = [
        score_1["technical_correctness"],
        score_1["real_life_probability"],
        score_1["theoretical_correctness"],
        score_1["knowledge_base_final_score"],
        score_1["audio_pitch_confidence"],
        score_1["asr_quality"],
        score_1["dataset_coverage"],
        score_1["chunk_clarity"],
        score_1["technical_strength"],
        score_1["successful_chunk_ratio"],
        score_1["favourability_score"],
    ]

    idea_2_values = [
        score_2["technical_correctness"],
        score_2["real_life_probability"],
        score_2["theoretical_correctness"],
        score_2["knowledge_base_final_score"],
        score_2["audio_pitch_confidence"],
        score_2["asr_quality"],
        score_2["dataset_coverage"],
        score_2["chunk_clarity"],
        score_2["technical_strength"],
        score_2["successful_chunk_ratio"],
        score_2["favourability_score"],
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=labels,
            y=[score_to_10(v) for v in idea_1_values],
            name="Idea 1",
            hovertemplate="<b>%{x}</b><br>Idea 1: %{y:.2f}/10<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            x=labels,
            y=[score_to_10(v) for v in idea_2_values],
            name="Idea 2",
            hovertemplate="<b>%{x}</b><br>Idea 2: %{y:.2f}/10<extra></extra>",
        )
    )

    fig.update_layout(
        title="Idea 1 vs Idea 2 Score Comparison",
        yaxis_title="Score out of 10",
        xaxis_title="Metric",
        yaxis=dict(range=[0, 10]),
        barmode="group",
        height=560,
        margin=dict(l=20, r=20, t=60, b=140),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="comparison_grouped_score_chart_unique",
    )

    st.subheader("📋 Favourability Contribution Table")
    display_favourability_contribution_table(score_1, score_2)


def display_comparison_results(result_1, result_2, show_debug: bool):
    score_1 = calculate_favourability_score(result_1)
    score_2 = calculate_favourability_score(result_2)

    decision = decide_better_idea(score_1, score_2)

    comparison_feedback = generate_dynamic_comparison_feedback(
        result_1=result_1,
        result_2=result_2,
        score_1=score_1,
        score_2=score_2,
        decision=decision,
    )

    st.header("⚖️ Two Idea Comparison Result")

    score_col_1, score_col_2 = st.columns(2)

    score_col_1.metric(
        "Idea 1 Favourability Score",
        format_score_out_of_total(score_1["favourability_score"]),
    )

    score_col_2.metric(
        "Idea 2 Favourability Score",
        format_score_out_of_total(score_2["favourability_score"]),
    )

    if decision["winner_key"] == "tie":
        st.warning(decision["winner"])
    else:
        st.success(decision["winner"])

    st.write(decision["reason"])

    recommendation = generate_recommendation_text(
        result_1=result_1,
        result_2=result_2,
        score_1=score_1,
        score_2=score_2,
        decision=decision,
    )

    st.subheader("✅ Final Recommendation")
    st.info(recommendation)

    st.subheader("Comparison Scores")

    comparison_rows = [
        {"Metric": "Technical Correctness", "Idea 1": format_score_out_of_total(score_1['technical_correctness']), "Idea 2": format_score_out_of_total(score_2['technical_correctness'])},
        {"Metric": "Real-life Probability", "Idea 1": format_score_out_of_total(score_1['real_life_probability']), "Idea 2": format_score_out_of_total(score_2['real_life_probability'])},
        {"Metric": "Theoretical Correctness", "Idea 1": format_score_out_of_total(score_1['theoretical_correctness']), "Idea 2": format_score_out_of_total(score_2['theoretical_correctness'])},
        {"Metric": "Knowledge-base Final Score", "Idea 1": format_score_out_of_total(score_1['knowledge_base_final_score']), "Idea 2": format_score_out_of_total(score_2['knowledge_base_final_score'])},
        {"Metric": "Audio Pitch Confidence", "Idea 1": format_score_out_of_total(score_1['audio_pitch_confidence']), "Idea 2": format_score_out_of_total(score_2['audio_pitch_confidence'])},
        {"Metric": "ASR Quality", "Idea 1": format_score_out_of_total(score_1['asr_quality']), "Idea 2": format_score_out_of_total(score_2['asr_quality'])},
        {"Metric": "Dataset Coverage", "Idea 1": format_score_out_of_total(score_1['dataset_coverage']), "Idea 2": format_score_out_of_total(score_2['dataset_coverage'])},
        {"Metric": "Chunk Clarity", "Idea 1": format_score_out_of_total(score_1['chunk_clarity']), "Idea 2": format_score_out_of_total(score_2['chunk_clarity'])},
        {"Metric": "Technical Strength", "Idea 1": format_score_out_of_total(score_1['technical_strength']), "Idea 2": format_score_out_of_total(score_2['technical_strength'])},
        {"Metric": "Successful Chunk Ratio", "Idea 1": format_score_out_of_total(score_1['successful_chunk_ratio']), "Idea 2": format_score_out_of_total(score_2['successful_chunk_ratio'])},
        {"Metric": "Favourability Score", "Idea 1": format_score_out_of_total(score_1['favourability_score']), "Idea 2": format_score_out_of_total(score_2['favourability_score'])},
    ]

    st.dataframe(comparison_rows, use_container_width=True, hide_index=True)

    display_comparison_visuals(score_1, score_2)

    st.subheader("🧠 Dynamic Comparison Feedback")

    for index, item in enumerate(comparison_feedback, start=1):
        st.write(f"{index}. {item}")

    with st.expander("📊 Knowledge-base Evaluation Visuals", expanded=False):
        kb_col_1, kb_col_2 = st.columns(2)

        with kb_col_1:
            display_kb_score_table(
                result_1.get("kb_result"),
                prefix="comparison_idea_1_kb_table_only",
            )

        with kb_col_2:
            display_kb_score_table(
                result_2.get("kb_result"),
                prefix="comparison_idea_2_kb_table_only",
            )

    st.subheader("Idea 1 Debug Match Metric Visuals")
    display_all_match_metric_visuals(
        result_1.get("kb_result"),
        prefix="comparison_idea_1_debug",
    )

    st.subheader("Idea 2 Debug Match Metric Visuals")
    display_all_match_metric_visuals(
        result_2.get("kb_result"),
        prefix="comparison_idea_2_debug",
    )

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Idea 1 Refined Transcript")
        st.text_area(
            "Idea 1 complete refined transcript",
            result_1.get("refined_full_transcript", "") or "[No refined transcript]",
            height=350,
            key="idea_1_comparison_refined_transcript",
        )

        idea_1_confidence = result_1.get("audio_pitch_confidence", {})
        st.metric(
            "Idea 1 Audio Pitch Confidence",
            format_score_out_of_total(idea_1_confidence.get('audio_pitch_confidence_rate', 0.0)),
        )

    with c2:
        st.subheader("Idea 2 Refined Transcript")
        st.text_area(
            "Idea 2 complete refined transcript",
            result_2.get("refined_full_transcript", "") or "[No refined transcript]",
            height=350,
            key="idea_2_comparison_refined_transcript",
        )

        idea_2_confidence = result_2.get("audio_pitch_confidence", {})
        st.metric(
            "Idea 2 Audio Pitch Confidence",
            format_score_out_of_total(idea_2_confidence.get('audio_pitch_confidence_rate', 0.0)),
        )

    st.subheader("Detailed Individual Results")

    detail_choice = st.radio(
        "Choose detailed result to view",
        ["Idea 1 Details", "Idea 2 Details"],
        horizontal=True,
        key="comparison_detail_result_radio",
    )

    if detail_choice == "Idea 1 Details":
        display_results(result=result_1, show_debug=show_debug, prefix="idea_1_details")
    else:
        display_results(result=result_2, show_debug=show_debug, prefix="idea_2_details")

    comparison_payload = {
        "idea_1_scores": score_1,
        "idea_2_scores": score_2,
        "decision": decision,
        "recommendation": recommendation,
        "comparison_feedback": comparison_feedback,
        "idea_1_result": result_1,
        "idea_2_result": result_2,
    }

    st.download_button(
        "Download comparison JSON",
        data=json.dumps(
            comparison_payload,
            ensure_ascii=False,
            indent=2,
            default=convert_numpy,
        ),
        file_name="two_idea_comparison_result.json",
        mime="application/json",
        use_container_width=True,
        key="download_two_idea_comparison_json",
    )

def validate_common_paths(model_path, kb_path, text_dataset_path):
    if not os.path.exists(model_path):
        st.error(f"Confidence model not found: {model_path}")
        return False

    if not os.path.exists(kb_path):
        st.error(f"Knowledge base not found: {kb_path}")
        return False

    if not os.path.exists(text_dataset_path):
        st.error(f"Tamil text dataset not found: {text_dataset_path}")
        return False

    return True


st.set_page_config(
    page_title="Tamil Idea Pitch Evaluator",
    page_icon="🎙️",
    layout="wide",
)

if "evaluation_result" not in st.session_state:
    st.session_state.evaluation_result = None

if "comparison_result" not in st.session_state:
    st.session_state.comparison_result = None

if "last_uploaded_audio_name" not in st.session_state:
    st.session_state.last_uploaded_audio_name = None


st.title("🎙️ Tamil Idea Pitch Evaluation System")
st.caption(
    "Primary single idea evaluation + two-idea comparison with interactive graph visuals and debug metric charts."
)

st.markdown(
    """
    <style>
    .stTextArea textarea {
        line-height: 1.9 !important;
        font-size: 16px !important;
        white-space: pre-wrap !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Settings")

    if st.button("Clear Previous Results / Cache"):
        st.session_state.clear()
        st.cache_resource.clear()
        st.rerun()

    app_mode = st.radio(
        "Choose Evaluation Mode",
        ["Single Idea Evaluation", "Compare Two Ideas"],
    )

    model_path = st.text_input("Confidence model path", MODEL_PATH)
    kb_path = st.text_input("Knowledge base path", KB_PATH)

    text_dataset_path = st.text_input(
        "Tamil text dataset path",
        TEXT_DATASET_PATH,
    )

    prefer_torch = st.checkbox(
        "Try Torch / Whisper first",
        value=False,
        help="Keep this OFF if torch gives DLL errors.",
    )

    show_debug = st.checkbox("Show debug details", value=True)

    st.info(
        "Primary mode evaluates one Tamil idea pitch. Compare mode evaluates two pitches and recommends the more favourable idea."
    )


if app_mode == "Single Idea Evaluation":
    st.header("🧠 Single Tamil Idea Evaluation")

    uploaded_audio = st.file_uploader(
        "Upload Tamil audio file",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        accept_multiple_files=False,
        key="single_uploaded_audio",
    )

    if uploaded_audio is not None:
        st.audio(uploaded_audio)

        if st.session_state.last_uploaded_audio_name != uploaded_audio.name:
            st.session_state.evaluation_result = None
            st.session_state.last_uploaded_audio_name = uploaded_audio.name

    run_button = st.button(
        "Run Evaluation",
        type="primary",
        use_container_width=True,
        key="single_run_evaluation_button",
    )

    if run_button:
        if uploaded_audio is None:
            st.error("Please upload an audio file first.")

        elif validate_common_paths(model_path, kb_path, text_dataset_path):
            temp_audio_path = None

            try:
                temp_audio_path = save_uploaded_audio(uploaded_audio)

                with st.spinner("Transcribing full audio and preparing complete transcripts..."):
                    result = run_pipeline(
                        temp_audio_path,
                        model_path,
                        kb_path,
                        text_dataset_path,
                        prefer_torch=prefer_torch,
                    )

                st.session_state.evaluation_result = result
                st.session_state.last_uploaded_audio_name = uploaded_audio.name

            except Exception as e:
                st.error(str(e))
                st.code(traceback.format_exc(), language="python")

            finally:
                if temp_audio_path and os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)

    if st.session_state.evaluation_result is not None:
        display_results(
            result=st.session_state.evaluation_result,
            show_debug=show_debug,
            prefix="single",
        )


elif app_mode == "Compare Two Ideas":
    st.header("⚖️ Compare Two Tamil Idea Pitches")

    st.info(
        "Upload two Tamil audio pitches. The system will transcribe, refine, evaluate, and compare both ideas using the same output structure as the primary website."
    )

    idea_1_audio = st.file_uploader(
        "Upload Idea 1 Tamil audio",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        accept_multiple_files=False,
        key="idea_1_audio_uploader",
    )

    idea_2_audio = st.file_uploader(
        "Upload Idea 2 Tamil audio",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        accept_multiple_files=False,
        key="idea_2_audio_uploader",
    )

    audio_col_1, audio_col_2 = st.columns(2)

    with audio_col_1:
        if idea_1_audio is not None:
            st.write("Idea 1 Audio")
            st.audio(idea_1_audio)

    with audio_col_2:
        if idea_2_audio is not None:
            st.write("Idea 2 Audio")
            st.audio(idea_2_audio)

    compare_button = st.button(
        "Compare Both Ideas",
        type="primary",
        use_container_width=True,
        key="compare_both_ideas_button",
    )

    if compare_button:
        if idea_1_audio is None or idea_2_audio is None:
            st.error("Please upload both Idea 1 and Idea 2 audio files.")

        elif validate_common_paths(model_path, kb_path, text_dataset_path):
            idea_1_temp = None
            idea_2_temp = None

            try:
                idea_1_temp = save_uploaded_audio(idea_1_audio)
                idea_2_temp = save_uploaded_audio(idea_2_audio)

                with st.spinner("Analyzing Idea 1..."):
                    result_1 = run_pipeline(
                        idea_1_temp,
                        model_path,
                        kb_path,
                        text_dataset_path,
                        prefer_torch=prefer_torch,
                    )

                with st.spinner("Analyzing Idea 2..."):
                    result_2 = run_pipeline(
                        idea_2_temp,
                        model_path,
                        kb_path,
                        text_dataset_path,
                        prefer_torch=prefer_torch,
                    )

                st.session_state.comparison_result = {
                    "idea_1": result_1,
                    "idea_2": result_2,
                }

            except Exception as e:
                st.error(str(e))
                st.code(traceback.format_exc(), language="python")

            finally:
                if idea_1_temp and os.path.exists(idea_1_temp):
                    os.remove(idea_1_temp)

                if idea_2_temp and os.path.exists(idea_2_temp):
                    os.remove(idea_2_temp)

    if st.session_state.comparison_result is not None:
        display_comparison_results(
            result_1=st.session_state.comparison_result["idea_1"],
            result_2=st.session_state.comparison_result["idea_2"],
            show_debug=show_debug,
        )


with st.expander("How this website works"):
    st.markdown(
        """
        **Single Idea Evaluation**
        1. Upload one Tamil audio pitch.
        2. The system splits the full audio into chunks.
        3. Each chunk is transcribed and refined.
        4. The website shows complete raw transcript, complete refined transcript, numbered key points, and chunk-wise transcript.
        5. The idea is evaluated using the Tamil knowledge base.
        6. Audio pitch confidence and audio-based feedback are generated from actual chunk analysis.

        **Compare Two Ideas**
        1. Upload two Tamil audio pitches.
        2. Each idea is processed using the same pipeline as the primary website.
        3. Both ideas are compared using technical correctness, real-life probability, theoretical correctness, knowledge-base final score, audio pitch confidence, and ASR quality.
        4. The system recommends which idea is more favourable.
        """
    )
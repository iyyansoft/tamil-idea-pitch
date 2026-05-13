import csv
import json
import os
import re
from difflib import SequenceMatcher


TAMIL_SENTENCE_ENDINGS = [
    "ஆகும்", "இருக்கும்", "உள்ளது", "முடியும்", "வேண்டும்",
    "செய்யும்", "செயல்படும்", "உதவும்", "கிடைக்கும்",
    "குறையும்", "அதிகரிக்கும்", "பயன்படும்", "சாத்தியம்",
    "தேவைப்படும்", "உருவாகும்", "நடைபெறும்", "காணலாம்",
    "புரியும்", "முடிகிறது", "இருக்கிறது", "செய்கிறது",
    "உதவுகிறது", "பயன்படுகிறது", "வருகிறது", "செல்கிறது",
    "கொடுக்கிறது", "பெறுகிறது", "அளிக்கிறது", "கட்டுப்படுத்துகிறது",
    "இயக்குகிறது", "சேமிக்கிறது", "கண்டறிகிறது", "அனுப்புகிறது",
    "குறைவாகும்", "அதிகமாகும்", "அமையும்", "வழங்கும்",
    "செய்யலாம்", "பயன்படுத்தலாம்", "இயக்கலாம்", "கண்டறியலாம்",
    "சேமிக்கலாம்", "உருவாக்கலாம்",
]

TAMIL_CONNECTORS = [
    "ஆனால்", "மேலும்", "அதனால்", "இதனால்", "எனவே",
    "ஏனெனில்", "பின்னர்", "அடுத்து", "முதலில்",
    "இரண்டாவது", "மூன்றாவது", "கடைசியாக",
    "அதேபோல்", "அதுமட்டுமல்லாமல்", "இருப்பினும்",
    "உதாரணமாக", "முக்கியமாக", "இதற்காக", "அதற்காக",
    "இதன் மூலம்", "அதன் மூலம்",
]

QUESTION_HINTS = ["எப்படி", "ஏன்", "என்ன", "எங்கு", "எப்போது", "யார்"]

UNCERTAIN_WORDS = [
    "தெரியவில்லை", "நினைக்கிறேன்", "ஏதோ", "இருக்கலாம்",
    "போலும்", "சுமார்", "maybe", "probably", "not sure",
]

TECHNICAL_HINT_WORDS = [
    "சூரிய", "ஆற்றல்", "மின்சாரம்", "மோட்டார்", "சென்சார்",
    "பேட்டரி", "தரவு", "அமைப்பு", "தொழில்நுட்பம்",
    "இயந்திரம்", "தானியங்கி", "கட்டுப்பாடு", "மாதிரி",
    "செயல்முறை", "அல்காரிதம்", "வெப்பநிலை", "அழுத்தம்",
    "நீர்", "பம்ப்", "மின்னழுத்தம்", "மின்கலம்", "கருவி",
    "செயலி", "கணினி", "மென்பொருள்", "வன்பொருள்",
    "விவசாய", "மருத்துவ", "கல்வி", "பாதுகாப்பு",
    "தகவல்", "தொடர்பு", "சாதனம்",
]


def normalize_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("\u200c", "")
    text = text.replace("\u200d", "")
    text = text.replace("\ufeff", "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([,.!?।]){2,}", r"\1", text)
    text = re.sub(r"\s+([,.!?।])", r"\1", text)
    text = re.sub(r"([,.!?।])([^\s])", r"\1 \2", text)
    return text.strip()


def tamil_word_tokenize(text: str):
    return re.findall(r"[\u0B80-\u0BFFa-zA-Z0-9]+", normalize_text(text))


def read_csv_texts(path: str):
    texts = []

    if not path or not os.path.exists(path):
        return texts

    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                value = (
                    row.get("text")
                    or row.get("tamil")
                    or row.get("transcript")
                    or row.get("sentence")
                    or row.get("clean_text")
                    or row.get("raw_text")
                    or row.get("recognized_text")
                    or ""
                )

                value = normalize_text(value)

                if value:
                    texts.append(value)

    except Exception:
        return texts

    return texts


def read_json_texts(path: str):
    texts = []

    if not path or not os.path.exists(path):
        return texts

    try:
        with open(path, "r", encoding="utf-8") as f:
            if path.lower().endswith(".jsonl"):
                data = []
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            else:
                data = json.load(f)
    except Exception:
        return texts

    records = []

    if isinstance(data, list):
        records = data

    elif isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                records.extend(value)
            elif isinstance(value, dict):
                records.append(value)

    for item in records:
        if not isinstance(item, dict):
            continue

        combined = " ".join(
            str(item.get(key, ""))
            for key in [
                "text", "tamil", "transcript", "sentence",
                "reasoning", "english", "concept", "category",
                "metric", "clean_text", "raw_text", "recognized_text",
            ]
        )

        combined = normalize_text(combined)

        if combined:
            texts.append(combined)

    return texts


def load_reference_texts(text_dataset_path: str = None, kb_path: str = None):
    texts = []

    for path in [text_dataset_path, kb_path]:
        if not path:
            continue

        lower = path.lower()

        if lower.endswith(".csv"):
            texts.extend(read_csv_texts(path))

        elif lower.endswith(".json") or lower.endswith(".jsonl"):
            texts.extend(read_json_texts(path))

    clean = []
    seen = set()

    for text in texts:
        text = normalize_text(text)

        if text and text not in seen:
            clean.append(text)
            seen.add(text)

    return clean


def build_dataset_terms(reference_texts, max_terms=1200):
    freq = {}

    for text in reference_texts:
        for word in tamil_word_tokenize(text):
            word = word.strip()

            if len(word) < 3:
                continue

            freq[word] = freq.get(word, 0) + 1

    sorted_terms = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [term for term, _ in sorted_terms[:max_terms]]


def add_connector_commas(text: str) -> str:
    text = normalize_text(text)

    for connector in TAMIL_CONNECTORS:
        text = re.sub(
            rf"\s+({re.escape(connector)})\s+",
            rf", \1 ",
            text,
        )

    return normalize_text(text)


def sentence_punctuation(sentence: str) -> str:
    sentence = normalize_text(sentence)

    if any(q in sentence for q in QUESTION_HINTS):
        return "?"

    return "."


def split_tamil_sentences(text: str, min_words=5, max_words=16):
    text = add_connector_commas(text)

    if not text:
        return []

    words = text.split()
    sentences = []
    current = []

    for word in words:
        clean_word = word.strip(" ,.!?।")
        current.append(clean_word)

        should_end = False

        if clean_word in TAMIL_SENTENCE_ENDINGS and len(current) >= min_words:
            should_end = True

        if len(current) >= max_words:
            should_end = True

        if should_end:
            sentence = " ".join(current).strip(" ,.!?।")
            sentence = normalize_text(sentence)

            if sentence:
                sentence += sentence_punctuation(sentence)
                sentences.append(sentence)

            current = []

    if current:
        sentence = " ".join(current).strip(" ,.!?।")
        sentence = normalize_text(sentence)

        if sentence:
            sentence += sentence_punctuation(sentence)
            sentences.append(sentence)

    cleaned = []

    for sentence in sentences:
        sentence = re.sub(r"^,\s*", "", sentence)
        sentence = normalize_text(sentence)

        if sentence:
            cleaned.append(sentence)

    return cleaned


def refine_chunk_text(raw_text: str) -> str:
    sentences = split_tamil_sentences(raw_text, min_words=4, max_words=14)
    return " ".join(sentences)


def refine_full_transcript(raw_text: str) -> str:
    sentences = split_tamil_sentences(raw_text, min_words=5, max_words=16)
    return format_sentences_as_paragraphs(sentences, sentences_per_paragraph=2)


def format_sentences_as_paragraphs(sentences, sentences_per_paragraph=2) -> str:
    if isinstance(sentences, str):
        sentences = re.split(r"(?<=[.!?।])\s+", sentences)

    sentences = [normalize_text(s) for s in sentences if normalize_text(s)]

    paragraphs = []
    temp = []

    for sentence in sentences:
        temp.append(sentence)

        if len(temp) >= sentences_per_paragraph:
            paragraphs.append(" ".join(temp))
            temp = []

    if temp:
        paragraphs.append(" ".join(temp))

    return "\n\n".join(paragraphs)


def format_raw_full_transcript(raw_transcript: str) -> str:
    """
    Complete raw transcript, but displayed neatly.
    Does not remove or skip words.
    """

    raw_transcript = normalize_text(raw_transcript)

    if not raw_transcript:
        return "[No raw transcript produced]"

    words = raw_transcript.split()
    lines = []
    current = []

    for word in words:
        current.append(word)

        if len(current) >= 18:
            lines.append(" ".join(current))
            current = []

    if current:
        lines.append(" ".join(current))

    paragraphs = []
    temp = []

    for line in lines:
        temp.append(line)

        if len(temp) == 3:
            paragraphs.append("\n".join(temp))
            temp = []

    if temp:
        paragraphs.append("\n".join(temp))

    return "\n\n".join(paragraphs)


def format_numbered_sentences(transcript: str) -> str:
    transcript = normalize_text(transcript)
    sentences = re.split(r"(?<=[.!?।])\s+", transcript)
    sentences = [normalize_text(s) for s in sentences if normalize_text(s)]

    if not sentences:
        return "[No numbered sentences produced]"

    return "\n".join(
        f"{i}. {sentence}"
        for i, sentence in enumerate(sentences, start=1)
    )


def build_chunk_transcript(chunk_results, use_refined=True):
    blocks = []

    for row in chunk_results:
        chunk_no = row.get("chunk", "")
        total_chunks = row.get("total_chunks", "")
        duration = row.get("duration_sec", "")

        if use_refined:
            text = row.get("refined_text", "")
            label = "Refined"
        else:
            text = row.get("raw_text", "")
            label = "Raw"

        text = normalize_text(text)

        if not text:
            text = "[No clear speech detected in this chunk]"

        block = (
            f"Chunk {chunk_no}/{total_chunks} | {label} | Duration: {duration}s\n"
            f"{text}"
        )

        blocks.append(block)

    return "\n\n".join(blocks)


def format_raw_transcript_by_chunks(chunk_results) -> str:
    return build_chunk_transcript(chunk_results, use_refined=False)


def format_refined_transcript_by_chunks(chunk_results) -> str:
    return build_chunk_transcript(chunk_results, use_refined=True)


def match_dataset_terms(text: str, dataset_terms, limit=20):
    text = normalize_text(text).lower()
    matched = []

    for term in dataset_terms:
        if term.lower() in text:
            matched.append(term)

        if len(matched) >= limit:
            break

    return matched


def sequence_similarity(a: str, b: str) -> float:
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def analyze_chunk_with_dataset(
    raw_text: str,
    refined_text: str,
    previous_refined_text: str | None,
    stt_quality: float,
    model_probability: float | None,
    dataset_terms,
):
    raw_text = normalize_text(raw_text)
    refined_text = normalize_text(refined_text)

    words = tamil_word_tokenize(refined_text)
    word_count = len(words)
    char_count = len(refined_text)

    matched_terms = match_dataset_terms(refined_text, dataset_terms)
    dataset_coverage = min(len(matched_terms) / 8.0, 1.0)

    technical_word_count = sum(
        1 for word in TECHNICAL_HINT_WORDS
        if word.lower() in refined_text.lower()
    )

    uncertain_word_count = sum(
        1 for word in UNCERTAIN_WORDS
        if word.lower() in refined_text.lower()
    )

    repetition_score = (
        sequence_similarity(refined_text, previous_refined_text)
        if previous_refined_text
        else 0.0
    )

    probability = model_probability if model_probability is not None else 0.5
    length_score = min(word_count / 15.0, 1.0)
    technical_score = min(technical_word_count / 4.0, 1.0)

    clarity_score = (
        0.25 * stt_quality
        + 0.25 * probability
        + 0.20 * length_score
        + 0.20 * dataset_coverage
        + 0.10 * technical_score
        - 0.15 * repetition_score
        - 0.10 * min(uncertain_word_count / 3.0, 1.0)
    )

    clarity_score = round(max(0.0, min(1.0, clarity_score)), 4)

    issues = []

    if not refined_text:
        issues.append("No clear text detected")

    if refined_text and word_count < 5:
        issues.append("Very short chunk")

    if stt_quality < 0.45:
        issues.append("Low STT quality")

    if uncertain_word_count > 0:
        issues.append("Uncertain wording detected")

    if repetition_score > 0.75:
        issues.append("Repeated content")

    if refined_text and not matched_terms:
        issues.append("No strong match with Tamil dataset concepts")

    if not issues:
        issues.append("Clear and useful chunk")

    if clarity_score >= 0.75:
        status = "Excellent"
    elif clarity_score >= 0.55:
        status = "Good"
    elif clarity_score >= 0.35:
        status = "Needs Review"
    else:
        status = "Weak"

    return {
        "refined_text": refined_text,
        "word_count": word_count,
        "char_count": char_count,
        "dataset_matched_terms": matched_terms,
        "dataset_coverage": round(dataset_coverage, 4),
        "technical_word_count": technical_word_count,
        "uncertain_word_count": uncertain_word_count,
        "repetition_score": round(repetition_score, 4),
        "clarity_score": clarity_score,
        "analysis_status": status,
        "analysis_issues": "; ".join(issues),
    }
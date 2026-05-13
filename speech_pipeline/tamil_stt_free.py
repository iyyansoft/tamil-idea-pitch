import os
import tempfile


TARGET_SAMPLE_RATE = 16000
TARGET_DBFS = -20.0


def normalize_audiosegment(audio_segment):
    """
    Converts audio to mono, 16kHz and boosts quiet audio.
    This helps prevent empty STT output.
    """

    audio_segment = audio_segment.set_channels(1)
    audio_segment = audio_segment.set_frame_rate(TARGET_SAMPLE_RATE)

    try:
        if audio_segment.dBFS != float("-inf"):
            gain_needed = TARGET_DBFS - audio_segment.dBFS

            if gain_needed > 0:
                gain_needed = min(gain_needed, 12)

            audio_segment = audio_segment.apply_gain(gain_needed)
    except Exception:
        pass

    return audio_segment


def audiosegment_to_temp_wav(audio_segment):
    """
    Converts pydub AudioSegment to temporary WAV.
    SpeechRecognition works best with WAV.
    """

    audio_segment = normalize_audiosegment(audio_segment)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav_path = temp_file.name
    temp_file.close()

    audio_segment.export(wav_path, format="wav")
    return wav_path


def transcribe_google_tamil(audio_segment):
    """
    Tamil STT using Google Speech Recognition.
    No torch required.
    Requires internet.
    """

    try:
        import speech_recognition as sr
    except ImportError:
        return {
            "text": "",
            "stt_engine": "google_speech_recognition",
            "language_code": "ta-IN",
            "whisper_quality": 0.0,
            "error": "SpeechRecognition not installed. Run: pip install SpeechRecognition",
        }

    recognizer = sr.Recognizer()
    wav_path = None

    try:
        if audio_segment is None:
            return {
                "text": "",
                "stt_engine": "google_speech_recognition",
                "language_code": "ta-IN",
                "whisper_quality": 0.0,
                "error": "Audio segment is None",
            }

        if len(audio_segment) < 300:
            return {
                "text": "",
                "stt_engine": "google_speech_recognition",
                "language_code": "ta-IN",
                "whisper_quality": 0.0,
                "error": "Audio chunk too short",
            }

        if audio_segment.dBFS == float("-inf"):
            return {
                "text": "",
                "stt_engine": "google_speech_recognition",
                "language_code": "ta-IN",
                "whisper_quality": 0.0,
                "error": "Silent audio chunk",
            }

        wav_path = audiosegment_to_temp_wav(audio_segment)

        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(
            audio_data,
            language="ta-IN",
        )

        text = str(text or "").strip()

        return {
            "text": text,
            "stt_engine": "google_speech_recognition",
            "language_code": "ta-IN",
            "whisper_quality": 0.75 if text else 0.0,
            "error": None if text else "Google STT returned empty text",
        }

    except sr.UnknownValueError:
        return {
            "text": "",
            "stt_engine": "google_speech_recognition",
            "language_code": "ta-IN",
            "whisper_quality": 0.0,
            "error": "Google STT could not understand this chunk",
        }

    except sr.RequestError as e:
        return {
            "text": "",
            "stt_engine": "google_speech_recognition",
            "language_code": "ta-IN",
            "whisper_quality": 0.0,
            "error": f"Google STT request failed: {e}",
        }

    except Exception as e:
        return {
            "text": "",
            "stt_engine": "google_speech_recognition",
            "language_code": "ta-IN",
            "whisper_quality": 0.0,
            "error": f"Google STT failed: {e}",
        }

    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass


def transcribe_whisper_lazy(audio_segment):
    """
    Optional Whisper STT.
    Only used if prefer_torch=True.
    If Whisper/Torch fails, app will not crash.
    """

    wav_path = None

    try:
        import whisper
    except Exception as e:
        return {
            "text": "",
            "stt_engine": "whisper_failed",
            "language_code": "ta",
            "whisper_quality": 0.0,
            "error": f"Whisper/Torch not available: {e}",
        }

    try:
        wav_path = audiosegment_to_temp_wav(audio_segment)

        model = whisper.load_model("small")

        result = model.transcribe(
            wav_path,
            language="ta",
            fp16=False,
        )

        text = str(result.get("text", "")).strip()

        return {
            "text": text,
            "stt_engine": "whisper",
            "language_code": "ta",
            "whisper_quality": 0.85 if text else 0.0,
            "error": None if text else "Whisper returned empty text",
        }

    except Exception as e:
        return {
            "text": "",
            "stt_engine": "whisper_failed",
            "language_code": "ta",
            "whisper_quality": 0.0,
            "error": f"Whisper transcription failed: {e}",
        }

    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass


def retry_google_with_audio_variants(audio_segment):
    """
    Retries Google STT using normal, louder, and very-loud audio variants.
    """

    attempts = []
    attempts.append(("normal", audio_segment))

    try:
        attempts.append(("louder_plus_6db", audio_segment.apply_gain(6)))
    except Exception:
        pass

    try:
        attempts.append(("louder_plus_10db", audio_segment.apply_gain(10)))
    except Exception:
        pass

    last_error = None

    for attempt_name, attempt_audio in attempts:
        result = transcribe_google_tamil(attempt_audio)

        if result.get("text", "").strip():
            result["retry_attempt"] = attempt_name
            return result

        last_error = result.get("error")

    return {
        "text": "",
        "stt_engine": "google_speech_recognition",
        "language_code": "ta-IN",
        "whisper_quality": 0.0,
        "error": f"All Google STT retry attempts failed. Last error: {last_error}",
        "retry_attempt": "failed_all",
    }


def transcribe_audiosegment_free(audio_segment, prefer_torch=False):
    """
    Main STT function used by Streamlit.

    prefer_torch=False:
        Google Tamil STT with retries.

    prefer_torch=True:
        Whisper first, then Google fallback with retries.
    """

    if prefer_torch:
        whisper_result = transcribe_whisper_lazy(audio_segment)

        if whisper_result.get("text", "").strip():
            return whisper_result

        google_result = retry_google_with_audio_variants(audio_segment)

        if not google_result.get("text", "").strip():
            google_result["error"] = (
                f"Whisper failed: {whisper_result.get('error')} | "
                f"Google failed: {google_result.get('error')}"
            )

        return google_result

    return retry_google_with_audio_variants(audio_segment)

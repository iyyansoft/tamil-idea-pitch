# speech_pipeline/audio_stream.py

import numpy as np
import sounddevice as sd
import soundfile as sf

from .config import SAMPLE_RATE, CHANNELS, CHUNK_SECONDS, SILENCE_THRESHOLD, TEMP_AUDIO_FILE


def record_chunk(seconds: int = CHUNK_SECONDS,
                 sample_rate: int = SAMPLE_RATE,
                 channels: int = CHANNELS):
    print(f"[MIC] Recording {seconds} second chunk...")
    audio = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype="float32"
    )
    sd.wait()

    if channels > 1:
        audio = np.mean(audio, axis=1)
    else:
        audio = audio.flatten()

    return audio.astype(np.float32)


def is_silent(audio: np.ndarray, threshold: float = SILENCE_THRESHOLD) -> bool:
    if audio.size == 0:
        return True
    rms = np.sqrt(np.mean(np.square(audio)))
    return rms < threshold


def save_temp_audio(audio: np.ndarray, file_path: str = TEMP_AUDIO_FILE, sample_rate: int = SAMPLE_RATE):
    sf.write(file_path, audio, sample_rate)
    return file_path
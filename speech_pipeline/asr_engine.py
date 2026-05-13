# speech_pipeline/asr_engine.py

import whisper
import numpy as np
from .config import WHISPER_MODEL, LANGUAGE, TASK, USE_FP16


class TamilASREngine:
    def __init__(self, model_name: str = WHISPER_MODEL):
        print(f"[INFO] Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name)

    def transcribe_array(self, audio_array: np.ndarray) -> str:
        if audio_array.ndim > 1:
            audio_array = audio_array.squeeze()

        audio_array = audio_array.astype(np.float32)

        result = self.model.transcribe(
            audio_array,
            language=LANGUAGE,
            task=TASK,
            fp16=USE_FP16,
            verbose=False
        )
        return result["text"].strip()

    def transcribe_file(self, file_path: str) -> str:
        result = self.model.transcribe(
            file_path,
            language=LANGUAGE,
            task=TASK,
            fp16=USE_FP16,
            verbose=False
        )
        return result["text"].strip()
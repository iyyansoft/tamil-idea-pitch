# speech_pipeline/config.py

WHISPER_MODEL = "small"
LANGUAGE = "ta"
TASK = "transcribe"

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SECONDS = 5
SILENCE_THRESHOLD = 0.01

USE_FP16 = False

# Temporary file used for each recorded chunk
TEMP_AUDIO_FILE = "temp_realtime.wav"
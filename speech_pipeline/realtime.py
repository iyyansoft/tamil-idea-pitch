# speech_pipeline/realtime.py

from .audio_stream import record_chunk, is_silent
from .asr_engine import TamilASREngine


class RealtimeTamilTranscriber:
    def __init__(self):
        self.asr = TamilASREngine()
        self.full_text_parts = []

    def start(self):
        print("\n=== Real-time Tamil Speech Recognition Started ===")
        print("Speak your idea in Tamil.")
        print("Press Ctrl + C to stop and finalize.\n")

        try:
            while True:
                audio = record_chunk()

                if is_silent(audio):
                    print("[SKIP] Silent / low-energy chunk")
                    continue

                text = self.asr.transcribe_array(audio)

                if text:
                    self.full_text_parts.append(text)
                    print(f"[TEXT] {text}")
                else:
                    print("[TEXT] No speech recognized")

        except KeyboardInterrupt:
            print("\n[INFO] Stopping transcription...")

        final_paragraph = " ".join(self.full_text_parts).strip()
        return final_paragraph
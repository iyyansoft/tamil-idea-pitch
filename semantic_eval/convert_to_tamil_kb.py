import json
import os
import time
from deep_translator import GoogleTranslator


class TamilTranslator:
    def __init__(self):
        self.translator = GoogleTranslator(source="en", target="ta")

    def translate(self, text):
        return self.translator.translate(text)


def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def convert_kb(input_file, output_file, sleep_sec=1.0):
    english_data = load_json(input_file, [])
    tamil_data = load_json(output_file, [])

    # Build set of already converted English texts
    done_english = set()
    for item in tamil_data:
        if isinstance(item, dict) and "english" in item:
            done_english.add(item["english"])

    translator = TamilTranslator()

    total = len(english_data)
    done = len(done_english)

    print(f"[INFO] English KB items     : {total}")
    print(f"[INFO] Already translated  : {done}")
    print(f"[INFO] Remaining           : {total - done}")

    for idx, text in enumerate(english_data, start=1):
        if text in done_english:
            continue

        try:
            tamil_text = translator.translate(text)

            tamil_data.append({
                "english": text,
                "tamil": tamil_text
            })

            done_english.add(text)

            # save after every item so progress is never lost
            save_json(output_file, tamil_data)

            print(f"[OK] {len(done_english)}/{total}")

            time.sleep(sleep_sec)

        except Exception as e:
            print(f"[ERROR] Failed at item {idx}: {e}")

            # Save progress before continuing
            save_json(output_file, tamil_data)

            # Wait a bit and continue
            time.sleep(3)

    print("[INFO] Translation completed.")
    print(f"[INFO] Final Tamil KB count: {len(tamil_data)}")


if __name__ == "__main__":
    convert_kb(
        input_file="data/knowledge_base.json",
        output_file="data/knowledge_base_tamil.json",
        sleep_sec=1.0
    )
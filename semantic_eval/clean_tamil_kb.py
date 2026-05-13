import json
import re


def clean_text(text):
    # remove URLs
    text = re.sub(r'http\S+', '', text)

    # remove special symbols
    text = re.sub(r'[\$\{\}\\]', '', text)

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def clean_kb(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = []
    seen = set()

    for item in data:
        eng = item.get("english", "").strip()
        tam = item.get("tamil", "").strip()

        if not eng or not tam:
            continue

        if eng in seen:
            continue

        tam = clean_text(tam)

        # filter very small texts
        if len(tam) < 30:
            continue

        cleaned.append({
            "english": eng,
            "tamil": tam
        })

        seen.add(eng)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Cleaned dataset size: {len(cleaned)}")


if __name__ == "__main__":
    clean_kb(
        "data/knowledge_base_tamil.json",
        "data/knowledge_base_tamil_clean.json"
    )
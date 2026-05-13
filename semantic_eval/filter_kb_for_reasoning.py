import json
import re

INPUT_FILE = "data/knowledge_base_tamil_clean.json"
OUTPUT_FILE = "data/knowledge_base_tamil_reasoning.json"

BAD_PATTERNS = [
    r"http\S+",
    r"www\.",
    r"\bubuntu\b",
    r"\bios\b",
    r"\bbash\b",
    r"\bsip\b",
    r"\bport 5060\b",
    r"\bfabric\b",
    r"\bwhoever that is\b",
    r"\bmy machine\b",
]

GOOD_HINTS = [
    "drone", "space", "vacuum", "battery", "sensor", "network",
    "signal", "filter", "beam", "load", "stress", "steel",
    "hydrogen", "generator", "cooling", "voltage", "power",
    "bridge", "traffic", "algorithm", "data", "model", "training",
    "gps", "accelerometer", "gyroscope", "internet", "cloud",
    "feasible", "practical", "safety", "efficient", "engineering",
]

def looks_bad(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in BAD_PATTERNS)

def looks_useful(text: str) -> bool:
    t = text.lower()
    return any(h in t for h in GOOD_HINTS)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

filtered = []
for item in data:
    eng = item.get("english", "").strip()
    tam = item.get("tamil", "").strip()

    if not eng or not tam:
        continue
    if len(eng) < 40:
        continue
    if looks_bad(eng):
        continue

    filtered.append({
        "english": eng,
        "tamil": tam
    })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(filtered, f, indent=2, ensure_ascii=False)

print(f"[INFO] Filtered KB size: {len(filtered)}")
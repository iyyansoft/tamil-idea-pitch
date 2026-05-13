import xml.etree.ElementTree as ET
import html
import re
import os


def clean_text(text):
    if not text:
        return ""
    
    text = html.unescape(text)
    text = re.sub(r'<.*?>', '', text)  # remove HTML
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_knowledge(xml_path, min_score=5, max_items=500):
    knowledge = []

    context = ET.iterparse(xml_path, events=("end",))
    
    for event, elem in context:
        if elem.tag == "row":
            post_type = elem.attrib.get("PostTypeId")
            score = int(elem.attrib.get("Score", 0))
            body = elem.attrib.get("Body")

            # only answers (PostTypeId = 2)
            if post_type == "2" and score >= min_score:
                text = clean_text(body)

                if len(text) > 50:  # filter small junk
                    knowledge.append(text)

            elem.clear()

            if len(knowledge) >= max_items:
                break

    return knowledge


def save_knowledge(data, out_file):
    import json
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    sources = [
        "data/Engineering/Posts.xml",
        "data/SoftwareEngineering/Posts.xml",
        "data/networkengineering/Posts.xml",
        "data/cs/Posts.xml"
    ]

    all_knowledge = []

    for path in sources:
        if os.path.exists(path):
            print(f"[INFO] Processing {path}")
            k = extract_knowledge(path)
            all_knowledge.extend(k)

    print(f"[INFO] Total knowledge items: {len(all_knowledge)}")

    save_knowledge(all_knowledge, "data/knowledge_base.json")
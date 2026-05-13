import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


TAMIL_STOPWORDS = {
    "ஒரு", "இந்த", "அந்த", "மற்றும்", "உம்", "என்று", "ஆக", "இது", "அது", "இல்", "க்கு",
    "உள்ள", "மேலும்", "மூலம்", "பற்றி", "போன்ற", "செய்ய", "செய்யும்", "பயன்படுத்தி",
    "system", "project", "idea"
}


class TamilEngineeringKB:
    def __init__(
        self,
        kb_path: str = "data/knowledge_base_tamil_reasoning.json",
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    ):
        base_dir = Path(__file__).resolve().parent.parent
        self.kb_path = Path(kb_path)

        if not self.kb_path.is_absolute():
            self.kb_path = base_dir / self.kb_path

        if not self.kb_path.exists():
            raise FileNotFoundError(f"Knowledge base file not found: {self.kb_path}")

        with open(self.kb_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.records = []
        for idx, item in enumerate(raw_data):
            tamil = str(item.get("tamil", "")).strip()
            english = str(item.get("english", "")).strip()
            reasoning = str(item.get("reasoning", "")).strip()

            combined = " ".join([x for x in [tamil, english, reasoning] if x]).strip()
            if not combined:
                continue

            self.records.append({
                "id": idx,
                "tamil": tamil,
                "english": english,
                "reasoning": reasoning,
                "combined": combined,
                "tokens": self._tokenize(combined),
            })

        if not self.records:
            raise ValueError("Knowledge base is empty or invalid.")

        self.texts = [r["combined"] for r in self.records]

        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1
        )
        self.tfidf = self.vectorizer.fit_transform(self.texts)

        self.model = None
        self.embeddings = None

        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(model_name)
                self.embeddings = self.model.encode(
                    self.texts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
            except Exception:
                self.model = None
                self.embeddings = None

    def _tokenize(self, text: str) -> List[str]:
        text = re.sub(r"[^\w\s\u0B80-\u0BFF]", " ", text.lower())
        parts = [p.strip() for p in text.split() if p.strip()]
        return [p for p in parts if p not in TAMIL_STOPWORDS and len(p) > 1]

    def _embed_query(self, query: str):
        if self.model is None or self.embeddings is None:
            return None
        return self.model.encode([query], convert_to_numpy=True)

    def _dense_scores(self, query: str) -> np.ndarray:
        q = self._embed_query(query)
        if q is None:
            return np.zeros(len(self.records), dtype=float)
        return cosine_similarity(q, self.embeddings)[0]

    def _sparse_scores(self, query: str) -> np.ndarray:
        q = self.vectorizer.transform([query])
        return cosine_similarity(q, self.tfidf)[0]

    def _coverage_score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0
        qset = set(query_tokens)
        dset = set(doc_tokens)
        overlap = len(qset & dset)
        return overlap / max(1, len(qset))

    def similarity(self, text_a: str, text_b: str) -> float:
        a_tokens = self._tokenize(text_a)
        b_tokens = self._tokenize(text_b)

        lexical = self._coverage_score(a_tokens, b_tokens)
        sparse = float(
            cosine_similarity(
                self.vectorizer.transform([text_a]),
                self.vectorizer.transform([text_b])
            )[0][0]
        )

        if self.model is not None:
            dense = float(
                cosine_similarity(
                    self.model.encode([text_a], convert_to_numpy=True),
                    self.model.encode([text_b], convert_to_numpy=True)
                )[0][0]
            )
            return float(0.55 * dense + 0.30 * sparse + 0.15 * lexical)

        return float(0.75 * sparse + 0.25 * lexical)

    def search(self, query: str, top_k: int = 5, lens: str = "technical") -> List[Dict]:
        query = str(query).strip()
        if not query:
            return []

        query_tokens = self._tokenize(query)
        dense = self._dense_scores(query)
        sparse = self._sparse_scores(query)

        lens_terms = {
            "technical": [
                "sensor", "circuit", "algorithm", "module", "system", "design",
                "சென்சார்", "அமைப்பு", "வடிவமைப்பு"
            ],
            "practical": [
                "cost", "prototype", "deploy", "maintenance", "battery", "market", "feasible",
                "செலவு", "பயன்பாடு", "நடைமுறை"
            ],
            "theoretical": [
                "principle", "theory", "analysis", "model", "equation", "logic",
                "கோட்பாடு", "பகுப்பாய்வு", "கணிதம்"
            ],
        }.get(lens, [])

        lens_text = query + " " + " ".join(lens_terms)
        lens_sparse = self._sparse_scores(lens_text)
        lens_dense = self._dense_scores(lens_text)

        results = []
        for i, rec in enumerate(self.records):
            coverage = self._coverage_score(query_tokens, rec["tokens"])

            score = (
                0.30 * sparse[i]
                + 0.30 * dense[i]
                + 0.20 * lens_sparse[i]
                + 0.10 * lens_dense[i]
                + 0.10 * coverage
            )

            results.append({
                "id": rec["id"],
                "tamil": rec["tamil"],
                "english": rec["english"],
                "reasoning": rec["reasoning"],
                "similarity": float(score),
                "dense_similarity": float(dense[i]),
                "sparse_similarity": float(sparse[i]),
                "coverage": float(coverage),
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
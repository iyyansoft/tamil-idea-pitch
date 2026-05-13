import re
from statistics import mean
from typing import Dict, List

from .knowledge_engine import TamilEngineeringKB


class TamilIdeaPitchEvaluator:
    def __init__(self, kb_path="data/knowledge_base_tamil_reasoning.json"):
        self.kb = TamilEngineeringKB(kb_path=kb_path)

    def _clip(self, x: float) -> float:
        return max(0.0, min(1.0, x))

    def _to10(self, x: float) -> float:
        return round(self._clip(x) * 10.0, 2)

    def _clean(self, text: str) -> str:
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        return text

    def _length_quality(self, text: str) -> float:
        wc = len(self._clean(text).split())

        if wc <= 4:
            return 0.15
        if wc <= 10:
            return 0.35
        if wc <= 20:
            return 0.65
        return 1.0

    def _evidence_summary(self, matches: List[Dict]) -> Dict[str, float]:
        if not matches:
            return {"top": 0.0, "avg": 0.0, "coherence": 0.0, "coverage": 0.0}

        sims = [m["similarity"] for m in matches]
        coverages = [m.get("coverage", 0.0) for m in matches]

        coherence_pairs = []
        for i in range(len(matches)):
            for j in range(i + 1, len(matches)):
                a = matches[i]["tamil"] or matches[i]["english"] or matches[i].get("reasoning", "")
                b = matches[j]["tamil"] or matches[j]["english"] or matches[j].get("reasoning", "")
                coherence_pairs.append(self.kb.similarity(a, b))

        return {
            "top": max(sims),
            "avg": mean(sims),
            "coherence": mean(coherence_pairs) if coherence_pairs else mean(sims),
            "coverage": mean(coverages),
        }

    def _make_explanation(self, title: str, matches: List[Dict]) -> str:
        if not matches:
            return f"{title} க்கு போதுமான அறிவுத்தள ஆதாரம் கிடைக்கவில்லை."

        top = matches[0]
        snippet = self._clean(
            top.get("tamil") or top.get("english") or top.get("reasoning") or ""
        )

        if len(snippet) > 220:
            snippet = snippet[:220] + "..."

        return (
            f"{title} மதிப்பீடு semantic retrieval மூலம் கிடைத்த "
            f"engineering evidence அடிப்படையில் கணிக்கப்பட்டது: {snippet}"
        )

    def evaluate(self, transcript_text: str, asr_quality: float = 1.0):
        transcript_text = self._clean(transcript_text)

        if not transcript_text:
            return {
                "transcript": transcript_text,
                "scores": {
                    "technical_correctness": 0.0,
                    "real_life_probability": 0.0,
                    "theoretical_correctness": 0.0,
                },
                "final_score": 0.0,
                "explanations": {
                    "technical_correctness": "Transcript empty.",
                    "real_life_probability": "Transcript empty.",
                    "theoretical_correctness": "Transcript empty.",
                },
                "feedback": ["Transcription quality is too low for idea evaluation."],
                "matches": {},
            }

        technical_matches = self.kb.search(transcript_text, top_k=5, lens="technical")
        practical_matches = self.kb.search(transcript_text, top_k=5, lens="practical")
        theory_matches = self.kb.search(transcript_text, top_k=5, lens="theoretical")

        tech = self._evidence_summary(technical_matches)
        prac = self._evidence_summary(practical_matches)
        theory = self._evidence_summary(theory_matches)

        length_quality = self._length_quality(transcript_text)
        asr_quality = self._clip(asr_quality)

        technical_score = self._to10(
            0.42 * tech["avg"] +
            0.22 * tech["top"] +
            0.18 * tech["coverage"] +
            0.18 * asr_quality
        )

        practical_score = self._to10(
            0.40 * prac["avg"] +
            0.22 * prac["coverage"] +
            0.20 * prac["top"] +
            0.10 * length_quality +
            0.08 * asr_quality
        )

        theory_score = self._to10(
            0.44 * theory["avg"] +
            0.24 * theory["coherence"] +
            0.18 * theory["coverage"] +
            0.14 * asr_quality
        )

        scores = {
            "technical_correctness": technical_score,
            "real_life_probability": practical_score,
            "theoretical_correctness": theory_score,
        }

        final_score = round(mean(scores.values()), 2)

        feedback = []
        if asr_quality < 0.45:
            feedback.append("Audio/transcription quality is limiting semantic evaluation. Consider better ASR settings or preprocessing.")
        if tech["coverage"] < 0.20:
            feedback.append("Pitch needs more concrete engineering terms, components, or mechanisms.")
        if prac["coverage"] < 0.20:
            feedback.append("Mention deployment, cost, materials, maintenance, users, or constraints for real-life feasibility.")
        if theory["coverage"] < 0.20:
            feedback.append("Add principle, process, formula, control logic, or scientific reasoning behind the idea.")
        if not feedback:
            feedback.append("Semantic evidence is reasonably aligned with the knowledge base.")

        return {
            "transcript": transcript_text,
            "scores": scores,
            "final_score": final_score,
            "explanations": {
                "technical_correctness": self._make_explanation("technical_correctness", technical_matches),
                "real_life_probability": self._make_explanation("real_life_probability", practical_matches),
                "theoretical_correctness": self._make_explanation("theoretical_correctness", theory_matches),
            },
            "feedback": feedback,
            "matches": {
                "technical_correctness": technical_matches,
                "real_life_probability": practical_matches,
                "theoretical_correctness": theory_matches,
            },
            "debug": {
                "asr_quality": round(asr_quality, 4),
                "length_quality": round(length_quality, 4),
                "technical_evidence": tech,
                "practical_evidence": prac,
                "theory_evidence": theory,
            },
        }
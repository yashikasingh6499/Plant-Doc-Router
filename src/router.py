from typing import Dict, Tuple

from src.config import SOURCE_CONFIG
from src.embeddings import get_embedding


class SemanticRouter:
    def __init__(self):
        self.source_profiles = {}
        self.keyword_map = {
            "safety": [
                "ppe", "spill", "alarm", "evacuate", "incident", "hazard",
                "lockout", "tagout", "chemical", "unsafe", "fire"
            ],
            "maintenance": [
                "repair", "vibration", "bearing", "motor", "pump", "sensor",
                "calibration", "conveyor", "troubleshoot", "trip", "fault"
            ],
            "quality": [
                "defect", "inspection", "tolerance", "acceptance", "batch",
                "rejection", "sample", "dimension", "quality", "label"
            ],
        }

        for source_key, cfg in SOURCE_CONFIG.items():
            self.source_profiles[source_key] = {
                "label": cfg["label"],
                "embedding": get_embedding(cfg["description"])
            }

    @staticmethod
    def _cosine(a, b) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def route(self, question: str) -> Tuple[str, Dict[str, float]]:
        q = question.lower()
        q_emb = get_embedding(question)

        scores = {}
        for source_key, profile in self.source_profiles.items():
            semantic = self._cosine(q_emb, profile["embedding"])
            keyword_hits = sum(1 for kw in self.keyword_map[source_key] if kw in q)
            keyword_bonus = min(keyword_hits * 0.08, 0.25)
            scores[source_key] = semantic + keyword_bonus

        best_source = max(scores, key=scores.get)
        return best_source, scores
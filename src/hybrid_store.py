import json
import os
import pickle
from typing import Dict, List

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from src.utils import ensure_dir, tokenize_for_bm25


class HybridStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: List[Dict] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25 = None

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return vectors / norms

    def add(self, embeddings: List[List[float]], metadata: List[Dict]) -> None:
        arr = np.array(embeddings, dtype="float32")
        arr = self._normalize(arr)
        self.index.add(arr)
        self.metadata.extend(metadata)

        new_tokens = [tokenize_for_bm25(m["chunk_text"]) for m in metadata]
        self.tokenized_corpus.extend(new_tokens)
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _minmax(self, scores: Dict[int, float]) -> Dict[int, float]:
        if not scores:
            return {}
        values = list(scores.values())
        min_v = min(values)
        max_v = max(values)
        if max_v - min_v < 1e-12:
            return {k: 1.0 for k in scores}
        return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}

    def search(
        self,
        query_text: str,
        query_embedding: List[float],
        top_k: int = 5,
        dense_candidates: int = 12,
        sparse_candidates: int = 12,
        alpha: float = 0.65,
    ) -> List[Dict]:
        query = np.array([query_embedding], dtype="float32")
        query = self._normalize(query)

        dense_scores_raw, dense_indices = self.index.search(query, dense_candidates)
        dense_scores = {}
        for idx, score in zip(dense_indices[0], dense_scores_raw[0]):
            if idx != -1:
                dense_scores[int(idx)] = float(score)

        query_tokens = tokenize_for_bm25(query_text)
        sparse_all = self.bm25.get_scores(query_tokens) if self.bm25 else []
        sparse_ranked = np.argsort(sparse_all)[::-1][:sparse_candidates]
        sparse_scores = {int(idx): float(sparse_all[idx]) for idx in sparse_ranked}

        dense_norm = self._minmax(dense_scores)
        sparse_norm = self._minmax(sparse_scores)

        candidate_ids = set(dense_norm.keys()) | set(sparse_norm.keys())
        merged = []

        for idx in candidate_ids:
            d = dense_norm.get(idx, 0.0)
            s = sparse_norm.get(idx, 0.0)
            hybrid = alpha * d + (1 - alpha) * s
            item = self.metadata[idx].copy()
            item["dense_score"] = d
            item["sparse_score"] = s
            item["hybrid_score"] = hybrid
            merged.append(item)

        merged.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return merged[:top_k]

    def save(self, folder_path: str) -> None:
        ensure_dir(folder_path)
        faiss.write_index(self.index, os.path.join(folder_path, "index.faiss"))

        with open(os.path.join(folder_path, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)

        with open(os.path.join(folder_path, "tokenized_corpus.pkl"), "wb") as f:
            pickle.dump(self.tokenized_corpus, f)

        with open(os.path.join(folder_path, "info.json"), "w", encoding="utf-8") as f:
            json.dump({"dim": self.dim, "size": len(self.metadata)}, f, indent=2)

    @classmethod
    def load(cls, folder_path: str):
        with open(os.path.join(folder_path, "info.json"), "r", encoding="utf-8") as f:
            info = json.load(f)

        store = cls(dim=info["dim"])
        store.index = faiss.read_index(os.path.join(folder_path, "index.faiss"))

        with open(os.path.join(folder_path, "metadata.pkl"), "rb") as f:
            store.metadata = pickle.load(f)

        with open(os.path.join(folder_path, "tokenized_corpus.pkl"), "rb") as f:
            store.tokenized_corpus = pickle.load(f)

        store.bm25 = BM25Okapi(store.tokenized_corpus)
        return store
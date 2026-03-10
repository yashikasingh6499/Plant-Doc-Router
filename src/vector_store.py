import os
import json
import pickle
from typing import List, Dict, Tuple

import faiss
import numpy as np

from src.utils import ensure_dir


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: List[Dict] = []

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

    def search(self, query_embedding: List[float], top_k: int = 4) -> List[Tuple[Dict, float]]:
        query = np.array([query_embedding], dtype="float32")
        query = self._normalize(query)

        scores, indices = self.index.search(query, top_k)
        results = []

        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((self.metadata[idx], float(score)))
        return results

    def save(self, folder_path: str) -> None:
        ensure_dir(folder_path)
        faiss.write_index(self.index, os.path.join(folder_path, "index.faiss"))
        with open(os.path.join(folder_path, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)
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

        return store
from typing import List
from sentence_transformers import SentenceTransformer

from src.config import EMBED_MODEL_NAME

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def get_embedding(text: str) -> List[float]:
    model = get_model()
    emb = model.encode(text, normalize_embeddings=True)
    return emb.tolist()


def get_embeddings(texts: List[str]) -> List[List[float]]:
    model = get_model()
    embs = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return embs.tolist()
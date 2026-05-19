from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

_EMBED_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_encoder() -> SentenceTransformer:
    return SentenceTransformer(_EMBED_MODEL)


def load_encoder() -> None:
    """Pre-load the sentence-transformer model (avoids a cold-start delay later)."""
    _get_encoder()


def encode(texts: list[str]) -> np.ndarray:
    return _get_encoder().encode(texts, show_progress_bar=False, batch_size=32)

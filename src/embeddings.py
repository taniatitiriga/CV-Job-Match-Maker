"""Chunking, embedding and pooling utilities.

These helpers mirror the logic used in
``notebooks/embed_processed_json_to_qdrant.ipynb`` so that the runtime query
vectors stay in the same space as the indexed dataset.
"""

from __future__ import annotations

import json
import urllib.request
from urllib.error import HTTPError, URLError

import numpy as np

from .config import (
    EMBED_DIM,
    EMBED_MODEL,
    EMBEDDING_API_URLS,
    HYBRID_W,
    MAX_CHUNK_TOKENS,
    SOFTMAX_ALPHA,
    TOKEN_OVERLAP,
    TOP_K,
)


_HAS_TIKTOKEN = False
_tik_enc = None

try:
    import tiktoken

    _tik_enc = tiktoken.get_encoding("cl100k_base")
    _HAS_TIKTOKEN = True
except ImportError:
    _tik_enc = None


def chunk_text_by_tokens(
    text: str,
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap_tokens: int = TOKEN_OVERLAP,
) -> list[str]:
    """Split text into overlapping windows of at most ``max_tokens`` tokens."""

    text = text.strip()
    if not text:
        return []

    overlap_tokens = max(0, min(overlap_tokens, max_tokens - 1))
    step = max(1, max_tokens - overlap_tokens)

    if _HAS_TIKTOKEN and _tik_enc is not None:
        ids = _tik_enc.encode(text)
        chunks: list[str] = []

        for i in range(0, len(ids), step):
            span = ids[i : i + max_tokens]
            if not span:
                continue

            while span:
                decoded = _tik_enc.decode(span).strip()
                if not decoded:
                    break
                if len(_tik_enc.encode(decoded)) <= max_tokens:
                    chunks.append(decoded)
                    break
                span = span[:-1]

        first = _tik_enc.decode(ids[:max_tokens]).strip()
        return chunks if chunks else ([first] if first else [])

    # Char-based fallback (~4 chars / token).
    approx_cpt = 4
    max_chars = max(1, max_tokens * approx_cpt)
    overlap_chars = overlap_tokens * approx_cpt
    stride = max(1, max_chars - overlap_chars)
    chunks = []
    n = len(text)
    start = 0

    while start < n:
        end = min(n, start + max_chars)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start += stride

    return chunks if chunks else [text]


def embed_one_chunk(chunk: str) -> list[float]:
    """Call the local embeddings HTTP endpoint, with endpoint fallbacks."""

    body = json.dumps({"model": EMBED_MODEL, "input": chunk}).encode("utf-8")
    last_error: Exception | None = None

    for api_url in EMBEDDING_API_URLS:
        req = urllib.request.Request(
            api_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            raise RuntimeError(
                f"embedding API HTTP {e.code} at {api_url}: {e.reason}"
            ) from e
        except URLError as e:
            last_error = e
            continue

        data = payload.get("data")
        if not data or not isinstance(data, list):
            raise RuntimeError(
                f"embedding API returned malformed payload at {api_url}: {payload}"
            )

        vec = data[0].get("embedding")
        if not isinstance(vec, list):
            raise RuntimeError(
                f"embedding API response missing embedding vector at {api_url}: {payload}"
            )

        return vec

    raise RuntimeError(
        "embedding API connection failed for all endpoints "
        f"{EMBEDDING_API_URLS}. Start embedding server with ./run-llama-servers.sh "
        f"and ensure port 8081 is reachable. Last error: {last_error}"
    )


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    for chunk in chunks:
        vec = embed_one_chunk(chunk)
        if len(vec) != EMBED_DIM:
            raise RuntimeError(
                f"unexpected embedding dimension {len(vec)} (expected {EMBED_DIM})"
            )
        out.append(vec)
    return out


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0:
        return v
    return v / n


def _chunk_salience(emb: np.ndarray) -> float:
    return float(np.linalg.norm(emb))


def _pool_mean_all(vectors: np.ndarray) -> np.ndarray:
    return _l2_normalize(vectors.mean(axis=0))


def _pool_max_score(vectors: np.ndarray, salience: np.ndarray) -> np.ndarray:
    idx = int(np.argmax(salience))
    return _l2_normalize(vectors[idx])


def _pool_topk_mean(
    vectors: np.ndarray, salience: np.ndarray, k: int = TOP_K
) -> np.ndarray:
    k_eff = max(1, min(k, len(vectors)))
    order = np.argsort(-salience)[:k_eff]
    return _l2_normalize(vectors[order].mean(axis=0))


def _pool_weighted_topk_mean(
    vectors: np.ndarray, salience: np.ndarray, k: int = TOP_K
) -> np.ndarray:
    k_eff = max(1, min(k, len(vectors)))
    order = np.argsort(-salience)[:k_eff]
    weights = np.arange(k_eff, 0, -1, dtype=np.float64)
    weighted = np.average(vectors[order], axis=0, weights=weights)
    return _l2_normalize(weighted)


def _pool_softmax(
    vectors: np.ndarray, salience: np.ndarray, alpha: float = SOFTMAX_ALPHA
) -> np.ndarray:
    s = salience.astype(np.float64)
    stabilized = alpha * (s - s.max())
    w = np.exp(stabilized)
    w /= w.sum()
    return _l2_normalize((w[:, None] * vectors).sum(axis=0))


def _pool_hybrid(
    vectors: np.ndarray,
    salience: np.ndarray,
    k: int = TOP_K,
    w: float = HYBRID_W,
) -> np.ndarray:
    best_idx = int(np.argmax(salience))
    v_max = vectors[best_idx]
    v_topk = _pool_topk_mean(vectors, salience, k=k)
    blended = w * v_max + (1.0 - w) * v_topk
    return _l2_normalize(blended)


def build_pooled_vectors(
    chunk_embeddings: list[list[float]],
) -> dict[str, np.ndarray]:
    mats = np.asarray(chunk_embeddings, dtype=np.float64)
    sal = np.array(
        [_chunk_salience(mats[i]) for i in range(len(mats))], dtype=np.float64
    )

    return {
        "max_score": _pool_max_score(mats, sal),
        "mean_all": _pool_mean_all(mats),
        "topk_mean_k5": _pool_topk_mean(mats, sal, k=TOP_K),
        "weighted_topk_mean_k5": _pool_weighted_topk_mean(mats, sal, k=TOP_K),
        "softmax_pooling_alpha10": _pool_softmax(mats, sal, alpha=SOFTMAX_ALPHA),
        "hybrid_max_topk_k5_w0_4": _pool_hybrid(mats, sal, k=TOP_K, w=HYBRID_W),
    }


def embed_document_text(text: str) -> dict[str, list[float]]:
    """Full pipeline: chunk → embed → pool. Returns one vector per pooling key."""

    chunks = chunk_text_by_tokens(text)
    if not chunks:
        raise ValueError("uploaded document yielded no embeddable text")

    embeddings = embed_chunks(chunks)
    pooled = build_pooled_vectors(embeddings)
    return {k: v.astype(float).tolist() for k, v in pooled.items()}

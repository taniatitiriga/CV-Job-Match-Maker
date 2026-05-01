"""Centralised configuration for the matching app.

Mirrors values used by the embedding/preprocessing notebooks so that query-time
vectors live in the exact same space as the indexed dataset.
"""

from __future__ import annotations

import os
from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CV_DATA_DIR = PROJECT_ROOT / "data" / "cv-dataset-processed"
JOB_DATA_DIR = PROJECT_ROOT / "data" / "job-postings-processed"

# Qdrant
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
SHARED_COLLECTION = os.environ.get(
    "QDRANT_COLLECTION", "resume-job-posting-poolings"
)

# Embedding HTTP endpoints (try in order). Mirrors the notebook setup so the
# same llama-server is reused at query time.
EMBEDDING_API_URLS: tuple[str, ...] = (
    os.environ.get("EMBEDDING_API_URL", "http://127.0.0.1:8081/v1/embeddings"),
    "http://localhost:8081/v1/embeddings",
    "http://172.18.80.1:8081/v1/embeddings",
)
EMBED_MODEL = os.environ.get("EMBED_MODEL", "mxbai-embed-large-v1-f16")
EMBED_DIM = 1024

# Chunking parameters must stay aligned with the indexing notebook.
MAX_CHUNK_TOKENS = 384
TOKEN_OVERLAP = 64

# Pooling configuration (must match build_pooled_vectors in the notebook).
TOP_K = 5
SOFTMAX_ALPHA = 10.0
HYBRID_W = 0.4

# Pooling strategies the user can pick from. The first entry is the default.
POOLING_METHODS: tuple[str, ...] = (
    "hybrid_max_topk_k5_w0_4",
    "max_score",
    "mean_all",
    "topk_mean_k5",
    "weighted_topk_mean_k5",
    "softmax_pooling_alpha10",
)
DEFAULT_POOLING_METHOD = POOLING_METHODS[0]

POOLING_METHOD_LABELS: dict[str, str] = {
    "hybrid_max_topk_k5_w0_4": "Hybrid (max + top-k mean) — recommended",
    "max_score": "Max-score (single best chunk)",
    "mean_all": "Mean of all chunks",
    "topk_mean_k5": "Top-k mean (k=5)",
    "weighted_topk_mean_k5": "Weighted top-k mean (k=5)",
    "softmax_pooling_alpha10": "Softmax pooled (alpha=10)",
}

# UI defaults
DEFAULT_TOP_K_RESULTS = 5
MAX_TOP_K_RESULTS = 10

"""Qdrant search helpers used by the Dash app."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from .config import QDRANT_URL, SHARED_COLLECTION


@dataclass
class MatchHit:
    score: float
    payload: dict[str, Any]
    point_id: str

    @property
    def kind(self) -> str:
        return str(self.payload.get("kind", ""))

    @property
    def domain(self) -> str:
        return str(self.payload.get("domain", ""))

    @property
    def document_id(self) -> str:
        return str(self.payload.get("document_id", ""))

    @property
    def document_key(self) -> str:
        return str(
            self.payload.get("document_key")
            or f"{self.kind}-{self.domain}-{self.document_id}"
        )

    @property
    def source_path(self) -> str:
        return str(self.payload.get("source_path", ""))

    @property
    def json_filename(self) -> str:
        return str(self.payload.get("json_filename", ""))


_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    """Return a process-wide ``QdrantClient`` (lazily constructed)."""

    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
    return _client


def search_similar_documents(
    *,
    query_vector: list[float],
    kind: str,
    domain: str,
    pooling_method: str,
    top_k: int,
    collection: str = SHARED_COLLECTION,
) -> list[MatchHit]:
    """Find top ``top_k`` similar documents within a kind/domain/pooling slice."""

    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    flt = Filter(
        must=[
            FieldCondition(key="kind", match=MatchValue(value=kind)),
            FieldCondition(key="domain", match=MatchValue(value=domain)),
            FieldCondition(
                key="pooling_method", match=MatchValue(value=pooling_method)
            ),
        ]
    )

    client = get_client()
    response = client.query_points(
        collection_name=collection,
        query=query_vector,
        query_filter=flt,
        limit=top_k,
        with_payload=True,
    )

    hits: list[MatchHit] = []
    for point in response.points:
        hits.append(
            MatchHit(
                score=float(point.score),
                payload=dict(point.payload or {}),
                point_id=str(point.id),
            )
        )
    return hits

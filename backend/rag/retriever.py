"""
Retrieval functions used by the investigation tools (Phase 4).
Every result keeps its source metadata so the final report can cite evidence.
"""

from typing import Optional
from rag.store import get_store, MANUALS_COLLECTION, INCIDENTS_COLLECTION


def _similarity(distance: float) -> float:
    # Chroma's default metric is squared L2. For unit-normalized embeddings,
    # cosine_similarity = 1 - squared_L2 / 2.
    return round(1 - distance / 2, 4)


def search_technical_documents(query: str, k: int = 4, failure_type: Optional[str] = None) -> list[dict]:
    """Semantic search over the manuals. Optionally restrict to one failure type's manual."""
    store = get_store(MANUALS_COLLECTION)
    metadata_filter = {"failure_type": failure_type} if failure_type else None
    results = store.similarity_search_with_score(query, k=k, filter=metadata_filter)
    return [
        {
            "content": doc.page_content,
            "source": doc.metadata["source"],
            "title": doc.metadata["title"],
            "section": doc.metadata["section"],
            "failure_type": doc.metadata["failure_type"],
            "similarity": _similarity(score),
        }
        for doc, score in results
    ]


def search_similar_incidents(description: str, k: int = 5, machine_id: Optional[str] = None,
                              exclude_incident_id: Optional[int] = None) -> list[dict]:
    """
    Semantic search over historical incidents.
    exclude_incident_id lets the evaluation harness hide the incident under investigation,
    so the agent can't 'find' its own answer.
    """
    store = get_store(INCIDENTS_COLLECTION)
    metadata_filter = {"machine_id": machine_id} if machine_id else None
    fetch_k = k + 1 if exclude_incident_id is not None else k
    results = store.similarity_search_with_score(description, k=fetch_k, filter=metadata_filter)

    out = []
    for doc, score in results:
        if exclude_incident_id is not None and doc.metadata["incident_id"] == exclude_incident_id:
            continue
        out.append({
            "content": doc.page_content,
            "source": doc.metadata["source"],
            "incident_id": doc.metadata["incident_id"],
            "machine_id": doc.metadata["machine_id"],
            "date": doc.metadata["date"],
            "failure_type": doc.metadata["failure_type"],
            "similarity": _similarity(score),
        })
    return out[:k]

"""
Retrieval functions used by the investigation tools (Phase 4).
Every result keeps its source metadata so the final report can cite evidence.
"""

from typing import Optional
from rag.store import get_store, MANUALS_COLLECTION, INCIDENTS_COLLECTION


def search_technical_documents(query: str, k: int = 4, failure_type: Optional[str] = None) -> list[dict]:
    """Semantic search over the manuals. Optionally restrict to one failure type's manual."""
    store = get_store(MANUALS_COLLECTION)
    where = {"failure_type": failure_type} if failure_type else None
    results = store.search(query, k=k, where=where)
    return [
        {
            "content": rec["text"],
            "source": rec["metadata"]["source"],
            "title": rec["metadata"]["title"],
            "section": rec["metadata"]["section"],
            "failure_type": rec["metadata"]["failure_type"],
            "similarity": round(sim, 4),
        }
        for rec, sim in results
    ]


def search_similar_incidents(description: str, k: int = 5, machine_id: Optional[str] = None,
                              exclude_incident_id: Optional[int] = None) -> list[dict]:
    """
    Semantic search over historical incidents.
    exclude_incident_id lets the evaluation harness hide the incident under investigation,
    so the agent can't 'find' its own answer.
    """
    store = get_store(INCIDENTS_COLLECTION)
    where = {"machine_id": machine_id} if machine_id else None
    fetch_k = k + 1 if exclude_incident_id is not None else k
    results = store.search(description, k=fetch_k, where=where)

    out = []
    for rec, sim in results:
        meta = rec["metadata"]
        if exclude_incident_id is not None and meta["incident_id"] == exclude_incident_id:
            continue
        out.append({
            "content": rec["text"],
            "source": meta["source"],
            "incident_id": meta["incident_id"],
            "machine_id": meta["machine_id"],
            "date": meta["date"],
            "failure_type": meta["failure_type"],
            "similarity": round(sim, 4),
        })
    return out[:k]
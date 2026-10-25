"""
Shared vector-store setup: ONNX MiniLM embeddings + a plain NumPy store.

CHANGE vs earlier versions: Chroma (HNSW index) crashed with access violations on
this laptop, even on random vectors. The data is tiny (49 + 339 records), so exact
cosine search over a NumPy matrix is instant and needs no native index.
Chroma is still installed, but ONLY for its ONNX embedder (same all-MiniLM-L6-v2 model).
"""

import os
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import json
from pathlib import Path
from typing import Optional

import numpy as np
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

RAG_DIR = Path(__file__).resolve().parent
STORE_DIR = RAG_DIR / "vector_store"

MANUALS_COLLECTION = "manuals"
INCIDENTS_COLLECTION = "incidents"

_embeddings = None
_stores: dict = {}


class OnnxMiniLMEmbeddings:
    """Wraps Chroma's ONNX MiniLM embedder. Output is L2-normalized."""

    def __init__(self):
        self._fn = ONNXMiniLM_L6_V2()  # ~80MB model, downloaded once then cached

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        out = []
        for i in range(0, len(texts), batch_size):
            out.extend([float(x) for x in v] for v in self._fn(texts[i:i + batch_size]))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embeddings() -> OnnxMiniLMEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OnnxMiniLMEmbeddings()
    return _embeddings


class NumpyStore:
    """One collection = <name>.npy (vectors) + <name>.json (id, text, metadata per row)."""

    def __init__(self, name: str):
        self.name = name
        self.vec_path = STORE_DIR / f"{name}.npy"
        self.meta_path = STORE_DIR / f"{name}.json"
        self.vectors: Optional[np.ndarray] = None
        self.records: list[dict] = []

    @classmethod
    def build(cls, name: str, records: list[dict]) -> "NumpyStore":
        """records: [{'id', 'text', 'metadata'}, ...]"""
        store = cls(name)
        STORE_DIR.mkdir(parents=True, exist_ok=True)
        vecs = np.asarray(get_embeddings().embed_documents([r["text"] for r in records]), dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        np.save(store.vec_path, vecs)
        store.meta_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        store.vectors, store.records = vecs, records
        return store

    def load(self) -> "NumpyStore":
        if not self.vec_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError(
                f"Vector store '{self.name}' not found in {STORE_DIR}. Run: python rag/03_build_vectorstore.py"
            )
        self.vectors = np.load(self.vec_path)
        self.records = json.loads(self.meta_path.read_text(encoding="utf-8"))
        return self

    def search(self, query: str, k: int = 4, where: Optional[dict] = None) -> list[tuple[dict, float]]:
        """Exact cosine search. 'where' = simple equality filter on metadata.
        Returns [(record, cosine_similarity), ...] best first."""
        q = np.asarray(get_embeddings().embed_query(query), dtype=np.float32)
        q /= np.linalg.norm(q)
        sims = self.vectors @ q
        results = []
        for i in np.argsort(-sims, kind="stable"):
            rec = self.records[int(i)]
            if where and any(rec["metadata"].get(key) != val for key, val in where.items()):
                continue
            results.append((rec, float(sims[i])))
            if len(results) >= k:
                break
        return results


def get_store(collection_name: str) -> NumpyStore:
    if collection_name not in _stores:
        _stores[collection_name] = NumpyStore(collection_name).load()
    return _stores[collection_name]
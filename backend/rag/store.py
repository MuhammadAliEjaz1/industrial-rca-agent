"""
Shared vector-store setup (local embeddings + local Chroma, no paid services).

CHANGE vs Phase 3: embeddings now run through ONNX Runtime (Chroma's built-in
all-MiniLM-L6-v2) instead of sentence-transformers + PyTorch. It is the SAME model,
so retrieval quality is equivalent, but it needs no torch - which fixed the
'c10.dll initialization failed' crash on this laptop.

IMPORTANT: vectors made by the old torch version and this ONNX version differ in the
last decimal places. Rebuild the store once (python rag/03_build_vectorstore.py) so
documents and queries come from the same embedder.
"""

from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

RAG_DIR = Path(__file__).resolve().parent
CHROMA_DIR = RAG_DIR / "chroma_db"

MANUALS_COLLECTION = "manuals"
INCIDENTS_COLLECTION = "incidents"

_embeddings = None


class OnnxMiniLMEmbeddings(Embeddings):
    """Adapts Chroma's ONNX MiniLM embedder to LangChain's Embeddings interface.
    Chroma's implementation L2-normalizes its output, which retriever._similarity assumes."""

    def __init__(self):
        self._fn = ONNXMiniLM_L6_V2()  # downloads ~80MB model on first use, then cached

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in v] for v in self._fn(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embeddings() -> OnnxMiniLMEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OnnxMiniLMEmbeddings()
    return _embeddings


def get_store(collection_name: str) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )

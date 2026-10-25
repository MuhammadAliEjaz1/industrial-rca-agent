"""
Phase 3: Build the local vector store (NumPy files, no Chroma index).

Run from the backend/ folder:
    python rag/03_build_vectorstore.py

Rebuilds from scratch each run. Two collections:
  - manuals   : authored technical documentation, chunked by section
  - incidents : historical failure incidents from industrial_rca.db
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.loaders import load_manual_chunks, load_incident_records, DB_PATH
from rag.store import NumpyStore, STORE_DIR, MANUALS_COLLECTION, INCIDENTS_COLLECTION


def add_records(collection_name: str, records: list[dict]):
    NumpyStore.build(collection_name, records)
    print(f"  {collection_name}: indexed {len(records)} chunks")


if __name__ == "__main__":
    if STORE_DIR.exists():
        shutil.rmtree(STORE_DIR, ignore_errors=True)
        print(f"Removed old vector store at {STORE_DIR}")

    print("Loading documents...")
    manuals = load_manual_chunks()
    print(f"  manual chunks: {len(manuals)}")

    print("Embedding and indexing...")
    add_records(MANUALS_COLLECTION, manuals)

    if DB_PATH.exists():
        incidents = load_incident_records()
        add_records(INCIDENTS_COLLECTION, incidents)
    else:
        print(f"WARNING: {DB_PATH} not found - skipping incidents collection. Run Phase 1 first.")

    print(f"\nDone. Vector store saved to {STORE_DIR}")
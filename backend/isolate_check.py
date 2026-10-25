import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
print("start", flush=True)
from rag.store import get_embeddings
emb = get_embeddings()
texts = [f"incident {i} torque 40 rpm 1500 tool wear 200 " * 10 for i in range(339)]

print("embedding 339 texts in one batch...", flush=True)
vecs = emb.embed_documents(texts)
print("embedding OK", len(vecs), flush=True)

import chromadb, tempfile
col = chromadb.PersistentClient(path=tempfile.mkdtemp()).get_or_create_collection("test_col")
print("adding 339 vectors...", flush=True)
col.add(ids=[str(i) for i in range(339)], embeddings=vecs)
print("add OK, count =", col.count(), flush=True)
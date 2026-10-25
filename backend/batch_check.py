import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import chromadb, tempfile, random
col = chromadb.PersistentClient(path=tempfile.mkdtemp()).get_or_create_collection("test_col")
random.seed(0)
for start in range(0, 339, 25):
    n = min(25, 339 - start)
    ids = [str(start + i) for i in range(n)]
    vecs = [[random.random() for _ in range(384)] for _ in range(n)]
    col.add(ids=ids, embeddings=vecs)
    print("added up to", start + n, flush=True)
print("done, count =", col.count(), flush=True)
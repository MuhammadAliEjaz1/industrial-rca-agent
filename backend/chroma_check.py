import chromadb, tempfile
print("start", flush=True)
client = chromadb.PersistentClient(path=tempfile.mkdtemp())
col = client.get_or_create_collection("test_col")
col.add(ids=["a", "b"], embeddings=[[0.1] * 384, [0.2] * 384])
print("add OK, count =", col.count(), flush=True)
print(col.query(query_embeddings=[[0.1] * 384], n_results=1)["ids"], flush=True)
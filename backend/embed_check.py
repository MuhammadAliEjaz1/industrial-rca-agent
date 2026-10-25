import math
print("start", flush=True)
from rag.store import get_embeddings
print("imported", flush=True)
emb = get_embeddings()
print("embedder created", flush=True)
v = emb.embed_query("torque overload")
print(len(v), round(math.sqrt(sum(x * x for x in v)), 3), flush=True)
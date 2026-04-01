# ================================================================
# FILE: qdrant_setup.py
# Run ONCE to upload embeddings from Kaggle export to Qdrant
# ================================================================

# qdrant_setup.py
# Run once: python qdrant_setup.py
# This migrates your FAISS embeddings to persistent Qdrant cloud.
 
import numpy as np
import json
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct)
from config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, EMBED_DIM, EXPORT_DIR
 
print("Connecting to Qdrant Cloud...")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
 
# Delete existing collection if re-running
try:
    client.delete_collection(COLLECTION_NAME)
    print("Deleted existing collection.")
except Exception:
    pass
 
# Create fresh collection
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config =VectorParams(
        size     =EMBED_DIM,
        distance =Distance.COSINE,
    )
)
print("Created collection:", COLLECTION_NAME)
 
# Load embeddings from Kaggle export
embeddings = np.load(EXPORT_DIR + "train_embeddings.npy")
with open(EXPORT_DIR + "embed_metadata.json") as f:
    metadata = json.load(f)
 
print("Loaded {} embeddings, dim={}".format(
    len(embeddings), embeddings.shape[1]))
 
# L2-normalise for cosine similarity
norms  = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
normed = (embeddings / norms).astype(np.float32)
 
# Upload in batches
points     = []
batch_size = 100
 
for i, (emb, meta) in enumerate(zip(normed, metadata)):
    points.append(PointStruct(
        id      =i,
        vector  =emb.tolist(),
        payload ={
            "trader_id":  meta["trader_id"],
            "last_date":  str(meta["last_date"])[:10],
            "true_label": meta["true_label"],
        }
    ))
 
print("Uploading {} points in batches of {}...".format(
    len(points), batch_size))
 
for i in range(0, len(points), batch_size):
    batch = points[i:i + batch_size]
    client.upsert(collection_name=COLLECTION_NAME, points=batch)
    print("  Uploaded {}/{}".format(
        min(i + batch_size, len(points)), len(points)))
 
print("Upload complete.")
 
# Verify with a test query
test_vec = normed[0].tolist()
results  = client.search(
    collection_name=COLLECTION_NAME,
    query_vector   =test_vec,
    limit          =3,
)
print("\\nTest retrieval (sanity check):")
for r in results:
    print("  Score {:.3f}: {} on {} — {}".format(
        r.score,
        r.payload["trader_id"],
        r.payload["last_date"],
        r.payload["true_label"],
    ))
 
print("\\nQdrant setup complete. Collection is ready for production.")

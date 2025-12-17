import os
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------------------------
# LOAD ENV
# ---------------------------------
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_CLUSTER_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
BOOK_PATH = os.getenv("BOOK_CONTENT_PATH")

COLLECTION = "RAG_AI_TextBook_Data"   # 👈 FIXED, single source of truth

# ---------------------------------
# QDRANT CLIENT
# ---------------------------------
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# ---------------------------------
# RESET COLLECTION (SAFE)
# ---------------------------------
existing = [c.name for c in qdrant.get_collections().collections]

if COLLECTION in existing:
    qdrant.delete_collection(collection_name=COLLECTION)
    print("🗑️ Existing collection deleted")

qdrant.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(
        size=384,                     # MUST MATCH MODEL
        distance=Distance.COSINE,
    ),
)

print("✅ Collection created with vector size 384")

# ---------------------------------
# EMBEDDING MODEL (FREE / OFFLINE)
# ---------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 dims

# ---------------------------------
# CHUNK FUNCTION
# ---------------------------------
def chunk_text(text, size=400, overlap=50):
    words = text.split()
    for i in range(0, len(words), size - overlap):
        yield " ".join(words[i:i + size])

# ---------------------------------
# INGEST BOOK FILES
# ---------------------------------
points = []

for root, _, files in os.walk(BOOK_PATH):
    for file in files:
        if file.endswith(".md"):
            file_path = os.path.join(root, file)

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = list(chunk_text(content))
            embeddings = model.encode(chunks, show_progress_bar=False)

            for text, emb in zip(chunks, embeddings):
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=emb.tolist(),
                        payload={
                            "text": text,
                            "source": file,
                        },
                    )
                )

# ---------------------------------
# UPSERT TO QDRANT
# ---------------------------------
qdrant.upsert(
    collection_name=COLLECTION,
    points=points,
)

print(f"✅ Ingested {len(points)} chunks successfully (FREE, NO API)")

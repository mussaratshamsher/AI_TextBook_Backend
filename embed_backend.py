import os
import uuid
from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# ---------------------------------
# LOAD ENV
# ---------------------------------
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_CLUSTER_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BOOK_CONTENT_PATH = os.getenv("BOOK_CONTENT_PATH")

COLLECTION_NAME = "embeding-physical-ai"

if not BOOK_CONTENT_PATH:
    raise ValueError("❌ BOOK_CONTENT_PATH is not set in .env")

# ---------------------------------
# RESOLVE BOOK PATH (CRITICAL FIX)
# ---------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ABS_BOOK_PATH = os.path.abspath(os.path.join(BASE_DIR, BOOK_CONTENT_PATH))

print("📂 BOOK_CONTENT_PATH (env):", BOOK_CONTENT_PATH)
print("📂 BOOK_CONTENT_PATH (resolved):", ABS_BOOK_PATH)

if not os.path.exists(ABS_BOOK_PATH):
    raise FileNotFoundError(f"❌ Book path does not exist: {ABS_BOOK_PATH}")

# ---------------------------------
# CLIENTS
# ---------------------------------
client_qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

client_gemini = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------------
# TEXT CHUNKING
# ---------------------------------
def chunk_text(text: str, chunk_size=500, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks

# ---------------------------------
# EMBEDDING (MATCHES AGENT FILE)
# ---------------------------------
def get_embedding(text: str):
    response = client_gemini.models.embed_content(
        model="text-embedding-004",
        contents=text
    )
    return response.embeddings[0].values

# ---------------------------------
# CREATE COLLECTION IF NEEDED
# ---------------------------------
def ensure_collection():
    collections = [c.name for c in client_qdrant.get_collections().collections]

    if COLLECTION_NAME not in collections:
        vector_size = len(get_embedding("dimension check"))
        client_qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        print(f"✅ Created collection: {COLLECTION_NAME}")
    else:
        print(f"ℹ️ Collection '{COLLECTION_NAME}' already exists")

# ---------------------------------
# LOAD MARKDOWN FILES (FAIL LOUD)
# ---------------------------------
def load_markdown_files():
    documents = []

    for root, _, files in os.walk(ABS_BOOK_PATH):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    documents.append({
                        "text": f.read(),
                        "source": os.path.relpath(file_path, ABS_BOOK_PATH)
                    })

    if not documents:
        raise RuntimeError(f"❌ No .md files found in {ABS_BOOK_PATH}")

    print(f"📘 Found {len(documents)} markdown files")
    return documents

# ---------------------------------
# INGEST INTO QDRANT
# ---------------------------------
def ingest_documents():
    documents = load_markdown_files()
    total_chunks = 0

    for doc in documents:
        chunks = chunk_text(doc["text"])
        points = []

        for chunk in chunks:
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=get_embedding(chunk),
                    payload={
                        "text": chunk,
                        "source": doc["source"]
                    }
                )
            )

        client_qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

        total_chunks += len(points)

    print(f"✅ Ingested {total_chunks} chunks from {len(documents)} files")

# ---------------------------------
# RUN
# ---------------------------------
if __name__ == "__main__":
    ensure_collection()
    ingest_documents()

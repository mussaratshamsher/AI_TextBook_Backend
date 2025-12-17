import os
from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# ----------------------------
# LOAD ENV
# ----------------------------
load_dotenv()

app = FastAPI()

# Gemini only for answer formatting
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Qdrant client
qdrant = QdrantClient(
    url=os.getenv("QDRANT_CLUSTER_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

COLLECTION = os.getenv("QDRANT_CLUSTER_NAME", "RAG_AI_TextBook_Data")

# Embedding model (same as ingest.py)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# Request model
# ----------------------------
class Query(BaseModel):
    question: str

# ----------------------------
# Query endpoint
# ----------------------------
@app.post("/query")
def query_book(data: Query):
    # 1️⃣ Embed query (local, free)
    query_vector = embedding_model.encode(data.question).tolist()

    # 2️⃣ Search Qdrant
    hits = qdrant.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=3
    )

    if not hits:
        return {"answer": "Not found in book."}

    # 3️⃣ Build context for Gemini
    context = "\n\n".join(
        f"[Source: {h.payload['source']}]\n{h.payload['text']}"
        for h in hits
    )

    prompt = f"""
Answer ONLY using the context below.
If not present, say exactly: "Not found in book."

Context:
{context}

Question:
{data.question}
"""

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    return {
        "answer": response.text,
        "sources": list({h.payload["source"] for h in hits})
    }

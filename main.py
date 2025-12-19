from typing import List, Optional
import os
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from openai import AsyncOpenAI

# ---------------------------------
# ENV + LOGGING
# ---------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
)
logger = logging.getLogger("rag_backend")

# ---------------------------------
# CONFIG
# ---------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_URL = os.getenv("QDRANT_CLUSTER_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "RAG_AI_TextBook_Data"

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY missing")

# ---------------------------------
# CLIENTS
# ---------------------------------

# Groq (OpenAI-compatible)
llm_client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

# Qdrant
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# Embeddings (local & fast)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------
# FASTAPI APP
# ---------------------------------

app = FastAPI(title="Physical AI Textbook RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------
# SCHEMAS
# ---------------------------------

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]


class HealthResponse(BaseModel):
    status: str


# ---------------------------------
# HELPERS
# ---------------------------------

def build_prompt(question: str, contexts: List[str]) -> str:
    joined_context = "\n\n---\n\n".join(contexts)

    return f"""
You are a STRICT textbook tutor.

Rules:
- Answer ONLY using the context below.
- Do NOT use external knowledge.
- Do NOT guess.
- If the answer is not explicitly found, reply exactly:
"Not found in book."

Context:
{joined_context}

Question:
{question}
""".strip()


# ---------------------------------
# ROUTES
# ---------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.post("/query", response_model=QueryResponse)
async def query_book(req: QueryRequest):
    # 1️⃣ Embed question
    try:
        query_vector = embedding_model.encode(req.question).tolist()
    except Exception:
        logger.exception("Embedding failed")
        raise HTTPException(status_code=500, detail="Embedding failed")

    # 2️⃣ Search Qdrant
    try:
        search_result = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=req.top_k,
            with_payload=True,
            with_vectors=False,
        )
    except Exception:
        logger.exception("Qdrant search failed")
        raise HTTPException(status_code=500, detail="Vector search failed")

    # 3️⃣ Filter low-quality matches
    hits = [p for p in search_result.points if p.score >= 0.30]

    if not hits:
        return QueryResponse(
            answer="Not found in book.",
            sources=[],
        )

    # 4️⃣ Build context
    contexts = []
    sources = set()

    for h in hits:
        payload = h.payload or {}
        text = payload.get("text")
        if not text:
            continue

        title = payload.get("title", "Book")
        heading = payload.get("heading", "")
        source = payload.get("source", "unknown")

        contexts.append(f"[{title} | {heading}]\n{text}")
        sources.add(source)

    if not contexts:
        return QueryResponse(
            answer="Not found in book.",
            sources=[],
        )

    prompt = build_prompt(req.question, contexts)

    # 5️⃣ Generate answer from Groq
    try:
        completion = await llm_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        answer = completion.choices[0].message.content.strip()
    except Exception:
        logger.exception("LLM generation failed")
        raise HTTPException(status_code=500, detail="Generation failed")

    return QueryResponse(
        answer=answer,
        sources=list(sources),
    )

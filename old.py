
# from typing import List, Optional
# import os
# import logging

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel

# import google.generativeai as genai
# from qdrant_client import QdrantClient
# from qdrant_client.http import models as rest
# from sentence_transformers import SentenceTransformer
# from dotenv import load_dotenv


# # ---------------------------
# # ENV + LOGGING
# # ---------------------------

# load_dotenv()

# logging.basicConfig(
#     level=logging.INFO,
#     format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
# )
# logger = logging.getLogger("rag_backend")


# # ---------------------------
# # CONFIG
# # ---------------------------

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# QDRANT_URL = os.getenv("QDRANT_CLUSTER_URL")
# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# QDRANT_COLLECTION = "RAG_AI_TextBook_Data"

# EMBEDDING_MODEL = "models/embedding-001"
# CHAT_MODEL = "gemini-1.5-flash"


# # ---------------------------
# # INIT GEMINI + QDRANT
# # ---------------------------

# def init_gemini():
#     if not GROQ_API_KEY:
#         raise RuntimeError("GROQ_API_KEY missing")
#     genai.configure(api_key=GROQ_API_KEY)
#     logger.info("Gemini configured successfully")


# def init_qdrant() -> QdrantClient:
#     client = QdrantClient(
#         url=QDRANT_URL,
#         api_key=QDRANT_API_KEY,
#     )
#     logger.info("Connected to Qdrant")
#     return client


# # ---------------------------
# # EMBEDDING & GENERATION
# # ---------------------------

# def embed_text(text: str) -> List[float]:
#     if not text.strip():
#         raise ValueError("Empty text cannot be embedded")

#     res = genai.embed_content(
#         model=EMBEDDING_MODEL,
#         content=text,
#         task_type="retrieval_query",
#     )
#     return res["embedding"]


# def generate_answer(prompt: str) -> str:
#     model = genai.GenerativeModel(CHAT_MODEL)
#     response = model.generate_content(prompt)
#     return response.text or ""


# # ---------------------------
# # FASTAPI APP
# # ---------------------------

# app = FastAPI(title="Physical AI Textbook RAG API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # safe for dev
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# qdrant: QdrantClient | None = None


# @app.on_event("startup")
# def startup_event():
#     global qdrant
#     init_gemini()
#     qdrant = init_qdrant()
#     logger.info("Startup complete")


# # ---------------------------
# # SCHEMAS
# # ---------------------------

# class ChatRequest(BaseModel):
#     query: str
#     top_k: int = 3
#     chapter_slug: Optional[str] = None


# class ChatResponse(BaseModel):
#     answer: str
#     contexts: List[dict]


# class HealthResponse(BaseModel):
#     status: str


# # ---------------------------
# # HELPERS
# # ---------------------------

# def search_qdrant(
#     query_embedding: List[float],
#     top_k: int,
#     chapter_slug: Optional[str] = None,
# ):
#     search_filter = None

#     if chapter_slug:
#         search_filter = rest.Filter(
#             must=[
#                 rest.FieldCondition(
#                     key="slug",
#                     match=rest.MatchValue(value=chapter_slug),
#                 )
#             ]
#         )

#     response = qdrant.query_points(
#         collection_name=QDRANT_COLLECTION,
#         query=query_embedding,
#         limit=top_k,
#         query_filter=search_filter,
#         with_payload=True,
#         with_vectors=False,
#     )

#     return response.points


# def build_rag_prompt(query: str, contexts: List[dict]) -> str:
#     joined_context = "\n\n---\n\n".join(
#         f"[{c.get('title','')} - {c.get('heading','')}]\n{c.get('text','')}"
#         for c in contexts
#     )

#     return f"""
# You are a tutor for a Physical AI & Humanoid Robotics textbook.
# Answer ONLY using the context below.
# If the answer is not present, say exactly:
# "Not found in book."

# Context:
# {joined_context}

# Question:
# {query}
# """.strip()


# # ---------------------------
# # ROUTES
# # ---------------------------

# @app.get("/health", response_model=HealthResponse)
# def health():
#     return HealthResponse(status="ok")


# @app.post("/query", response_model=ChatResponse)
# def query(req: ChatRequest):
#     try:
#         query_emb = embed_text(req.query)
#     except Exception as e:
#         logger.exception("Embedding failed")
#         raise HTTPException(status_code=500, detail="Embedding failed")

#     try:
#         points = search_qdrant(
#             query_emb,
#             top_k=req.top_k,
#             chapter_slug=req.chapter_slug,
#         )
#     except Exception:
#         logger.exception("Qdrant search failed")
#         raise HTTPException(status_code=500, detail="Vector search failed")

#     contexts = []
#     for p in points:
#         payload = p.payload or {}
#         contexts.append(
#             {
#                 "text": payload.get("text", ""),
#                 "title": payload.get("title", ""),
#                 "slug": payload.get("slug", ""),
#                 "heading": payload.get("heading", ""),
#                 "score": p.score,
#             }
#         )

#     if not contexts:
#         return ChatResponse(
#             answer="Not found in book.",
#             contexts=[],
#         )

#     prompt = build_rag_prompt(req.query, contexts)

#     try:
#         answer = generate_answer(prompt)
#     except Exception:
#         logger.exception("Gemini generation failed")
#         raise HTTPException(status_code=500, detail="Generation failed")

#     return ChatResponse(answer=answer, contexts=contexts)

# ----------------------------

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, Runner


# ----------------------------
# LOAD ENV
load_dotenv()
#  MODEL SETUP
# ----------------------------
MODEL_NAME = "gemini-2.0-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

openai_client = AsyncOpenAI(
    api_key = GEMINI_API_KEY,
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model = MODEL_NAME,
    openai_client = openai_client
)
config = RunConfig(
    model = model,
    tracing_disabled = True
)
# ----------------------------
# FASTAPI APP
# ----------------------------
app = FastAPI(title="RAG AI Textbook API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # frontend dev safe
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {"status": "RAG backend running locally"}

# ----------------------------
# CLIENTS
# ----------------------------
qdrant = QdrantClient(
    url=os.getenv("QDRANT_CLUSTER_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

COLLECTION = "RAG_AI_TextBook_Data"  # 🔒 SAME AS INGEST

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 dims

# ----------------------------
# AGENT (STRICT RAG)
# ----------------------------
rag_agent = Agent(
    name="Textbook RAG Agent"
)
agent_input=f"""
You answer questions ONLY using the provided book context.
Rules:
- Use only the given context.
- Do NOT use external knowledge.
- If the answer is not explicitly found, respond exactly:
  "Not found in book."
"""
# ----------------------------
# REQUEST MODEL
# ----------------------------
class Query(BaseModel):
    question: str

# ----------------------------
# QUERY ENDPOINT
# ----------------------------
@app.post("/query")
async def query_book(data: Query):
    # 1️⃣ Embed query
    query_vector = embedding_model.encode(data.question).tolist()

    # 2️⃣ Query Qdrant (NEW SDK)
    response = qdrant.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=3,
    )
    hits = response.points
    if not hits:
        return {
            "answer": "Not found in book.",
            "sources": [],
        }

    # 3️⃣ Build context
    context = "\n\n".join(
        f"[Source: {h.payload.get('source', 'unknown')}]\n{h.payload.get('text', '')}"
        for h in hits
    )

    # 4️⃣ Agent prompt
    agent_input = f"""
SYSTEM RULES (STRICT):
You answer questions ONLY using the provided book context.
- Use only the given context
- Do NOT use external knowledge
- If the answer is not explicitly found, reply exactly:
"Not found in book."

--------------------

Context:
{context}

Question:
{data.question}
"""

    # 5️⃣ Run agent
    result = await Runner.run(
        rag_agent,
        agent_input,
        run_config=config
    )

    return {
        "answer": result.final_output.strip()
        if result.final_output
        else "Not found in book.",
        "sources": list({h.payload.get("source", "unknown") for h in hits}),
    }

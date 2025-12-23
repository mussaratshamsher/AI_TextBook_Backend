import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
import google.generativeai as genai
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_CLUSTER_URL1")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY1")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("gemini_embed_key")  # Required for embedding queries

COLLECTION_NAME = "embeding-physical-ai"
app = FastAPI(title="Book RAG Agent")

# -----------------------------
# Initialize Clients
# -----------------------------
client_qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
client_groq = Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# -----------------------------
# Pydantic Request Model
# -----------------------------
class QueryRequest(BaseModel):
    question: str

# -----------------------------
# Embedding Function
# -----------------------------
def get_embedding(text: str):
    """Generate embedding using Gemini (must match ingestion model)."""
    response = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document",
    )
    return response['embedding']

# -----------------------------
# Retrieve Context from Qdrant
# -----------------------------
def retrieve_context(query: str, top_k: int = 3):
    """Get top-k most relevant chunks from Qdrant for a query."""
    query_vector = get_embedding(query)

    search_result = client_qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True
    )

    if not search_result:
        return ""

    return "\n\n".join(hit.payload["text"] for hit in search_result)

# -----------------------------
# Chat Endpoint
# -----------------------------
@app.post("/chat")
async def chat(request: QueryRequest):
    # 1️⃣ Retrieve context
    context = retrieve_context(request.question)

    if not context:
        return {"answer": "I'm sorry, I couldn't find any information about that in the book."}

    # Optional: truncate context to avoid token overflow (Groq LLM)
    context = context[:8000]

    # 2️⃣ Build the system prompt (hallucination-safe)
    system_prompt = f"""
You are a helpful AI assistant specialized in answering questions about a Physical AI 
and Humanoid Robotics book.

Use the provided CONTEXT to answer the USER QUESTION. 
Do NOT use external knowledge. Answer strictly from the CONTEXT.
If the answer is not in the context, politely state that you only have information regarding the book content.

CONTEXT:
{context}
"""

    # 3️⃣ Generate answer using Groq LLM
    try:
        chat_completion = client_groq.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.question},
            ],
            model="llama-3.3-70b-versatile"  # or your preferred Groq model
        )

        answer = chat_completion.choices[0].message.content
        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# Health Check
# -----------------------------
@app.get("/", summary="Health Check")
def read_root():
    return {"status": "ok", "message": "RAG Agent Service is running"}

# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

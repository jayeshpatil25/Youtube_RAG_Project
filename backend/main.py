"""
FastAPI Backend — YouTube RAG Chat API
Endpoints: load video, chat, status
"""

import os
import sys
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

sys.path.insert(0, os.path.dirname(__file__))

# Load .env from project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

from rag_pipeline import RAGPipeline

# ── Global State ──────────────────────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
pipeline: RAGPipeline = None
ingestion_status = {
    "state": "idle",          # idle | loading | ready | error
    "message": "",
    "video_url": "",
    "stats": {},
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        print("WARNING: GROQ_API_KEY not set in .env file!")
    else:
        print("Groq API key loaded from .env")

    print("Initializing RAG pipeline (loading embedding model)...")
    pipeline = RAGPipeline(groq_api_key=GROQ_API_KEY)
    print("Backend ready!")
    yield
    print("Shutting down...")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="YouTube RAG Chat API",
    description="Semantic Chunking + ChromaDB + BM25 + RRF + Groq",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ──────────────────────────────────────────────────

class LoadVideoRequest(BaseModel):
    youtube_url: str


class ChatRequest(BaseModel):
    question: str


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run_ingestion(youtube_url: str):
    """Background task: run the full ingestion pipeline."""
    global ingestion_status
    try:
        ingestion_status["state"] = "loading"
        ingestion_status["message"] = "Fetching transcript and processing..."
        ingestion_status["video_url"] = youtube_url

        stats = pipeline.ingest_video(youtube_url)

        ingestion_status["state"] = "ready"
        ingestion_status["message"] = f"Video indexed successfully! ({stats['chunk_count']} semantic chunks)"
        ingestion_status["stats"] = stats
    except Exception as e:
        ingestion_status["state"] = "error"
        ingestion_status["message"] = str(e)
        ingestion_status["stats"] = {}


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "groq_configured": bool(GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here"),
        "pipeline_ready": pipeline is not None,
        "video_loaded": pipeline.is_ready if pipeline else False,
    }


@app.post("/api/load-video")
async def load_video(request: LoadVideoRequest, background_tasks: BackgroundTasks):
    """
    Trigger video ingestion in background.
    Poll /api/status to track progress.
    """
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY not configured. Please set it in the .env file.",
        )

    url = request.youtube_url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="YouTube URL cannot be empty.")

    # Reset status
    ingestion_status["state"] = "loading"
    ingestion_status["message"] = "Starting ingestion..."
    ingestion_status["video_url"] = url
    ingestion_status["stats"] = {}

    background_tasks.add_task(_run_ingestion, url)
    return {"message": "Video ingestion started. Poll /api/status for updates."}


@app.get("/api/status")
async def get_status():
    """Return current ingestion status."""
    return ingestion_status


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Answer a question using hybrid RAG over the loaded video.
    Requires a video to be loaded first via /api/load-video.
    """
    if not pipeline or not pipeline.is_ready:
        raise HTTPException(
            status_code=400,
            detail="No video loaded. Please load a YouTube video first.",
        )

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = pipeline.answer(question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

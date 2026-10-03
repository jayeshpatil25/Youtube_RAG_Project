# VidChat - Hybrid Search RAG for YouTube

VidChat is a **Retrieval-Augmented Generation (RAG)** application that lets users ask questions about YouTube videos using their transcripts.

It combines **semantic search and BM25 keyword search** using **Reciprocal Rank Fusion (RRF)** to retrieve relevant context before generating answers with **qwen via Groq**.

## Features

*  YouTube transcript extraction
*  Semantic transcript chunking
*  Hybrid retrieval using **ChromaDB + BM25**
*  Reciprocal Rank Fusion (RRF)
*  Llama 3.1-powered question answering
*  Retrieved source chunks for context
*  React frontend with FastAPI backend

## Tech Stack

**Frontend:** React, Vite
**Backend:** Python, FastAPI
**RAG:** LangChain, ChromaDB, BM25, Sentence Transformers
**LLM:** qwen, Groq
**Embeddings:** HuggingFace `all-MiniLM-L6-v2`

## Architecture

```text
YouTube Video
      ↓
Transcript Extraction
      ↓
Semantic Chunking
      ↓
 ┌─────────────┬─────────────┐
 │  ChromaDB   │    BM25     │
 │   Semantic  │   Keyword   │
 │   Search    │   Search    │
 └──────┬──────┴──────┬──────┘
        └──────┬──────┘
               ↓
             RRF
               ↓
        Relevant Context
               ↓
       Llama 3.1 + Groq
               ↓
          Final Answer
```

"""
RAG Pipeline Core
Semantic chunking + ChromaDB + BM25 + Reciprocal Rank Fusion
"""

import os
import re
import shutil
import numpy as np
import nltk
from typing import Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from youtube_transcript_api import YouTubeTranscriptApi
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize

# ── Constants ────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "rag_youtube_collection"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "qwen/qwen3.6-27b"

RAG_PROMPT_TEMPLATE = """
You are an expert assistant that answers questions based on YouTube video transcripts.
Use ONLY the provided context to answer. If the answer is not in the context, say:
"I couldn't find that information in the video transcript."

Be concise, accurate, and helpful.

Context from video transcript:
──────────────────────────────
{context}
──────────────────────────────

Question: {question}

Answer:"""


# ── RAG Pipeline ─────────────────────────────────────────────────────────────

class RAGPipeline:
    """Full RAG pipeline: ingest → semantic chunk → ChromaDB + BM25 → RRF → Groq."""

    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.embeddings_model: Optional[HuggingFaceEmbeddings] = None
        self.vector_store: Optional[Chroma] = None
        self.bm25_index: Optional[BM25Okapi] = None
        self.semantic_chunks: list[Document] = []
        self.llm: Optional[ChatGroq] = None
        self.current_video_url: Optional[str] = None
        self.is_ready: bool = False

        self._init_models()

    def _init_models(self):
        """Load embedding model and LLM (done once at startup)."""
        print("Loading HuggingFace embedding model...")
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=0.1,
            max_tokens=1024,
            groq_api_key=self.groq_api_key,
        )
        print("Models loaded.")

    # ── Transcript Extraction ────────────────────────────────────────────────

    @staticmethod
    def _extract_video_id(url: str) -> str:
        pattern = r"(?:v=|youtu\.be/|embed/|v/|watch\?v=|shorts/)([A-Za-z0-9_-]{11})"
        match = re.search(pattern, url)
        if not match:
            raise ValueError(f"Cannot extract video ID from: {url}")
        return match.group(1)

    def _fetch_transcript(self, youtube_url: str) -> str:
        video_id = self._extract_video_id(youtube_url)
        languages = ["en", "en-US", "en-GB", "a.en"]

        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        except AttributeError:
            transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=languages)
        except Exception:
            transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=languages)

        transcript_list = list(transcript_list)
        text = " ".join(getattr(entry, "text", str(entry)) for entry in transcript_list)
        text = text.replace("\n", " ")
        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ── Semantic Chunking ────────────────────────────────────────────────────

    def _semantic_chunker(
        self,
        text: str,
        breakpoint_percentile: float = 85,
        min_chunk_size: int = 150,
        max_chunk_size: int = 2000,
    ) -> list[Document]:
        sentences = sent_tokenize(text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if not sentences:
            return [
                Document(
                    page_content=text.strip(),
                    metadata={"chunk_id": 0, "chunk_size": len(text.strip()), "source": self.current_video_url},
                )
            ]

        if len(sentences) == 1:
            return [
                Document(
                    page_content=sentences[0],
                    metadata={"chunk_id": 0, "chunk_size": len(sentences[0]), "source": self.current_video_url},
                )
            ]

        embeddings = np.array(self.embeddings_model.embed_documents(sentences))

        similarities = [
            cosine_similarity(embeddings[i].reshape(1, -1), embeddings[i + 1].reshape(1, -1))[0][0]
            for i in range(len(embeddings) - 1)
        ]

        if not similarities:
            return [
                Document(
                    page_content=" ".join(sentences),
                    metadata={"chunk_id": 0, "chunk_size": len(" ".join(sentences)), "source": self.current_video_url},
                )
            ]

        threshold = np.percentile(similarities, 100 - breakpoint_percentile)
        breakpoints = [i + 1 for i, s in enumerate(similarities) if s < threshold]

        boundaries = [0] + breakpoints + [len(sentences)]
        raw_chunks = [
            " ".join(sentences[s:e]).strip()
            for s, e in zip(boundaries[:-1], boundaries[1:])
            if " ".join(sentences[s:e]).strip()
        ]

        # Merge tiny, split huge
        final = []
        buffer = ""
        for chunk in raw_chunks:
            if len(buffer) + len(chunk) < min_chunk_size:
                buffer = (buffer + " " + chunk).strip()
            else:
                if buffer:
                    final.append(buffer)
                buffer = chunk
        if buffer:
            final.append(buffer)

        processed = []
        for chunk in final:
            if len(chunk) <= max_chunk_size:
                processed.append(chunk)
            else:
                sents = sent_tokenize(chunk)
                mid = len(sents) // 2
                processed.append(" ".join(sents[:mid]))
                processed.append(" ".join(sents[mid:]))

        return [
            Document(
                page_content=c,
                metadata={"chunk_id": i, "chunk_size": len(c), "source": self.current_video_url},
            )
            for i, c in enumerate(processed)
            if c.strip()
        ]

    # ── Index Building ───────────────────────────────────────────────────────

    def ingest_video(self, youtube_url: str) -> dict:
        """Full ingestion pipeline for a YouTube video."""
        self.is_ready = False
        self.current_video_url = youtube_url

        # 1. Fetch transcript
        transcript = self._fetch_transcript(youtube_url)
        word_count = len(transcript.split())

        # 2. Semantic chunk
        self.semantic_chunks = self._semantic_chunker(transcript)

        # 3. Build ChromaDB
        if os.path.exists(CHROMA_PERSIST_DIR):
            shutil.rmtree(CHROMA_PERSIST_DIR)
        self.vector_store = Chroma.from_documents(
            documents=self.semantic_chunks,
            embedding=self.embeddings_model,
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        self.dense_retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})

        # 4. Build BM25
        tokenized = [self._tokenize(d.page_content) for d in self.semantic_chunks]
        self.bm25_index = BM25Okapi(tokenized)

        self.is_ready = True
        return {
            "video_url": youtube_url,
            "transcript_chars": len(transcript),
            "word_count": word_count,
            "chunk_count": len(self.semantic_chunks),
            "avg_chunk_size": sum(len(c.page_content) for c in self.semantic_chunks) // len(self.semantic_chunks),
        }

    # ── Retrieval ────────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return [t for t in tokens if len(t) > 2]

    def _bm25_retrieve(self, query: str, top_k: int = 10) -> list[Document]:
        scores = self.bm25_index.get_scores(self._tokenize(query))
        indices = np.argsort(scores)[::-1][:top_k]
        return [self.semantic_chunks[i] for i in indices]

    def _rrf_fuse(self, lists: list[list[Document]], k: int = 60, top_n: int = 5):
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}
        for ranked in lists:
            for rank, doc in enumerate(ranked, 1):
                key = str(doc.metadata.get("chunk_id", doc.page_content[:40]))
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
                doc_map[key] = doc
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [(doc_map[k], v) for k, v in sorted_items]

    def hybrid_retrieve(self, query: str, top_k: int = 10, top_n: int = 5):
        dense = self.dense_retriever.invoke(query)
        sparse = self._bm25_retrieve(query, top_k=top_k)
        return self._rrf_fuse([dense, sparse], top_n=top_n)

    # ── Answer Generation ────────────────────────────────────────────────────

    def answer(self, question: str) -> dict:
        if not self.is_ready:
            raise RuntimeError("Pipeline not ready. Load a video first.")

        fused = self.hybrid_retrieve(question, top_k=10, top_n=5)
        docs = [d for d, _ in fused]
        rrf_scores = [s for _, s in fused]

        context = "\n\n".join(
            f"[Chunk {d.metadata['chunk_id']}]:\n{d.page_content}" for d in docs
        )

        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        chain = prompt | self.llm | StrOutputParser()
        answer_text = chain.invoke({"context": context, "question": question})

        return {
            "answer": answer_text,
            "sources": [
                {
                    "chunk_id": d.metadata["chunk_id"],
                    "content": d.page_content[:300] + ("..." if len(d.page_content) > 300 else ""),
                    "rrf_score": round(s, 6),
                }
                for d, s in zip(docs, rrf_scores)
            ],
        }

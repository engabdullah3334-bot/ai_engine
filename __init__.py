# -*- coding: utf-8 -*-
"""
ai_engine
=========
Standalone AI Engine:
  1. Multilingual Embeddings (`embed_chunks`, `embed_query`)
  2. LLM Text Analysis & RAG (`analyze`)
  3. Standalone Persistent Chatbot (`chat`, `chat_stream`, `launch_ui`)

Usage:
    from ai_engine import AIEngine

    ai = AIEngine()
    vectors = ai.embed_chunks(chunks)
    resp = ai.chat("Summarize the key principles of modular architecture.")
"""

from .engine import AIEngine
from .schemas import AnalysisResult, ChatResponse, EmbeddingRecord

__all__ = [
    "AIEngine",
    "EmbeddingRecord",
    "AnalysisResult",
    "ChatResponse",
]

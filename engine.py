# -*- coding: utf-8 -*-
"""
ai_engine/engine.py
===================
`AIEngine` — Standalone AI Facade providing three clean capabilities:
  1. Numerical Embeddings (`embed_chunks`, `embed_query`) ready for storage.
  2. LLM Text Analysis & RAG (`analyze`) over chunks or queries.
  3. Standalone Chatbot (`chat`, `chat_stream`, `launch_ui`) with
     100% automatic conversation persistence (either in its own local
     SQLite archive when running standalone, or in `StorageEngine` when bound).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterator

from .analyzer import TextAnalyzer
from .chatbot.context_memory import ContextManager
from .chatbot.controller import AIController
from .chatbot.handlers import CodingHandler, GeneralHandler, RAGHandler
from .chatbot.local_archive import LocalConversationArchive
from .embedder import EmbeddingGenerator
from .schemas import AnalysisResult, ChatResponse, EmbeddingRecord
from .settings import (
    DEFAULT_CHAT_DB_PATH,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_OLLAMA_URL,
    BotConfig,
)


class AIEngine:
    """
    Standalone AI Engine (Embeddings + Text Analysis + Persistent Chatbot).

    Usage:
        from ai_engine import AIEngine

        ai = AIEngine()
        vectors = ai.embed_chunks(chunks)
        summary = ai.analyze(chunks, task="summarize")
        reply   = ai.chat("Explain the benefits of modular software architecture")
    """

    def __init__(
        self,
        llm_model: str = DEFAULT_LLM_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        ollama_base_url: str = DEFAULT_OLLAMA_URL,
        chat_db_path: str | Path | None = None,
        chat_store: Any | None = None,
        models_cache_dir: str | Path | None = None,
        custom_embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        custom_chat_fn: Callable[[list[dict[str, str]], str], str] | None = None,
    ) -> None:
        self.controller = AIController(
            model_name=llm_model,
            base_url=ollama_base_url,
            custom_chat_fn=custom_chat_fn,
        )
        self.embedder = EmbeddingGenerator(
            model_name=embedding_model,
            cache_dir=models_cache_dir,
            custom_embed_fn=custom_embed_fn,
        )
        self.analyzer = TextAnalyzer(self.controller)
        self.memory = ContextManager()

        # Bind external `StorageEngine` if provided, else use built-in persistent SQLite archive
        self.archive = (
            chat_store
            if chat_store is not None
            else LocalConversationArchive(chat_db_path or DEFAULT_CHAT_DB_PATH)
        )

        self.general_handler = GeneralHandler(
            self.controller, BotConfig.GENERAL["system_prompt"]
        )
        self.coding_handler = CodingHandler(
            self.controller, BotConfig.CODING["system_prompt"]
        )
        self.rag_handler = RAGHandler(
            self.controller, BotConfig.RAG["system_prompt"]
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 1. Embeddings API (Numerical vectors ready for storage)
    # ══════════════════════════════════════════════════════════════════════════

    def embed_chunks(
        self, chunks: list[Any], use_contextualized: bool = True
    ) -> list[EmbeddingRecord]:
        """Generate numerical embedding vectors for an array of chunks."""
        return self.embedder.embed_chunks(chunks, use_contextualized=use_contextualized)

    def embed_query(self, query: str) -> EmbeddingRecord:
        """Generate a numerical embedding vector for a search query."""
        return self.embedder.embed_query(query)

    def embed(self, chunks_or_query: Any) -> list[EmbeddingRecord] | EmbeddingRecord:
        """Smart convenience dispatcher: embeds a `str` query or a `list` of chunks."""
        if isinstance(chunks_or_query, str):
            return self.embed_query(chunks_or_query)
        return self.embed_chunks(list(chunks_or_query))

    def unload_embedder(self) -> None:
        self.embedder.unload()

    # ══════════════════════════════════════════════════════════════════════════
    # 2. LLM Text Analysis & RAG API
    # ══════════════════════════════════════════════════════════════════════════

    def analyze(
        self,
        inputs: Any,
        task: str = "summarize",
        query: str | None = None,
        system_prompt: str | None = None,
    ) -> AnalysisResult:
        """Analyze chunks or text (`summarize`, `key_points`, `qa`, `critique`, or custom)."""
        return self.analyzer.analyze(
            inputs=inputs, task=task, query=query, system_prompt=system_prompt
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 3. Persistent Chatbot API (Standalone or Project-Linked)
    # ══════════════════════════════════════════════════════════════════════════

    def _mem_key(self, mode: str, session_id: int) -> str:
        return f"{mode}:{session_id}"

    def change_llm_model(self, model_name: str) -> None:
        self.controller.change_model(model_name)

    def create_chat_session(
        self,
        mode: str = "general",
        title: str = "New Chat",
        file_id: int | None = None,
    ) -> int:
        return int(
            self.archive.create_chat_session(
                mode=mode,
                title=title,
                model_name=self.controller.model_name,
                file_id=file_id,
            )
        )

    def resume_chat_session(
        self, session_id: int, mode: str = "general"
    ) -> list[dict[str, Any]]:
        """Load archived messages from persistent DB back into active sliding-window memory."""
        saved = self.archive.get_chat_messages(session_id)
        self.memory.load_messages(self._mem_key(mode, session_id), saved)
        return saved

    def get_active_context(
        self, mode: str = "general", session_id: int | None = None
    ) -> list[dict[str, str]]:
        if session_id is None:
            return []
        key = self._mem_key(mode, session_id)
        ctx = self.memory.get_context(key)
        if not ctx:
            saved = self.archive.get_chat_messages(session_id)
            if saved:
                self.memory.load_messages(key, saved)
                ctx = self.memory.get_context(key)
        return ctx

    def chat_stream(
        self,
        prompt: str,
        session_id: int | None = None,
        mode: str = "general",
        language: str = "Auto-Detect",
        context_chunks: list[Any] | None = None,
        file_id: int | None = None,
    ) -> Iterator[str]:
        """
        Stream assistant reply token-by-token AND automatically persist both the
        user prompt and full assistant reply into the conversation archive.
        """
        if session_id is None:
            session_id = self.create_chat_session(
                mode=mode, title=prompt, file_id=file_id
            )

        key = self._mem_key(mode, session_id)
        if not self.memory.get_context(key):
            prior = self.archive.get_chat_messages(session_id)
            if prior:
                self.memory.load_messages(key, prior)

        self.memory.add_message(key, "user", prompt)
        self.archive.save_chat_message(session_id, "user", prompt)

        if mode == "coding":
            sys_prompt = BotConfig.CODING["system_prompt"]
            messages = self.memory.build_messages(key, sys_prompt)
            stream = self.coding_handler.handle(messages, language=language)
        elif mode == "rag" or context_chunks:
            sys_prompt = BotConfig.RAG["system_prompt"]
            messages = self.memory.build_messages(key, sys_prompt)
            stream = self.rag_handler.handle(messages, context_chunks=context_chunks)
        else:
            sys_prompt = BotConfig.GENERAL["system_prompt"]
            messages = self.memory.build_messages(key, sys_prompt)
            stream = self.general_handler.handle(messages)

        full_reply = ""
        for token in stream:
            full_reply += token
            yield token

        self.memory.add_message(key, "assistant", full_reply)
        self.archive.save_chat_message(session_id, "assistant", full_reply)

    def chat(
        self,
        prompt: str,
        session_id: int | None = None,
        mode: str = "general",
        language: str = "Auto-Detect",
        context_chunks: list[Any] | None = None,
        file_id: int | None = None,
    ) -> ChatResponse:
        """
        Synchronous single-call chatbot turn (auto-creates session if `session_id=None`
        and saves the conversation permanently).
        """
        if session_id is None:
            session_id = self.create_chat_session(
                mode=mode, title=prompt, file_id=file_id
            )
        reply = "".join(
            self.chat_stream(
                prompt=prompt,
                session_id=session_id,
                mode=mode,
                language=language,
                context_chunks=context_chunks,
                file_id=file_id,
            )
        )
        return ChatResponse(
            session_id=session_id,
            mode=mode,
            role="assistant",
            content=reply,
            model_name=self.controller.model_name,
        )

    def get_chat_sessions(
        self, mode: str | None = None, file_id: int | None = None
    ) -> list[dict[str, Any]]:
        return self.archive.get_chat_sessions(mode=mode, file_id=file_id)

    def get_chat_messages(self, session_id: int) -> list[dict[str, Any]]:
        return self.archive.get_chat_messages(session_id)

    def delete_chat_session(self, session_id: int) -> bool:
        return bool(self.archive.delete_chat_session(session_id))

    def get_chat_stats(self) -> dict[str, int]:
        return self.archive.get_chat_stats()

    @staticmethod
    def launch_ui() -> None:
        """Launch the calm standalone Web UI without importing UI code into the engine."""
        app_path = Path(__file__).resolve().parent / "ui" / "app.py"
        subprocess.run([sys.executable, str(app_path)], check=False)

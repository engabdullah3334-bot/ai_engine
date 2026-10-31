# -*- coding: utf-8 -*-
"""
ai_engine/chatbot/context_memory.py
===================================
Sliding-window + token-budget context manager with archive-session hydration.
"""

from __future__ import annotations

from typing import Any

from ..settings import MemoryConfig


class ContextManager:
    """Manages active conversational context per session/tab using a sliding window."""

    def __init__(
        self,
        window_size: int = MemoryConfig.SLIDING_WINDOW_SIZE,
        max_tokens: int = MemoryConfig.MAX_CONTEXT_TOKENS,
    ) -> None:
        self.window_size = window_size
        self.max_tokens = max_tokens
        self.session_memory: dict[str, list[dict[str, str]]] = {}

    def add_message(self, key: str, role: str, content: str) -> None:
        bucket = self.session_memory.setdefault(key, [])
        bucket.append({"role": role, "content": content})
        if len(bucket) > self.window_size * 3:
            # Keep generous display history in memory; `build_messages` trims for LLM context
            self.session_memory[key] = bucket[-(self.window_size * 3) :]

    def load_messages(self, key: str, messages: list[dict[str, Any]]) -> None:
        """Hydrate active memory from archived SQLite messages when resuming a session."""
        self.session_memory[key] = [
            {"role": str(m["role"]), "content": str(m["content"])} for m in messages
        ]

    def get_context(self, key: str) -> list[dict[str, str]]:
        return list(self.session_memory.get(key, []))

    def clear(self, key: str) -> None:
        self.session_memory[key] = []

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return int(len(text.split()) * 1.3)

    def build_messages(self, key: str, system_prompt: str) -> list[dict[str, str]]:
        """Build token-safe sliding-window message list for the LLM."""
        raw_messages = self.get_context(key)[-self.window_size :]
        current_tokens = self.estimate_tokens(system_prompt)
        safe_messages: list[dict[str, str]] = []

        for msg in reversed(raw_messages):
            msg_tokens = self.estimate_tokens(msg["content"])
            if current_tokens + msg_tokens < self.max_tokens:
                safe_messages.insert(0, msg)
                current_tokens += msg_tokens
            else:
                break

        return safe_messages


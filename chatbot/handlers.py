# -*- coding: utf-8 -*-
"""
ai_engine/chatbot/handlers.py
=============================
Specialized chat handlers: `GeneralHandler`, `CodingHandler`, and `RAGHandler`.
"""

from __future__ import annotations

from typing import Any, Iterator

from .controller import AIController


class GeneralHandler:
    """General-purpose context-aware assistant."""

    def __init__(self, controller: AIController, system_prompt: str) -> None:
        self.controller = controller
        self.system_prompt = system_prompt

    def handle(self, messages: list[dict[str, str]]) -> Iterator[str]:
        return self.controller.chat_stream(messages, self.system_prompt)


class CodingHandler:
    """Senior software engineering & code architecture assistant."""

    def __init__(self, controller: AIController, base_system_prompt: str) -> None:
        self.controller = controller
        self.base_system_prompt = base_system_prompt

    def handle(
        self, messages: list[dict[str, str]], language: str = "Auto-Detect"
    ) -> Iterator[str]:
        if language == "Auto-Detect":
            spec = " Analyze the task and choose the cleanest language & design."
        else:
            spec = f" Focus specifically on idiomatic {language} development."
        return self.controller.chat_stream(messages, self.base_system_prompt + spec)


class RAGHandler:
    """Document-grounded RAG conversational handler over retrieved chunks."""

    def __init__(self, controller: AIController, base_system_prompt: str) -> None:
        self.controller = controller
        self.base_system_prompt = base_system_prompt

    def handle(
        self,
        messages: list[dict[str, str]],
        context_chunks: list[Any] | None = None,
    ) -> Iterator[str]:
        if not context_chunks:
            return self.controller.chat_stream(messages, self.base_system_prompt)

        formatted: list[str] = []
        for idx, item in enumerate(context_chunks, start=1):
            if isinstance(item, str):
                formatted.append(f"[{idx}] {item}")
            elif hasattr(item, "contextualized_content"):
                formatted.append(f"[{idx}] {item.contextualized_content}")
            elif isinstance(item, dict):
                txt = (
                    item.get("contextualized_content")
                    or item.get("content")
                    or item.get("text")
                    or ""
                )
                formatted.append(f"[{idx}] {txt}")
            else:
                formatted.append(f"[{idx}] {item}")

        sys_prompt = (
            f"{self.base_system_prompt}\n\n"
            "Reference Chunks:\n"
            + "\n\n---\n\n".join(formatted)
        )
        return self.controller.chat_stream(messages, sys_prompt)


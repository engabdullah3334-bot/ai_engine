# -*- coding: utf-8 -*-
"""
ai_engine/chatbot/controller.py
===============================
LLM streaming & synchronous controller supporting:
  1. `ollama` Python SDK (when installed),
  2. Native HTTP fallback to `OLLAMA_BASE_URL/api/chat` (zero extra dependencies!),
  3. Optional `custom_chat_fn` for custom cloud providers or offline unit tests.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Callable, Iterator

from ..settings import DEFAULT_LLM_MODEL, DEFAULT_OLLAMA_URL


class AIController:
    """Communicates with Ollama (or an injected LLM callback) with streaming support."""

    def __init__(
        self,
        model_name: str = DEFAULT_LLM_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        custom_chat_fn: Callable[[list[dict[str, str]], str], str] | None = None,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self._custom_chat_fn = custom_chat_fn

    def change_model(self, model_name: str) -> None:
        self.model_name = model_name

    def is_online(self) -> bool:
        if self._custom_chat_fn is not None:
            return True
        req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
        try:
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        system_context: str,
    ) -> Iterator[str]:
        """Yield assistant response chunks token-by-token."""
        if self._custom_chat_fn is not None:
            yield self._custom_chat_fn(messages, system_context)
            return

        full_messages = [{"role": "system", "content": system_context}] + list(messages)

        # Prefer official `ollama` SDK if installed, else direct HTTP streaming
        try:
            import ollama

            stream = ollama.chat(
                model=self.model_name,
                messages=full_messages,
                stream=True,
            )
            for chunk in stream:
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
            return
        except ImportError:
            pass
        except Exception as exc:
            yield f"[Ollama Connection Error: {exc}]"
            return

        # Zero-dependency HTTP streaming directly against Ollama REST API
        payload = json.dumps(
            {"model": self.model_name, "messages": full_messages, "stream": True}
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    obj: dict[str, Any] = json.loads(line)
                    token = obj.get("message", {}).get("content", "")
                    if token:
                        yield token
        except Exception as exc:
            yield f"[Ollama Connection Error ({self.base_url}): {exc}]"

    def chat_once(
        self,
        messages: list[dict[str, str]],
        system_context: str,
    ) -> str:
        """Collect streamed response into a single complete string."""
        return "".join(self.chat_stream(messages, system_context))


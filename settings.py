# -*- coding: utf-8 -*-
"""
ai_engine/settings.py
=====================
Centralized local configuration for `ai_engine` (named `settings.py` and imported
relatively so it never collides with any external project's `config.py`).
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CHAT_DB_PATH = PACKAGE_DIR / "data" / "chat_history.db"

DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class AppConfig:
    APP_TITLE = "AI Engine"
    PAGE_ICON = "🧠"
    LAYOUT = "centered"
    DEFAULT_MODEL = DEFAULT_LLM_MODEL
    AVAILABLE_MODELS = ["llama3", "mistral", "codellama", "gemma"]


class ThemeConfig:
    FONT_FAMILY = "'Inter', 'Segoe UI', sans-serif"


class BotConfig:
    GENERAL = {
        "id": "general",
        "name": "General Assistant",
        "icon": "🌐",
        "avatar": "🤖",
        "system_prompt": (
            "You are a clear, versatile, and helpful AI assistant. "
            "Respond naturally in the user's language based on the context of their message."
        ),
        "welcome": "Ready for conversation. Every session is automatically saved.",
        "placeholder": "Ask anything...",
        "tab_label": "General Chat",
    }
    CODING = {
        "id": "coding",
        "name": "Code Architect",
        "icon": "💻",
        "avatar": "👨‍💻",
        "system_prompt": (
            "You are a senior full-stack software architect. Write clean, modular, "
            "documented code and explain design decisions clearly."
        ),
        "welcome": "Ready to write, review, or refactor code with persistent history.",
        "placeholder": "Describe a coding task or paste a snippet...",
        "tab_label": "Coding Assistant",
    }
    RAG = {
        "id": "rag",
        "name": "Document Analyst",
        "icon": "📚",
        "avatar": "🧠",
        "system_prompt": (
            "You are an accurate research and document analysis assistant. Base your answers "
            "strictly on the supplied document chunks when context is provided."
        ),
    }


class MemoryConfig:
    SLIDING_WINDOW_SIZE = 10
    MAX_CONTEXT_TOKENS = 4096
    HISTORY_DB_PATH = str(DEFAULT_CHAT_DB_PATH)

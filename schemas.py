# -*- coding: utf-8 -*-
"""
ai_engine/schemas.py
====================
Standard output contracts for `AIEngine`:
  - `EmbeddingRecord` (ready for `StorageEngine.save_embeddings`)
  - `AnalysisResult`  (ready for `StorageEngine.save_analysis`)
  - `ChatResponse`    (returned by synchronous `AIEngine.chat`)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class EmbeddingRecord:
    """Numerical embedding vector ready for relational/vector persistence."""

    record_id: str
    text: str
    vector: list[float]
    dimension: int
    model_name: str
    chunk_id: str | None = None
    chunk_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisResult:
    """Structured text analysis output from an LLM over chunks or queries."""

    task: str
    content: str
    model_name: str
    query: str | None = None
    source_chunk_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChatResponse:
    """Complete chat turn response with its persisted `session_id`."""

    session_id: int
    mode: str
    role: str
    content: str
    model_name: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


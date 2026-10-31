# -*- coding: utf-8 -*-
"""
ai_engine/embedder.py
=====================
Lazy-loading multilingual embedding generator (`paraphrase-multilingual-MiniLM-L12-v2`).
Outputs `EmbeddingRecord` objects ready for storage without forcing a specific vector DB!
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from .schemas import EmbeddingRecord
from .settings import DEFAULT_EMBEDDING_MODEL


def _extract_item_fields(item: Any, pos: int, use_context: bool) -> dict[str, Any]:
    if isinstance(item, str):
        return {
            "chunk_id": f"item_{pos:04d}",
            "chunk_index": pos,
            "text": item,
            "metadata": {},
        }
    if use_context and hasattr(item, "contextualized_content"):
        ctx_txt = item.contextualized_content
    else:
        ctx_txt = None

    if hasattr(item, "to_dict") and callable(item.to_dict):
        d = item.to_dict()
    elif isinstance(item, dict):
        d = item
    else:
        d = vars(item)

    txt = str(
        (ctx_txt if use_context and ctx_txt else None)
        or (d.get("contextualized_content") if use_context else None)
        or d.get("content")
        or d.get("text")
        or ""
    )
    meta = dict(d.get("metadata") or {})
    if d.get("section_breadcrumb"):
        meta.setdefault("section_breadcrumb", d["section_breadcrumb"])
    return {
        "chunk_id": str(d.get("chunk_id") or f"chunk_{pos:04d}"),
        "chunk_index": int(d.get("chunk_index") or pos),
        "text": txt,
        "metadata": meta,
    }


class EmbeddingGenerator:
    """Computes dense float embeddings lazily via SentenceTransformers or a custom fn."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        cache_dir: str | Path | None = None,
        custom_embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> None:
        self.model_name = model_name
        if cache_dir:
            os.environ["HF_HOME"] = str(Path(cache_dir).resolve())
        elif "HF_HOME" not in os.environ and Path(r"D:\Models\huggingface").exists():
            os.environ["HF_HOME"] = r"D:\Models\huggingface"

        self._model: Any = None
        self._custom_fn = custom_embed_fn

    def load(self) -> None:
        if self._model is not None or self._custom_fn is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)

    def unload(self) -> None:
        import gc

        self._model = None
        gc.collect()

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._custom_fn is not None:
            return self._custom_fn(texts)
        if self._model is None:
            self.load()
        return self._model.encode(texts, show_progress_bar=False).tolist()

    def embed_chunks(
        self,
        chunks: list[Any],
        use_contextualized: bool = True,
    ) -> list[EmbeddingRecord]:
        """Convert `list[Chunk]` (or `list[str]`) into `list[EmbeddingRecord]`."""
        if not chunks:
            return []

        items = [
            _extract_item_fields(c, pos=i + 1, use_context=use_contextualized)
            for i, c in enumerate(chunks)
        ]
        texts = [it["text"] for it in items]
        vectors = self.encode_texts(texts)

        records: list[EmbeddingRecord] = []
        for it, vec in zip(items, vectors):
            records.append(
                EmbeddingRecord(
                    record_id=f"emb_{it['chunk_id']}",
                    chunk_id=it["chunk_id"],
                    chunk_index=it["chunk_index"],
                    text=it["text"],
                    vector=vec,
                    dimension=len(vec),
                    model_name="custom_embed" if self._custom_fn else self.model_name,
                    metadata=it["metadata"],
                )
            )
        return records

    def embed_query(self, query: str) -> EmbeddingRecord:
        """Embed a single natural-language search query."""
        vec = self.encode_texts([query])[0]
        return EmbeddingRecord(
            record_id="query_embedding",
            text=query,
            vector=vec,
            dimension=len(vec),
            model_name="custom_embed" if self._custom_fn else self.model_name,
        )


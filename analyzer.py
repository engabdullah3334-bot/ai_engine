# -*- coding: utf-8 -*-
"""
ai_engine/analyzer.py
=====================
LLM-powered text analysis & RAG question-answering over chunks or raw text.
"""

from __future__ import annotations

from typing import Any

from .chatbot.controller import AIController
from .schemas import AnalysisResult

TASK_PROMPTS: dict[str, str] = {
    "summarize": (
        "Provide a clear, well-structured summary in concise bullet points for the following content, "
        "preserving domain terminology:"
    ),
    "key_points": (
        "Extract the core ideas, primary concepts, and key takeaways from the following content:"
    ),
    "critique": (
        "Provide a structured analytical critique highlighting strengths, assumptions, and gaps if any:"
    ),
    "qa": (
        "Answer the question accurately based strictly on the provided reference chunks:"
    ),
}


def _collect_chunks_context(inputs: Any) -> tuple[str, list[str]]:
    if isinstance(inputs, str):
        return inputs, []

    if isinstance(inputs, list):
        parts: list[str] = []
        ids: list[str] = []
        for idx, item in enumerate(inputs, start=1):
            if isinstance(item, str):
                parts.append(f"[{idx}] {item}")
                continue
            if hasattr(item, "contextualized_content"):
                txt = item.contextualized_content
                cid = getattr(item, "chunk_id", f"chunk_{idx}")
            elif isinstance(item, dict):
                txt = str(
                    item.get("contextualized_content")
                    or item.get("content")
                    or item.get("text")
                    or ""
                )
                cid = str(item.get("chunk_id") or item.get("vector_uid") or f"chunk_{idx}")
            else:
                txt = str(item)
                cid = f"chunk_{idx}"
            ids.append(cid)
            parts.append(f"[{cid}]\n{txt}")
        return "\n\n---\n\n".join(parts), ids

    if hasattr(inputs, "plain_text"):
        return str(inputs.plain_text), []
    if isinstance(inputs, dict):
        return str(inputs.get("content") or inputs.get("text") or inputs), []
    return str(inputs), []


class TextAnalyzer:
    """Runs structured LLM analyses (`summarize`, `key_points`, `critique`, `qa`, or custom) over chunks."""

    def __init__(self, controller: AIController) -> None:
        self.controller = controller

    def analyze(
        self,
        inputs: Any,
        task: str = "summarize",
        query: str | None = None,
        system_prompt: str | None = None,
    ) -> AnalysisResult:
        context_text, chunk_ids = _collect_chunks_context(inputs)
        instruction = TASK_PROMPTS.get(task, task)

        if query:
            user_prompt = (
                f"{instruction}\n\n"
                f"Question / Query: {query}\n\n"
                f"Reference Chunks:\n{context_text}"
            )
        else:
            user_prompt = f"{instruction}\n\nReference Content:\n{context_text}"

        sys_ctx = system_prompt or (
            "You are an expert text analyst and researcher. Deliver accurate, well-structured "
            "responses in the same language as the query or reference text."
        )
        answer = self.controller.chat_once(
            messages=[{"role": "user", "content": user_prompt}],
            system_context=sys_ctx,
        )
        return AnalysisResult(
            task=task,
            query=query,
            content=answer,
            model_name=self.controller.model_name,
            source_chunk_ids=chunk_ids,
        )

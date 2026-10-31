# -*- coding: utf-8 -*-
"""
ai_engine/__main__.py
=====================
Standalone launcher for `ai_engine`:
  - `python -m ai_engine` or `--ui`   -> Launches calm standalone Web UI
  - `python -m ai_engine --cli`       -> Launches interactive terminal Chatbot (saved chats)
  - `python -m ai_engine --self-test` -> Runs standalone verification suite
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ai_engine import AIEngine
else:
    from .engine import AIEngine


def _configure_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def run_self_test() -> None:
    _configure_utf8()
    print("=" * 64)
    print("  AIEngine — Standalone Self-Test Suite")
    print("=" * 64)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        def _mock_embed(texts: list[str]) -> list[list[float]]:
            return [[float(len(t)), 0.5, 0.25, 0.125] for t in texts]

        def _mock_chat(messages: list[dict[str, str]], sys_ctx: str) -> str:
            last_user = messages[-1]["content"] if messages else ""
            return f"Persisted analytical reply to: {last_user[:40]}"

        ai = AIEngine(
            chat_db_path=tmp / "standalone_chats.db",
            custom_embed_fn=_mock_embed,
            custom_chat_fn=_mock_chat,
        )

        # 1. Embeddings ready for storage
        chunks = [
            {
                "chunk_id": "chk_0001",
                "chunk_index": 1,
                "content": "Modular architecture isolates responsibilities cleanly.",
                "section_breadcrumb": "Part 1 > Chapter 1",
            },
            {
                "chunk_id": "chk_0002",
                "chunk_index": 2,
                "content": "Persistent storage tracks every engine execution.",
                "section_breadcrumb": "Part 1 > Chapter 2",
            },
        ]
        vecs = ai.embed_chunks(chunks)
        q_vec = ai.embed_query("How are responsibilities isolated?")
        assert len(vecs) == 2 and vecs[0].dimension == 4 and q_vec.dimension == 4
        print(f"  [PASS] Embeddings generated -> {len(vecs)} chunk vectors (dim={vecs[0].dimension})")

        # 2. LLM Text Analysis
        analysis = ai.analyze(chunks, task="summarize")
        assert analysis.content and analysis.source_chunk_ids == ["chk_0001", "chk_0002"]
        print(f"  [PASS] Text analysis        -> task='{analysis.task}', sources={len(analysis.source_chunk_ids)}")

        # 3. Standalone Chatbot + 100% Persistent Conversation Archive
        turn1 = ai.chat("Hello AI assistant!")
        turn2 = ai.chat("Continue explaining in the same session.", session_id=turn1.session_id)
        saved_sessions = ai.get_chat_sessions()
        saved_msgs = ai.get_chat_messages(turn1.session_id)
        assert len(saved_sessions) == 1
        assert len(saved_msgs) == 4  # 2 user + 2 assistant messages saved
        assert turn2.session_id == turn1.session_id
        print(
            f"  [PASS] Persistent Chatbot   -> session_id={turn1.session_id}, "
            f"archived_messages={len(saved_msgs)}"
        )

    print("=" * 64)
    print("  ALL AI ENGINE SELF-TESTS PASSED SUCCESSFULLY!")
    print("=" * 64)


def run_cli_chatbot(model: str) -> None:
    _configure_utf8()
    ai = AIEngine(llm_model=model)
    print("=" * 60)
    print(f"  AI Standalone CLI Chatbot (Model: {model})")
    print("  All conversations are automatically persisted to SQLite.")
    print("  Type 'exit' or 'quit' to close.")
    print("=" * 60)

    session_id: int | None = None
    while True:
        try:
            prompt = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nConversation archived. Exiting chatbot.")
            break
        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit", "q"):
            print("Conversation archived. Exiting chatbot.")
            break
        if session_id is None:
            session_id = ai.create_chat_session(mode="general", title=prompt)
            print(f"  [Created & persisted session #{session_id}]")

        print("Assistant > ", end="", flush=True)
        for tok in ai.chat_stream(prompt, session_id=session_id, mode="general"):
            print(tok, end="", flush=True)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="AIEngine Standalone Launcher")
    parser.add_argument("--ui", action="store_true", help="Launch calm standalone Web UI (default)")
    parser.add_argument("--cli", action="store_true", help="Launch terminal CLI Chatbot")
    parser.add_argument("--model", type=str, default="llama3", help="Ollama LLM model tag")
    parser.add_argument("--self-test", action="store_true", help="Run standalone self-test suite")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
    elif args.cli:
        run_cli_chatbot(args.model)
    else:
        AIEngine.launch_ui()


if __name__ == "__main__":
    main()

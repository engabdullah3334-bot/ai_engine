# `ai_engine` — Multilingual Embeddings, LLM Analysis, & Persistent Chatbot

**`ai_engine`** is a standalone Python AI package uniting three core capabilities behind a single clean facade (`AIEngine`): dense multilingual vector embeddings, structured LLM text/chunk analysis, and a multi-session standalone chatbot with 100% automatic SQLite conversation persistence.

### Engineering Problem Solved
Integrating embeddings, document summarization/QA, and conversational assistants typically forces developers to wire together disparate SDKs, manage sliding-window token memory manually, and build custom SQLite tables just to prevent chat history from vanishing when a script exits. Furthermore, AI modules are often coupled tightly to a single project's database, making them impossible to run standalone.

`AIEngine` solves this dual-mode requirement cleanly:
1. **Standalone Mode:** Out of the box, `AIEngine` automatically archives every chat session and message turn into its own embedded SQLite WAL database (`ai_engine/data/chat_history.db`) and provides both a terminal CLI (`--cli`) and a calm browser Web UI (`--ui`).
2. **Project-Bound Mode:** Pass any compatible `chat_store` (such as `StorageEngine`) via `AIEngine(chat_store=db)` to route all sessions and messages directly into a central project database without changing a single line of application code.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Testing](#testing)
- [License and Contributing](#license-and-contributing)

---

## Prerequisites

- **Operating System:** Windows, Linux, or macOS.
- **Python:** `>= 3.10`
- **Database:** Built-in Python `sqlite3` (WAL mode enabled automatically for persistent chat archiving).
- **Python Libraries (`requirements.txt` — for live neural embeddings & Ollama SDK):**
  - `sentence-transformers >= 2.6.0` (multilingual dense vector generation)
  - `ollama >= 0.3.0` (local LLM streaming & synchronous generation)
- **External Service *(Optional)*:**
  - **Ollama** running locally (`ollama serve`, e.g., `ollama pull llama3`) for live generative LLM responses. Both `AIEngine` (`custom_embed_fn` / `custom_chat_fn`) and the standalone Web UI include deterministic offline fallbacks so persistence and workflows can be tested immediately without a live LLM server.

---

## Installation

### Option A: Within the Full Workspace

```powershell
git clone <repository-url> engines
cd engines
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r ai_engine/requirements.txt
```

### Option B: Standalone Copy into Any External Project

```powershell
Copy-Item -Recurse .\ai_engine C:\path\to\your_project\ai_engine
pip install -r C:\path\to\your_project\ai_engine\requirements.txt
```

---

## Usage

### 1. Python API Example

```python
from ai_engine import AIEngine

ai = AIEngine(
    llm_model="llama3",
    embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
)

# 1. Generate dense vectors ready for database storage
chunk_vectors = ai.embed_chunks(chunks, use_contextualized=True)
query_vector  = ai.embed_query("How are responsibilities isolated across modules?")

# 2. Structured LLM analysis over chunks or raw text
summary = ai.analyze(chunks, task="summarize")
answer  = ai.analyze(chunks, task="qa", query="What hashing algorithm is primary?")

# 3. Persistent Chatbot (auto-creates session & saves every turn to SQLite)
turn1 = ai.chat("Explain clean architecture in three bullet points.", mode="general")
turn2 = ai.chat("Show a minimal Python dataclass example.", session_id=turn1.session_id, mode="coding")

print("Session ID:", turn1.session_id)
print("Archived Messages:", len(ai.get_chat_messages(turn1.session_id)))
```

### 2. Command-Line Interface & Calm Standalone Web UI

```powershell
# Launch the calm standalone Web UI (Persistent Chatbot, Text Analysis, & Embeddings tabs)
python -m ai_engine --ui

# Launch interactive terminal chatbot (every turn persisted automatically)
python -m ai_engine --cli --model llama3
```

### 3. Expected Input & Output Shapes

- **Embeddings Output (`EmbeddingRecord.to_dict()`):**
  ```json
  {
    "record_id": "chk_0001_vec",
    "text": "### Part 1 > Chapter 1\n\nModular architecture isolates responsibilities.",
    "vector": [0.0412, -0.0891, "..."],
    "dimension": 384,
    "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
    "chunk_id": "chk_0001",
    "chunk_index": 1,
    "metadata": {}
  }
  ```
- **Analysis Output (`AnalysisResult.to_dict()`):**
  ```json
  {
    "task": "summarize",
    "content": "Concise synthesis of the supplied chunks...",
    "model_name": "llama3",
    "query": null,
    "source_chunk_ids": ["chk_0001", "chk_0002"],
    "metadata": {"chunk_count": 2},
    "created_at": "2026-10-30T19:12:00"
  }
  ```
- **Chat Output (`ChatResponse.to_dict()`):**
  ```json
  {
    "session_id": 1,
    "mode": "general",
    "role": "assistant",
    "content": "Here are three core principles...",
    "model_name": "llama3",
    "created_at": "2026-10-30T19:12:05"
  }
  ```

---

## Architecture

### Package File Structure

```text
ai_engine/
├── __init__.py              # Exports AIEngine, EmbeddingRecord, AnalysisResult, ChatResponse
├── __main__.py              # Standalone launcher (--ui, --cli, --self-test)
├── engine.py                # Unified AIEngine facade (Embeddings + Analysis + Persistent Chat)
├── schemas.py               # EmbeddingRecord, AnalysisResult, and ChatResponse dataclasses
├── settings.py              # Relative configuration (DEFAULT_LLM_MODEL, BotConfig, MemoryConfig)
├── embedder.py              # Lazy-loaded SentenceTransformer embedder + custom_embed_fn support
├── analyzer.py              # Task-oriented chunk/text LLM analyzer (summarize, key_points, qa, critique)
├── chatbot/
│   ├── __init__.py
│   ├── controller.py        # Streaming & synchronous Ollama LLM client + custom_chat_fn hook
│   ├── context_memory.py    # Sliding-window session memory manager
│   ├── handlers.py          # Specialized GeneralHandler, CodingHandler, and RAGHandler
│   └── local_archive.py     # Built-in SQLite WAL persistent conversation store
├── ui/
│   ├── __init__.py
│   └── app.py               # Zero-dependency calm standalone HTTP Web UI (port 8705)
├── requirements.txt
└── README.md
```

### Internal Data Flow

```text
                     ┌──► `EmbeddingGenerator` ──► `list[EmbeddingRecord]`
                     │
Caller ──► `AIEngine`├──► `TextAnalyzer` ────────► `AnalysisResult`
                     │
                     └──► `Chat` (`general` | `coding` | `rag`)
                            ├─► `ContextManager` (sliding window of 10 turns)
                            ├─► `AIController` (streams tokens via Ollama or `custom_chat_fn`)
                            └─► `LocalConversationArchive` (or bound `StorageEngine`) persists user & assistant turns
```

---

## Configuration

Configured via environment variables (read in `ai_engine/settings.py`) and `AIEngine(...)` constructor parameters:

| Variable / Parameter | Source | Default | Description |
| :--- | :--- | :--- | :--- |
| `OLLAMA_BASE_URL` | Env Var / `ollama_base_url` | `"http://localhost:11434"` | Base URL for the Ollama server. |
| `OLLAMA_MODEL` | Env Var / `llm_model` | `"llama3"` | Default LLM model tag for chat and analysis. |
| `embedding_model` | `AIEngine(...)` | `"paraphrase-multilingual-MiniLM-L12-v2"` | SentenceTransformers model name or path (`384`-dim multilingual). |
| `chat_db_path` | `AIEngine(...)` | `ai_engine/data/chat_history.db` | Local SQLite database path for standalone chat persistence. |
| `chat_store` | `AIEngine(...)` | `None` | Optional external persistence object (e.g., `StorageEngine`) implementing chat archive methods. |
| `SLIDING_WINDOW_SIZE` | `MemoryConfig` | `10` | Recent messages kept in active prompt context per session. |
| `--port` (UI CLI flag) | `ui/app.py` | `8705` (auto-fallback) | Localhost port for `python ai_engine/ui/app.py`. |

---

## API Reference

### `AIEngine`

- **`AIEngine(llm_model="llama3", embedding_model="paraphrase-multilingual-MiniLM-L12-v2", ollama_base_url="http://localhost:11434", chat_db_path=None, chat_store=None, models_cache_dir=None, custom_embed_fn=None, custom_chat_fn=None)`**
- **Embeddings:**
  - `embed_chunks(chunks: list[Any], use_contextualized: bool = True) -> list[EmbeddingRecord]`
  - `embed_query(query: str) -> EmbeddingRecord`
  - `embed(chunks_or_query: str | list[Any]) -> EmbeddingRecord | list[EmbeddingRecord]`
  - `unload_embedder() -> None`
- **Text & RAG Analysis:**
  - `analyze(inputs: Any, task: str = "summarize", query: str | None = None, system_prompt: str | None = None) -> AnalysisResult`
    Supported built-in `task` presets: `"summarize"`, `"key_points"`, `"qa"`, `"critique"`, or any custom task instruction.
- **Persistent Chatbot:**
  - `chat(prompt: str, session_id: int | None = None, mode: str = "general", language: str = "Auto-Detect", context_chunks: list[Any] | None = None, file_id: int | None = None) -> ChatResponse`
  - `chat_stream(prompt: str, session_id: int | None = None, mode: str = "general", language: str = "Auto-Detect", context_chunks: list[Any] | None = None, file_id: int | None = None) -> Iterator[str]`
  - `create_chat_session(mode: str = "general", title: str = "New Chat", file_id: int | None = None) -> int`
  - `resume_chat_session(session_id: int, mode: str = "general") -> list[dict[str, Any]]`
  - `get_chat_sessions(mode: str | None = None, file_id: int | None = None) -> list[dict[str, Any]]`
  - `get_chat_messages(session_id: int) -> list[dict[str, Any]]`
  - `delete_chat_session(session_id: int) -> bool`
  - `get_chat_stats() -> dict[str, int]`
  - `AIEngine.launch_ui() -> None` *(staticmethod)*

---

## Testing

Run the deterministic standalone self-test suite (verifies chunk/query embedding generation, structured text analysis, multi-turn session continuity, and automatic SQLite conversation archiving with clean file-handle closure):

```powershell
python -m ai_engine --self-test
```

---

## License and Contributing

- **License:** MIT License.
- **Contributing:** Keep `settings.py` and internal imports strictly relative so `ai_engine` never collides with host-project modules. Write all comments and docs in English, ensure SQLite connections close deterministically, and verify `python -m ai_engine --self-test` passes before opening a Pull Request.

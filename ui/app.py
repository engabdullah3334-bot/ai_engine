# -*- coding: utf-8 -*-
"""
ai_engine/ui/app.py
===================
Minimalist, calm, zero-dependency standalone Web UI for `AIEngine`.
Sole purpose: clearly explain `AIEngine`'s three capabilities (Persistent Chatbot,
LLM Text/Chunk Analysis, and Vector Embeddings) and execute them directly without
intruding on core engine code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_PKG_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from ai_engine import AIEngine
from ai_engine.settings import DEFAULT_CHAT_DB_PATH, DEFAULT_LLM_MODEL


def _deterministic_embed(texts: list[str], dim: int = 16) -> list[list[float]]:
    """Lightweight deterministic unit-norm hash embedding for instant offline inspection."""
    vectors: list[list[float]] = []
    for text in texts:
        vec = [0.0] * dim
        tokens = [t.lower() for t in text.split() if t.strip()] or ["empty"]
        for tok in tokens:
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            for idx in range(dim):
                val = (digest[idx] / 127.5) - 1.0
                vec[idx] += val
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vectors.append([round(v / norm, 6) for v in vec])
    return vectors


def _offline_assistant_reply(messages: list[dict[str, str]], system_prompt: str) -> str:
    """Clean deterministic fallback responder when local Ollama server is unavailable."""
    user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
    latest = user_msgs[-1].strip() if user_msgs else ""
    lines = [ln.strip() for ln in latest.splitlines() if ln.strip()]
    word_count = len(latest.split())
    preview = (lines[0][:140] + ("..." if len(lines[0]) > 140 else "")) if lines else "empty input"
    return (
        f"[Offline Local Responder — Persisted to SQLite]\n"
        f"Processed {word_count} word(s) across {len(lines)} line(s) (turn #{len(user_msgs)} in session).\n"
        f"Key focus: \"{preview}\"\n"
        f"Start a local Ollama server (`ollama serve`) and switch Backend to 'Ollama Live' for full generative LLM inference."
    )


def _build_engine(
    chat_db_path: str | None = None,
    llm_model: str = DEFAULT_LLM_MODEL,
    llm_backend: str = "auto",
    embed_backend: str = "preview",
) -> AIEngine:
    db_target = Path(chat_db_path) if chat_db_path else DEFAULT_CHAT_DB_PATH
    embed_fn = _deterministic_embed if embed_backend == "preview" else None

    if llm_backend == "offline":
        chat_fn = _offline_assistant_reply
    elif llm_backend == "ollama":
        chat_fn = None
    else:
        # Auto mode: probe local Ollama port 11434 quickly; fall back cleanly if offline
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.25)
            ollama_up = sock.connect_ex(("127.0.0.1", 11434)) == 0
        chat_fn = None if ollama_up else _offline_assistant_reply

    return AIEngine(
        llm_model=llm_model or DEFAULT_LLM_MODEL,
        chat_db_path=db_target,
        custom_embed_fn=embed_fn,
        custom_chat_fn=chat_fn,
    )


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Engine</title>
<style>
  :root { --bg:#f8fafc; --card:#ffffff; --border:#e2e8f0; --text:#0f172a; --muted:#64748b; --accent:#334155; --accent-hover:#1e293b; --ok-bg:#ecfdf5; --ok-txt:#065f46; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; background:var(--bg); color:var(--text); line-height:1.55; }
  .wrap { max-width:880px; margin:2.2rem auto; padding:0 1.25rem; }
  h1 { font-size:1.35rem; font-weight:600; margin:0 0 1rem; letter-spacing:-0.01em; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:1.25rem 1.4rem; margin-bottom:1.1rem; }
  .label { font-size:0.73rem; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; color:var(--muted); margin-bottom:0.35rem; }
  .mission { margin:0 0 0.75rem; color:#334155; font-size:0.94rem; }
  .contracts { font-size:0.83rem; color:var(--muted); padding-top:0.65rem; border-top:1px solid var(--border); }
  .contracts code, pre { font-family:ui-monospace,SFMono-Regular,Consolas,monospace; background:#f1f5f9; padding:0.12rem 0.38rem; border-radius:4px; font-size:0.82rem; }
  .tabs { display:flex; gap:0.5rem; margin-bottom:1rem; flex-wrap:wrap; }
  .tab { padding:0.5rem 1rem; border-radius:7px; border:1px solid var(--border); background:#fff; color:var(--muted); font-size:0.86rem; font-weight:500; cursor:pointer; }
  .tab.active { background:var(--accent); color:#fff; border-color:var(--accent); }
  .grid3 { display:grid; grid-template-columns:1fr 1fr 1fr; gap:0.85rem; margin-bottom:0.85rem; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:0.85rem; margin-bottom:0.85rem; }
  label.field { display:block; font-size:0.82rem; color:var(--muted); margin-bottom:0.28rem; }
  input[type="text"], select, textarea { width:100%; padding:0.52rem 0.68rem; border:1px solid var(--border); border-radius:7px; font-size:0.87rem; background:#fff; color:var(--text); font-family:inherit; }
  textarea { min-height:115px; resize:vertical; font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-size:0.82rem; }
  button { padding:0.55rem 1.05rem; border-radius:7px; border:1px solid transparent; font-size:0.87rem; font-weight:500; cursor:pointer; background:var(--accent); color:#fff; }
  button:hover { background:var(--accent-hover); }
  button.secondary { background:#fff; color:var(--text); border-color:var(--border); }
  button.secondary:hover { background:#f1f5f9; }
  .chat-box { border:1px solid var(--border); border-radius:8px; background:#fcfdff; height:290px; overflow-y:auto; padding:0.85rem; margin-bottom:0.8rem; }
  .msg { margin-bottom:0.7rem; padding:0.65rem 0.85rem; border-radius:8px; font-size:0.88rem; white-space:pre-wrap; }
  .msg.user { background:#eef2ff; color:#1e1b4b; margin-left:2rem; }
  .msg.assistant { background:#f1f5f9; color:#0f172a; margin-right:2rem; }
  .msg-role { font-size:0.72rem; font-weight:600; text-transform:uppercase; opacity:0.65; margin-bottom:0.2rem; }
  .stat-pill { font-size:0.78rem; color:var(--muted); background:#f1f5f9; padding:0.25rem 0.6rem; border-radius:999px; }
  pre { padding:0.85rem; overflow-x:auto; margin:0.4rem 0 0; border:1px solid var(--border); border-radius:7px; white-space:pre-wrap; }
  .hidden { display:none; }
  @media (max-width:700px) { .grid3, .grid2 { grid-template-columns:1fr; } }
</style>
</head>
<body>
<div class="wrap">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem;margin-bottom:0.8rem;">
    <h1 style="margin:0;">AI Engine</h1>
    <span class="stat-pill" id="archiveStats">Archive: loading...</span>
  </div>

  <div class="card">
    <div class="label">Engine Mission</div>
    <p class="mission">Standalone AI engine providing numerical vector embeddings ready for database storage, structured LLM text &amp; chunk analysis, and a persistent multi-session chatbot backed by automatic SQLite conversation archiving.</p>
    <div class="contracts"><b>Input:</b> Text Chunks, Search Queries, or Chat Prompts &nbsp;&middot;&nbsp; <b>Output:</b> <code>EmbeddingRecord[]</code>, <code>AnalysisResult</code>, or persisted <code>ChatResponse</code></div>
  </div>

  <div class="card" style="padding:0.9rem 1.2rem;">
    <div class="grid3" style="margin-bottom:0;">
      <div>
        <label class="field">LLM Backend</label>
        <select id="llmBackend">
          <option value="auto">Auto (Ollama with Offline Fallback)</option>
          <option value="ollama">Ollama Live Server</option>
          <option value="offline">Instant Offline Responder</option>
        </select>
      </div>
      <div>
        <label class="field">LLM Model Tag</label>
        <input type="text" id="llmModel" value="__DEFAULT_MODEL__">
      </div>
      <div>
        <label class="field">Conversation Archive DB</label>
        <input type="text" id="chatDbPath" value="__DEFAULT_DB__" onchange="loadSessions()">
      </div>
    </div>
  </div>

  <div class="tabs">
    <button class="tab active" onclick="switchTab('chat', this)">1. Persistent Chatbot</button>
    <button class="tab" onclick="switchTab('analyze', this)">2. Text &amp; Chunk Analysis</button>
    <button class="tab" onclick="switchTab('embed', this)">3. Vector Embeddings</button>
  </div>

  <!-- TAB 1: PERSISTENT CHATBOT -->
  <div id="tab-chat" class="card">
    <div class="grid3">
      <div>
        <label class="field">Saved Session</label>
        <select id="sessionSelect" onchange="selectSession()"></select>
      </div>
      <div>
        <label class="field">Assistant Mode</label>
        <select id="chatMode">
          <option value="general">General Assistant</option>
          <option value="coding">Coding Specialist</option>
        </select>
      </div>
      <div style="display:flex;align-items:flex-end;gap:0.5rem;">
        <button class="secondary" onclick="newSession()">+ New Chat</button>
        <button class="secondary" onclick="deleteSession()">Delete</button>
      </div>
    </div>

    <div class="chat-box" id="chatTranscript"></div>
    <div style="display:flex;gap:0.6rem;">
      <input type="text" id="chatPrompt" placeholder="Write a message (automatically persisted to SQLite)..." onkeydown="if(event.key==='Enter')sendChat()">
      <button id="sendBtn" onclick="sendChat()">Send</button>
    </div>
  </div>

  <!-- TAB 2: ANALYSIS -->
  <div id="tab-analyze" class="card hidden">
    <div class="grid2">
      <div>
        <label class="field">Analysis Task</label>
        <select id="anaTask">
          <option value="summarize">Summarize</option>
          <option value="key_points">Extract Key Points</option>
          <option value="critique">Technical Critique</option>
          <option value="qa">Question Answering (QA)</option>
        </select>
      </div>
      <div>
        <label class="field">Optional Query (for QA or targeted focus)</label>
        <input type="text" id="anaQuery" placeholder="e.g. How are conversations persisted?">
      </div>
    </div>
    <label class="field">Input Text or Chunks JSON</label>
    <textarea id="anaInput">Modular engines communicate strictly through clean data contracts. StorageEngine records chronological lineage for every processed file, while AIEngine archives every chat turn automatically.</textarea>
    <div style="margin-top:0.8rem;">
      <button onclick="runAnalyze()">Run Analysis</button>
    </div>
    <div id="anaOutBox" class="hidden" style="margin-top:1rem;">
      <div class="label">Analysis Output</div>
      <pre id="anaOut"></pre>
    </div>
  </div>

  <!-- TAB 3: EMBEDDINGS -->
  <div id="tab-embed" class="card hidden">
    <div class="grid2">
      <div>
        <label class="field">Embedding Backend</label>
        <select id="embBackend">
          <option value="preview">Instant Deterministic Vectors (16-dim Preview)</option>
          <option value="st">SentenceTransformers Neural Model</option>
        </select>
      </div>
      <div>
        <label class="field">Input Type</label>
        <select id="embType">
          <option value="lines">Multiple Chunk Lines (one chunk per line)</option>
          <option value="query">Single Search Query</option>
        </select>
      </div>
    </div>
    <label class="field">Text Lines or Query</label>
    <textarea id="embInput">Security engine validates file signatures and hashes.
Chunking engine preserves chapter hierarchy and atomic quotes.
Storage engine tracks every engine execution in chronological order.</textarea>
    <div style="margin-top:0.8rem;display:flex;gap:0.6rem;">
      <button onclick="runEmbed()">Generate Embeddings</button>
      <button class="secondary hidden" id="dlEmbBtn" onclick="downloadEmbeddings()">Download JSON</button>
    </div>
    <div id="embOutBox" class="hidden" style="margin-top:1rem;">
      <div class="label">Generated Vector Records</div>
      <pre id="embOut"></pre>
    </div>
  </div>
</div>

<script>
let currentSessionId = null;
let lastEmbeddings = null;

function switchTab(name, btn) {
  ['chat','analyze','embed'].forEach(t => document.getElementById('tab-' + t).classList.add('hidden'));
  document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.remove('hidden');
  btn.classList.add('active');
}

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function baseCfg() {
  return {
    chat_db_path: document.getElementById('chatDbPath').value,
    llm_model: document.getElementById('llmModel').value,
    llm_backend: document.getElementById('llmBackend').value
  };
}

async function loadSessions() {
  const res = await fetch('/api/sessions', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(baseCfg())
  });
  const data = await res.json();
  if (!data.ok) return;
  document.getElementById('archiveStats').textContent =
    `Archive: ${data.stats.total_sessions} session(s) · ${data.stats.total_messages} message(s)`;

  const sel = document.getElementById('sessionSelect');
  sel.innerHTML = '<option value="">(New Session on Next Message)</option>' +
    data.sessions.map(s => `<option value="${s.id}">#${s.id} — ${esc(s.title)} (${s.mode})</option>`).join('');

  if (currentSessionId && data.sessions.some(s => s.id === currentSessionId)) {
    sel.value = String(currentSessionId);
    await selectSession();
  } else {
    currentSessionId = null;
    renderMessages([]);
  }
}

function renderMessages(msgs) {
  const box = document.getElementById('chatTranscript');
  if (!msgs || !msgs.length) {
    box.innerHTML = '<div style="color:#64748b;font-size:0.86rem;padding:0.5rem;">No messages in active view yet. Send a message below to start a persisted session.</div>';
    return;
  }
  box.innerHTML = msgs.map(m => `
    <div class="msg ${m.role === 'user' ? 'user' : 'assistant'}">
      <div class="msg-role">${esc(m.role)}</div>${esc(m.content)}
    </div>
  `).join('');
  box.scrollTop = box.scrollHeight;
}

async function selectSession() {
  const val = document.getElementById('sessionSelect').value;
  if (!val) { currentSessionId = null; renderMessages([]); return; }
  currentSessionId = parseInt(val, 10);
  const res = await fetch('/api/messages', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({...baseCfg(), session_id: currentSessionId})
  });
  const data = await res.json();
  if (data.ok) renderMessages(data.messages);
}

function newSession() {
  currentSessionId = null;
  document.getElementById('sessionSelect').value = '';
  renderMessages([]);
}

async function deleteSession() {
  if (!currentSessionId) return;
  await fetch('/api/delete_session', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({...baseCfg(), session_id: currentSessionId})
  });
  currentSessionId = null;
  await loadSessions();
}

async function sendChat() {
  const input = document.getElementById('chatPrompt');
  const prompt = input.value.trim();
  if (!prompt) return;
  input.value = '';
  const btn = document.getElementById('sendBtn');
  btn.disabled = true; btn.textContent = '...';
  try {
    const res = await fetch('/api/chat', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        ...baseCfg(),
        session_id: currentSessionId,
        mode: document.getElementById('chatMode').value,
        prompt
      })
    });
    const data = await res.json();
    if (data.ok) {
      currentSessionId = data.response.session_id;
      await loadSessions();
    } else {
      alert('Chat error: ' + data.error);
    }
  } finally {
    btn.disabled = false; btn.textContent = 'Send';
  }
}

async function runAnalyze() {
  const raw = document.getElementById('anaInput').value.trim();
  let inputs = raw;
  if (raw.startsWith('[')) { try { inputs = JSON.parse(raw); } catch (_) {} }
  const res = await fetch('/api/analyze', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      ...baseCfg(),
      task: document.getElementById('anaTask').value,
      query: document.getElementById('anaQuery').value || null,
      inputs
    })
  });
  const data = await res.json();
  document.getElementById('anaOutBox').classList.remove('hidden');
  document.getElementById('anaOut').textContent = JSON.stringify(data.ok ? data.result : data, null, 2);
}

async function runEmbed() {
  const raw = document.getElementById('embInput').value.trim();
  const embType = document.getElementById('embType').value;
  const res = await fetch('/api/embed', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      ...baseCfg(),
      embed_backend: document.getElementById('embBackend').value,
      emb_type: embType,
      raw
    })
  });
  const data = await res.json();
  document.getElementById('embOutBox').classList.remove('hidden');
  if (data.ok) {
    lastEmbeddings = data.records;
    document.getElementById('dlEmbBtn').classList.remove('hidden');
    document.getElementById('embOut').textContent = JSON.stringify(data.preview, null, 2);
  } else {
    document.getElementById('embOut').textContent = JSON.stringify(data, null, 2);
  }
}

function downloadEmbeddings() {
  if (!lastEmbeddings) return;
  const blob = new Blob([JSON.stringify(lastEmbeddings, null, 2)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'embeddings.json';
  a.click();
}

window.addEventListener('DOMContentLoaded', loadSessions);
</script>
</body>
</html>
""".replace("__DEFAULT_MODEL__", DEFAULT_LLM_MODEL).replace(
    "__DEFAULT_DB__", str(DEFAULT_CHAT_DB_PATH).replace("\\", "/")
)


class UIHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            raw = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            ai = _build_engine(
                chat_db_path=body.get("chat_db_path"),
                llm_model=body.get("llm_model", DEFAULT_LLM_MODEL),
                llm_backend=body.get("llm_backend", "auto"),
                embed_backend=body.get("embed_backend", "preview"),
            )

            if self.path == "/api/sessions":
                self._send_json(
                    {
                        "ok": True,
                        "sessions": ai.get_chat_sessions(),
                        "stats": ai.get_chat_stats(),
                    }
                )
            elif self.path == "/api/messages":
                sid = int(body["session_id"])
                self._send_json({"ok": True, "messages": ai.get_chat_messages(sid)})
            elif self.path == "/api/delete_session":
                sid = int(body["session_id"])
                self._send_json({"ok": True, "deleted": ai.delete_chat_session(sid)})
            elif self.path == "/api/chat":
                resp = ai.chat(
                    prompt=str(body.get("prompt", "")),
                    session_id=body.get("session_id"),
                    mode=str(body.get("mode", "general")),
                )
                self._send_json(
                    {
                        "ok": True,
                        "response": resp.to_dict(),
                        "messages": ai.get_chat_messages(resp.session_id),
                    }
                )
            elif self.path == "/api/analyze":
                res = ai.analyze(
                    inputs=body.get("inputs", ""),
                    task=str(body.get("task", "summarize")),
                    query=body.get("query") or None,
                )
                self._send_json({"ok": True, "result": res.to_dict()})
            elif self.path == "/api/embed":
                raw = str(body.get("raw", "")).strip()
                try:
                    if body.get("emb_type") == "query":
                        rec = ai.embed_query(raw)
                        records = [rec.to_dict()]
                    else:
                        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
                        chunks = [
                            {"chunk_id": f"chk_{i:04d}", "chunk_index": i, "content": ln}
                            for i, ln in enumerate(lines, start=1)
                        ]
                        records = [r.to_dict() for r in ai.embed_chunks(chunks)]
                    preview = [
                        {
                            "record_id": r["record_id"],
                            "dimension": r["dimension"],
                            "model_name": r["model_name"],
                            "vector_first_6": r["vector"][:6],
                            "text": r["text"][:90],
                        }
                        for r in records
                    ]
                    self._send_json({"ok": True, "records": records, "preview": preview})
                finally:
                    ai.unload_embedder()
            else:
                self.send_error(404)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=200)


def _pick_port(preferred: int = 8705) -> int:
    if preferred > 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", preferred)) != 0:
                return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def serve_ui(port: int = 0, open_browser: bool = True) -> None:
    chosen = _pick_port(port or 8705)
    server = ThreadingHTTPServer(("127.0.0.1", chosen), UIHandler)
    url = f"http://127.0.0.1:{chosen}"
    print(f"AI Engine UI running at {url} (Press Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping AI Engine UI.")
    finally:
        server.server_close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="AI Engine Standalone UI")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    serve_ui(port=args.port, open_browser=not args.no_browser)

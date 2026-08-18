"""
SWE-Vision Web App — ChatGPT-style interface for the VLM Tool Call Agent.

Upload images, enter prompts, and watch the agent reason step-by-step
with real-time streaming of tool calls, code execution, and results.

Usage:
    python apps/web_app.py [--port 8080] [--host 0.0.0.0]

Then open http://localhost:8080 in your browser.
"""

import argparse
import asyncio
import datetime
import json
import os
import queue
import sys
import threading
import uuid
from pathlib import Path

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    render_template_string,
    request,
    send_file,
)
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so ``import swe_vision`` works
# when running this script directly (e.g. ``python apps/web_app.py``).
# ---------------------------------------------------------------------------
_SWEVISSION_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _SWEVISSION_ROOT.parent
if str(_SWEVISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_SWEVISSION_ROOT))


def _project_runtime_dir(env_name: str, default: Path) -> str:
    """仅接受项目根内的运行目录，避免遗留环境变量指向旧项目。"""
    configured = os.environ.get(env_name)
    if configured:
        candidate = Path(configured).expanduser().resolve()
        try:
            candidate.relative_to(_PROJECT_ROOT)
        except ValueError:
            pass
        else:
            return str(candidate)
    return str(default)

try:
    from swe_vision.agent import VLMToolCallAgent
    from swe_vision.trajectory import TrajectoryRecorder
    AGENT_AVAILABLE = True
    AGENT_IMPORT_ERROR = ""
except ImportError as e:
    AGENT_AVAILABLE = False
    AGENT_IMPORT_ERROR = str(e)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SESSION_BASE = os.path.join(
    _project_runtime_dir(
        "VLM_WEB_SESSION_DIR",
        _PROJECT_ROOT / "runtime" / "swe-vision" / "web_sessions",
    ),
    "vlm_web_sessions",
)
os.makedirs(SESSION_BASE, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

# In-memory session store  {session_id: {queue, thread, ...}}
sessions: dict = {}


# ═══════════════════════════════════════════════════════════════════
# Streaming Trajectory Recorder
# ═══════════════════════════════════════════════════════════════════

class StreamingTrajectoryRecorder(TrajectoryRecorder if AGENT_AVAILABLE else object):
    """Extends TrajectoryRecorder to push SSE events to a queue."""

    def __init__(self, save_dir, event_queue, session_id):
        super().__init__(save_dir)
        self._eq = event_queue
        self._sid = session_id

    def _emit(self, event):
        self._eq.put(event)

    def _img_url(self, rel_path):
        return f"/api/files/{self._sid}/trajectory/{rel_path}"

    # -- overrides -------------------------------------------------------

    def record_user_step(self, query, image_paths=None):
        super().record_user_step(query, image_paths)
        step = self.steps[-1]
        images = [self._img_url(p) for p in step.get("images", [])]
        self._emit({"type": "user_message", "data": {"text": query, "images": images}})

    def record_assistant_step(self, content_text, tool_calls=None, reasoning_details=None):
        super().record_assistant_step(content_text, tool_calls, reasoning_details)
        step = self.steps[-1]

        if reasoning_details:
            if isinstance(reasoning_details, str):
                rd = reasoning_details
            else:
                rd = json.dumps(reasoning_details, ensure_ascii=False, indent=2)
            self._emit({"type": "thinking", "data": {"content": rd}})

        if content_text:
            self._emit({"type": "assistant_text", "data": {"content": content_text}})

        for tc in (step.get("tool_calls") or []):
            args_str = tc.get("arguments", "{}")
            try:
                parsed = json.loads(args_str) if isinstance(args_str, str) else args_str
            except Exception:
                parsed = {}
            self._emit({
                "type": "tool_call",
                "data": {
                    "name": tc.get("name", ""),
                    "id": tc.get("id", ""),
                    "code": parsed.get("code", ""),
                    "answer": parsed.get("answer", ""),
                    "arguments": args_str,
                },
            })

    def record_tool_step(self, tool_call_id, tool_name, code, text_output, base64_images=None):
        super().record_tool_step(tool_call_id, tool_name, code, text_output, base64_images)
        step = self.steps[-1]
        images = [self._img_url(p) for p in step.get("images", [])]
        self._emit({
            "type": "tool_result",
            "data": {
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "output": text_output,
                "images": images,
                "is_error": "[Error]" in (text_output or "") or "[Execution Error]" in (text_output or ""),
            },
        })

    def record_finish(self, answer):
        super().record_finish(answer)
        self._emit({"type": "finish", "data": {"answer": answer}})


# ═══════════════════════════════════════════════════════════════════
# Web VLM Agent (subclass)
# ═══════════════════════════════════════════════════════════════════

if AGENT_AVAILABLE:
    class WebVLMAgent(VLMToolCallAgent):
        """VLMToolCallAgent that streams steps to an SSE queue."""

        def __init__(self, event_queue, session_id, **kwargs):
            super().__init__(**kwargs)
            self._event_queue = event_queue
            self._session_id = session_id

        def _init_trajectory(self, query, image_paths):
            save_dir = os.path.join(SESSION_BASE, self._session_id, "trajectory")
            recorder = StreamingTrajectoryRecorder(save_dir, self._event_queue, self._session_id)
            recorder.set_metadata(
                model=self.model,
                start_time=TrajectoryRecorder._now_iso(),
                query=query,
                image_paths=image_paths or [],
                max_iterations=self.max_iterations,
                system_prompt=self.system_prompt,
            )
            return recorder


# ═══════════════════════════════════════════════════════════════════
# Agent Thread
# ═══════════════════════════════════════════════════════════════════

def run_agent_thread(event_queue, session_id, prompt, image_paths, config):
    """Run the VLM agent in a background thread with its own asyncio loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    agent = WebVLMAgent(
        event_queue=event_queue,
        session_id=session_id,
        model=config.get("model", "openai/gpt-5.2"),
        api_key=config.get("api_key") or None,
        base_url=config.get("base_url") or None,
        max_iterations=config.get("max_iterations", 30),
        verbose=True,
        reasoning=config.get("reasoning", True),
    )

    try:
        event_queue.put({"type": "status", "data": {"message": "Starting agent and Docker kernel..."}})
        answer = loop.run_until_complete(agent.run(prompt, image_paths if image_paths else None))
    except Exception as e:
        import traceback
        event_queue.put({"type": "error", "data": {"message": f"{e}\n{traceback.format_exc()}"}})
    finally:
        try:
            loop.run_until_complete(agent.cleanup())
        except Exception:
            pass
        loop.close()
        event_queue.put(None)  # sentinel


# ═══════════════════════════════════════════════════════════════════
# Flask Routes
# ═══════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template_string(
        HTML_TEMPLATE,
        agent_available=AGENT_AVAILABLE,
        agent_error=AGENT_IMPORT_ERROR,
    )


@app.route("/api/chat", methods=["POST"])
def api_chat():
    if not AGENT_AVAILABLE:
        return jsonify({"error": f"Agent not available: {AGENT_IMPORT_ERROR}"}), 500

    prompt = request.form.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    model = request.form.get("model", "openai/gpt-5.2")
    api_key = request.form.get("api_key", "")
    base_url = request.form.get("base_url", "")
    reasoning = request.form.get("reasoning", "true") == "true"
    max_iterations = int(request.form.get("max_iterations", "30"))

    session_id = uuid.uuid4().hex[:12]
    upload_dir = os.path.join(SESSION_BASE, session_id, "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    image_paths = []
    for f in request.files.getlist("images"):
        if f.filename:
            safe = secure_filename(f.filename)
            path = os.path.join(upload_dir, safe)
            f.save(path)
            image_paths.append(path)

    eq = queue.Queue()
    sessions[session_id] = {"queue": eq}

    t = threading.Thread(
        target=run_agent_thread,
        args=(eq, session_id, prompt, image_paths, {
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
            "reasoning": reasoning,
            "max_iterations": max_iterations,
        }),
        daemon=True,
    )
    t.start()
    sessions[session_id]["thread"] = t

    image_urls = [
        f"/api/files/{session_id}/uploads/{os.path.basename(p)}" for p in image_paths
    ]
    return jsonify({"session_id": session_id, "image_urls": image_urls})


@app.route("/api/stream/<session_id>")
def api_stream(session_id):
    if session_id not in sessions:
        abort(404)

    eq = sessions[session_id]["queue"]

    def generate():
        while True:
            try:
                event = eq.get(timeout=300)
                if event is None:
                    yield "data: {\"type\":\"done\"}\n\n"
                    break
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except queue.Empty:
                yield ": keepalive\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/files/<session_id>/<path:filepath>")
def serve_file(session_id, filepath):
    full = os.path.realpath(os.path.join(SESSION_BASE, session_id, filepath))
    base = os.path.realpath(os.path.join(SESSION_BASE, session_id))
    if not full.startswith(base):
        abort(403)
    if not os.path.isfile(full):
        abort(404)
    return send_file(full)


# ═══════════════════════════════════════════════════════════════════
# HTML Template
# ═══════════════════════════════════════════════════════════════════

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SWE-Vision Agent</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.0/marked.min.js"></script>
<style>
:root {
  --bg: #0a0a0a;
  --bg-chat: #0f0f0f;
  --surface: #1a1a1a;
  --surface-hover: #222;
  --border: #2a2a2a;
  --border-light: #333;
  --text: #e8e8e8;
  --text-secondary: #999;
  --text-muted: #666;
  --accent: #3b82f6;
  --accent-hover: #2563eb;
  --purple: #8b5cf6;
  --purple-dim: rgba(139,92,246,0.12);
  --green: #10b981;
  --green-dim: rgba(16,185,129,0.12);
  --red: #ef4444;
  --red-dim: rgba(239,68,68,0.12);
  --yellow: #f59e0b;
  --yellow-dim: rgba(245,158,11,0.12);
  --cyan: #06b6d4;
  --code-bg: #111;
  --radius: 12px;
  --radius-sm: 8px;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: var(--bg);
  color: var(--text);
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* Header */
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 24px;
  border-bottom: 1px solid var(--border);
  background: var(--bg);
  flex-shrink: 0;
  z-index: 10;
}
.header .logo {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 16px;
  font-weight: 700;
}
.header .logo-icon {
  width: 32px;
  height: 32px;
  background: linear-gradient(135deg, var(--accent), var(--purple));
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
}
.header-actions { display: flex; gap: 8px; }
.btn-icon {
  width: 36px; height: 36px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text-secondary);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  transition: all 0.15s;
}
.btn-icon:hover { background: var(--surface-hover); color: var(--text); border-color: var(--border-light); }

/* Chat Area */
.chat-container {
  flex: 1;
  overflow-y: auto;
  scroll-behavior: smooth;
  background: var(--bg-chat);
}
.chat-messages {
  max-width: 820px;
  margin: 0 auto;
  padding: 24px 20px 120px;
}

/* Welcome */
.welcome {
  text-align: center;
  padding: 80px 20px 40px;
  animation: fadeIn 0.5s ease;
}
.welcome h2 {
  font-size: 28px;
  font-weight: 700;
  margin-bottom: 8px;
  background: linear-gradient(135deg, var(--text), var(--text-secondary));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.welcome p { color: var(--text-secondary); font-size: 15px; max-width: 500px; margin: 0 auto; line-height: 1.6; }

/* Messages */
.message {
  margin-bottom: 24px;
  animation: slideUp 0.3s ease;
}
.message-user {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
}
.message-user .bubble {
  background: var(--accent);
  color: #fff;
  padding: 12px 16px;
  border-radius: var(--radius) var(--radius) 4px var(--radius);
  max-width: 75%;
  font-size: 14px;
  line-height: 1.6;
  word-break: break-word;
}
.message-user .msg-images {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 8px;
  justify-content: flex-end;
}
.message-user .msg-images img {
  max-height: 180px;
  max-width: 260px;
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: transform 0.15s;
}
.message-user .msg-images img:hover { transform: scale(1.03); }

.message-assistant {
  display: flex;
  gap: 12px;
  align-items: flex-start;
}
.avatar {
  width: 32px; height: 32px;
  border-radius: 50%;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  font-weight: 700;
  margin-top: 2px;
}
.avatar-assistant {
  background: linear-gradient(135deg, var(--purple), var(--accent));
  color: #fff;
}
.assistant-content {
  flex: 1;
  min-width: 0;
}

/* Steps inside assistant message */
.step-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  margin-bottom: 10px;
  overflow: hidden;
  animation: fadeIn 0.3s ease;
}

/* Thinking */
.thinking-card { border-color: rgba(139,92,246,0.25); }
.thinking-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  cursor: pointer;
  user-select: none;
  font-size: 13px;
  color: var(--purple);
  font-weight: 600;
  transition: background 0.15s;
}
.thinking-header:hover { background: var(--purple-dim); }
.thinking-header .arrow {
  transition: transform 0.2s;
  font-size: 11px;
}
.thinking-card.open .thinking-header .arrow { transform: rotate(90deg); }
.thinking-body {
  display: none;
  padding: 0 14px 12px;
  font-size: 13px;
  color: var(--text-secondary);
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 400px;
  overflow-y: auto;
}
.thinking-card.open .thinking-body { display: block; }

/* Assistant text */
.assistant-text {
  padding: 2px 0;
  font-size: 14px;
  line-height: 1.75;
  margin-bottom: 10px;
}
.assistant-text p { margin-bottom: 8px; }
.assistant-text code {
  background: var(--code-bg);
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 13px;
  font-family: 'SF Mono', Consolas, 'Liberation Mono', monospace;
}
.assistant-text pre {
  background: var(--code-bg);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 14px;
  overflow-x: auto;
  margin: 8px 0;
}
.assistant-text pre code { background: none; padding: 0; }

/* Tool call */
.toolcall-card { border-color: rgba(245,158,11,0.25); }
.toolcall-label {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  font-size: 13px;
  border-bottom: 1px solid var(--border);
}
.toolcall-label .fn-badge {
  background: var(--yellow-dim);
  color: var(--yellow);
  padding: 2px 10px;
  border-radius: 12px;
  font-weight: 600;
  font-size: 12px;
}
.toolcall-label .fn-icon { font-size: 14px; }
.toolcall-code {
  position: relative;
  padding: 14px;
  background: var(--code-bg);
  overflow-x: auto;
}
.toolcall-code pre {
  margin: 0;
  font-family: 'SF Mono', Consolas, 'Liberation Mono', monospace;
  font-size: 13px;
  line-height: 1.6;
  color: var(--text);
}
.copy-btn {
  position: absolute;
  top: 8px;
  right: 8px;
  background: var(--surface);
  border: 1px solid var(--border);
  color: var(--text-secondary);
  padding: 3px 8px;
  border-radius: 6px;
  font-size: 11px;
  cursor: pointer;
  opacity: 0;
  transition: opacity 0.15s;
}
.toolcall-code:hover .copy-btn { opacity: 1; }
.copy-btn:hover { background: var(--surface-hover); color: var(--text); }

/* Tool result */
.result-card { border-color: rgba(16,185,129,0.25); }
.result-card.error { border-color: rgba(239,68,68,0.25); }
.result-label {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  font-size: 13px;
  border-bottom: 1px solid var(--border);
}
.result-label .res-badge {
  padding: 2px 10px;
  border-radius: 12px;
  font-weight: 600;
  font-size: 12px;
}
.result-card .res-badge { background: var(--green-dim); color: var(--green); }
.result-card.error .res-badge { background: var(--red-dim); color: var(--red); }
.result-output {
  padding: 12px 14px;
  font-family: 'SF Mono', Consolas, 'Liberation Mono', monospace;
  font-size: 13px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--text-secondary);
  max-height: 400px;
  overflow-y: auto;
}
.result-card.error .result-output { color: var(--red); }
.result-images {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding: 10px 14px;
}
.result-images img {
  max-height: 300px;
  max-width: 100%;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
  cursor: pointer;
  transition: transform 0.15s;
}
.result-images img:hover { transform: scale(1.02); }

/* Finish */
.finish-card {
  background: var(--green-dim);
  border: 1px solid rgba(16,185,129,0.3);
  border-radius: var(--radius);
  padding: 16px;
  margin-bottom: 10px;
  animation: fadeIn 0.4s ease;
}
.finish-card .finish-label {
  color: var(--green);
  font-weight: 700;
  font-size: 13px;
  margin-bottom: 6px;
  display: flex;
  align-items: center;
  gap: 6px;
}
.finish-card .finish-text {
  font-size: 15px;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-word;
}

/* Loading */
.loading-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 0;
  font-size: 13px;
  color: var(--text-muted);
  animation: pulse 1.5s infinite;
}
.loading-dot {
  width: 6px; height: 6px;
  background: var(--text-muted);
  border-radius: 50%;
  animation: bounce 1.4s infinite ease-in-out;
}
.loading-dot:nth-child(1) { animation-delay: -0.32s; }
.loading-dot:nth-child(2) { animation-delay: -0.16s; }

/* Error */
.error-card {
  background: var(--red-dim);
  border: 1px solid rgba(239,68,68,0.3);
  border-radius: var(--radius);
  padding: 14px;
  font-size: 13px;
  color: var(--red);
  white-space: pre-wrap;
  word-break: break-word;
  margin-bottom: 10px;
}

/* Input Area */
.input-area {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  background: linear-gradient(transparent, var(--bg) 30%);
  padding: 20px 20px 24px;
  z-index: 10;
}
.input-wrapper {
  max-width: 820px;
  margin: 0 auto;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  overflow: hidden;
  transition: border-color 0.2s;
}
.input-wrapper:focus-within { border-color: var(--accent); }

.upload-preview {
  display: none;
  flex-wrap: wrap;
  gap: 8px;
  padding: 12px 14px 0;
}
.upload-preview.has-files { display: flex; }
.upload-thumb {
  position: relative;
  width: 64px; height: 64px;
  border-radius: var(--radius-sm);
  overflow: hidden;
  border: 1px solid var(--border);
}
.upload-thumb img { width: 100%; height: 100%; object-fit: cover; }
.upload-thumb .remove-btn {
  position: absolute;
  top: 2px; right: 2px;
  width: 18px; height: 18px;
  background: rgba(0,0,0,0.7);
  border: none;
  border-radius: 50%;
  color: #fff;
  font-size: 11px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  line-height: 1;
}

.input-row {
  display: flex;
  align-items: flex-end;
  gap: 8px;
  padding: 12px 14px;
}
.input-row textarea {
  flex: 1;
  background: none;
  border: none;
  color: var(--text);
  font-family: inherit;
  font-size: 14px;
  line-height: 1.5;
  resize: none;
  outline: none;
  max-height: 200px;
  min-height: 24px;
}
.input-row textarea::placeholder { color: var(--text-muted); }

.input-actions {
  display: flex;
  gap: 4px;
  flex-shrink: 0;
}
.btn-attach {
  width: 36px; height: 36px;
  border: none;
  background: none;
  color: var(--text-secondary);
  cursor: pointer;
  border-radius: var(--radius-sm);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  transition: all 0.15s;
}
.btn-attach:hover { background: var(--surface-hover); color: var(--text); }
.btn-send {
  width: 36px; height: 36px;
  border: none;
  background: var(--accent);
  color: #fff;
  cursor: pointer;
  border-radius: var(--radius-sm);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  transition: all 0.15s;
}
.btn-send:hover { background: var(--accent-hover); }
.btn-send:disabled { opacity: 0.4; cursor: not-allowed; }

/* Settings Panel */
.settings-overlay {
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.6);
  z-index: 100;
  justify-content: center;
  align-items: center;
}
.settings-overlay.open { display: flex; }
.settings-panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  width: 480px;
  max-width: 90vw;
  max-height: 85vh;
  overflow-y: auto;
  padding: 24px;
  animation: fadeIn 0.2s ease;
}
.settings-panel h3 {
  font-size: 18px;
  margin-bottom: 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.settings-panel .close-btn {
  background: none;
  border: none;
  color: var(--text-secondary);
  font-size: 20px;
  cursor: pointer;
}
.form-group { margin-bottom: 16px; }
.form-group label {
  display: block;
  font-size: 13px;
  font-weight: 600;
  color: var(--text-secondary);
  margin-bottom: 6px;
}
.form-group input, .form-group select {
  width: 100%;
  padding: 10px 12px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  color: var(--text);
  font-family: inherit;
  font-size: 14px;
  outline: none;
  transition: border-color 0.15s;
}
.form-group input:focus, .form-group select:focus { border-color: var(--accent); }
.form-group .hint { font-size: 12px; color: var(--text-muted); margin-top: 4px; }

.toggle-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 0;
}
.toggle-row label { margin-bottom: 0; }
.toggle {
  width: 44px; height: 24px;
  background: var(--border);
  border-radius: 12px;
  position: relative;
  cursor: pointer;
  transition: background 0.2s;
}
.toggle.on { background: var(--accent); }
.toggle::after {
  content: '';
  position: absolute;
  top: 2px; left: 2px;
  width: 20px; height: 20px;
  background: #fff;
  border-radius: 50%;
  transition: transform 0.2s;
}
.toggle.on::after { transform: translateX(20px); }

/* Lightbox */
.lightbox {
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.92);
  z-index: 200;
  justify-content: center;
  align-items: center;
  cursor: zoom-out;
}
.lightbox.open { display: flex; }
.lightbox img { max-width: 95vw; max-height: 95vh; border-radius: 8px; }

/* Status badge */
.status-indicator {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  background: var(--yellow-dim);
  border-radius: 20px;
  font-size: 12px;
  color: var(--yellow);
  font-weight: 600;
  margin-bottom: 10px;
  animation: pulse 2s infinite;
}

/* Animations */
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
@keyframes slideUp { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }
@keyframes bounce {
  0%, 80%, 100% { transform: scale(0); }
  40% { transform: scale(1); }
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--border-light); }

/* Responsive */
@media (max-width: 640px) {
  .chat-messages { padding: 16px 12px 120px; }
  .header { padding: 10px 14px; }
  .message-user .bubble { max-width: 90%; }
}
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <div class="logo">
    <div class="logo-icon">V</div>
    <span>SWE-Vision Agent</span>
  </div>
  <div class="header-actions">
    <button class="btn-icon" onclick="clearChat()" title="New Chat">+</button>
    <button class="btn-icon" onclick="openSettings()" title="Settings">&#9881;</button>
  </div>
</div>

<!-- Chat -->
<div class="chat-container" id="chatContainer">
  <div class="chat-messages" id="chatMessages">
    <div class="welcome" id="welcome">
      <h2>SWE-Vision Agent</h2>
      <p>Upload an image and ask a question. The agent will use a Jupyter notebook to analyze it step-by-step, showing you its reasoning and code execution in real time.</p>
    </div>
  </div>
</div>

<!-- Input -->
<div class="input-area">
  <div class="input-wrapper" id="inputWrapper">
    <div class="upload-preview" id="uploadPreview"></div>
    <div class="input-row">
      <div class="input-actions">
        <button class="btn-attach" onclick="document.getElementById('fileInput').click()" title="Upload Image">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>
        </button>
        <input type="file" id="fileInput" accept="image/*" multiple hidden>
      </div>
      <textarea id="promptInput" rows="1" placeholder="Ask anything about an image..." onkeydown="handleKeyDown(event)" oninput="autoResize(this)"></textarea>
      <button class="btn-send" id="sendBtn" onclick="sendMessage()" title="Send">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
      </button>
    </div>
  </div>
</div>

<!-- Settings -->
<div class="settings-overlay" id="settingsOverlay" onclick="if(event.target===this)closeSettings()">
  <div class="settings-panel">
    <h3>Settings <button class="close-btn" onclick="closeSettings()">&times;</button></h3>
    <div class="form-group">
      <label>API Key</label>
      <input type="password" id="settApiKey" placeholder="sk-...">
      <div class="hint">OpenAI-compatible API key. Stored in browser only.</div>
    </div>
    <div class="form-group">
      <label>Base URL</label>
      <input type="text" id="settBaseUrl" placeholder="https://api.openai.com/v1">
      <div class="hint">Leave empty for default. Supports OpenRouter, local endpoints, etc.</div>
    </div>
    <div class="form-group">
      <label>Model</label>
      <input type="text" id="settModel" placeholder="openai/gpt-5.2">
    </div>
    <div class="form-group">
      <label>Max Iterations</label>
      <input type="number" id="settMaxIter" value="30" min="1" max="100">
    </div>
    <div class="form-group">
      <div class="toggle-row">
        <label>Enable Reasoning</label>
        <div class="toggle on" id="settReasoning" onclick="this.classList.toggle('on')"></div>
      </div>
    </div>
  </div>
</div>

<!-- Lightbox -->
<div class="lightbox" id="lightbox" onclick="closeLightbox()">
  <img id="lightboxImg" src="">
</div>

<!-- Drag overlay -->
<div id="dragOverlay" style="display:none;position:fixed;inset:0;background:rgba(59,130,246,0.1);border:3px dashed var(--accent);z-index:50;pointer-events:none;justify-content:center;align-items:center;">
  <div style="color:var(--accent);font-size:20px;font-weight:600;">Drop images here</div>
</div>

<script>
// --- State ---
let uploadedFiles = [];
let isProcessing = false;
let currentEventSource = null;

// --- Settings ---
function loadSettings() {
  return {
    apiKey: localStorage.getItem('vlm_api_key') || '',
    baseUrl: localStorage.getItem('vlm_base_url') || '',
    model: localStorage.getItem('vlm_model') || 'openai/gpt-5.2',
    maxIter: parseInt(localStorage.getItem('vlm_max_iter') || '30'),
    reasoning: localStorage.getItem('vlm_reasoning') !== 'false',
  };
}
function saveSettings() {
  localStorage.setItem('vlm_api_key', document.getElementById('settApiKey').value);
  localStorage.setItem('vlm_base_url', document.getElementById('settBaseUrl').value);
  localStorage.setItem('vlm_model', document.getElementById('settModel').value);
  localStorage.setItem('vlm_max_iter', document.getElementById('settMaxIter').value);
  localStorage.setItem('vlm_reasoning', document.getElementById('settReasoning').classList.contains('on'));
}
function openSettings() {
  const s = loadSettings();
  document.getElementById('settApiKey').value = s.apiKey;
  document.getElementById('settBaseUrl').value = s.baseUrl;
  document.getElementById('settModel').value = s.model;
  document.getElementById('settMaxIter').value = s.maxIter;
  const tog = document.getElementById('settReasoning');
  tog.classList.toggle('on', s.reasoning);
  document.getElementById('settingsOverlay').classList.add('open');
}
function closeSettings() {
  saveSettings();
  document.getElementById('settingsOverlay').classList.remove('open');
}

// --- UI Helpers ---
function scrollToBottom() {
  const c = document.getElementById('chatContainer');
  requestAnimationFrame(() => { c.scrollTop = c.scrollHeight; });
}

function openLightbox(src) {
  document.getElementById('lightboxImg').src = src;
  document.getElementById('lightbox').classList.add('open');
}
function closeLightbox() {
  document.getElementById('lightbox').classList.remove('open');
}
document.addEventListener('keydown', e => { if (e.key === 'Escape') { closeLightbox(); closeSettings(); } });

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 200) + 'px';
}

function handleKeyDown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function clearChat() {
  if (currentEventSource) { currentEventSource.close(); currentEventSource = null; }
  document.getElementById('chatMessages').innerHTML = `
    <div class="welcome" id="welcome">
      <h2>SWE-Vision Agent</h2>
      <p>Upload an image and ask a question. The agent will use a Jupyter notebook to analyze it step-by-step.</p>
    </div>`;
  isProcessing = false;
  document.getElementById('sendBtn').disabled = false;
}

// --- File Upload ---
const fileInput = document.getElementById('fileInput');
fileInput.addEventListener('change', () => {
  for (const f of fileInput.files) addFile(f);
  fileInput.value = '';
});

function addFile(file) {
  if (!file.type.startsWith('image/')) return;
  uploadedFiles.push(file);
  renderPreviews();
}
function removeFile(idx) {
  uploadedFiles.splice(idx, 1);
  renderPreviews();
}
function renderPreviews() {
  const el = document.getElementById('uploadPreview');
  el.innerHTML = '';
  el.classList.toggle('has-files', uploadedFiles.length > 0);
  uploadedFiles.forEach((f, i) => {
    const thumb = document.createElement('div');
    thumb.className = 'upload-thumb';
    const img = document.createElement('img');
    img.src = URL.createObjectURL(f);
    const btn = document.createElement('button');
    btn.className = 'remove-btn';
    btn.textContent = '\u00d7';
    btn.onclick = () => removeFile(i);
    thumb.append(img, btn);
    el.append(thumb);
  });
}

// Drag & Drop
document.addEventListener('dragover', e => { e.preventDefault(); document.getElementById('dragOverlay').style.display = 'flex'; });
document.addEventListener('dragleave', e => { if (e.relatedTarget === null) document.getElementById('dragOverlay').style.display = 'none'; });
document.addEventListener('drop', e => {
  e.preventDefault();
  document.getElementById('dragOverlay').style.display = 'none';
  for (const f of e.dataTransfer.files) addFile(f);
});

// --- Markdown ---
marked.setOptions({
  breaks: true,
  gfm: true,
  highlight: function(code, lang) {
    if (lang && hljs.getLanguage(lang)) return hljs.highlight(code, {language: lang}).value;
    return hljs.highlightAuto(code).value;
  }
});

function renderMarkdown(text) {
  try { return marked.parse(text); } catch { return escapeHtml(text); }
}
function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// --- Message Rendering ---
function addUserMessage(text, files) {
  const wel = document.getElementById('welcome');
  if (wel) wel.remove();

  const msg = document.createElement('div');
  msg.className = 'message message-user';
  let html = `<div class="bubble">${escapeHtml(text)}</div>`;
  if (files && files.length > 0) {
    html += '<div class="msg-images">';
    for (const f of files) {
      html += `<img src="${URL.createObjectURL(f)}" onclick="openLightbox(this.src)">`;
    }
    html += '</div>';
  }
  msg.innerHTML = html;
  document.getElementById('chatMessages').append(msg);
  scrollToBottom();
}

function createAssistantMessage() {
  const msg = document.createElement('div');
  msg.className = 'message message-assistant';
  msg.innerHTML = `
    <div class="avatar avatar-assistant">V</div>
    <div class="assistant-content" id="assistantContent_${Date.now()}"></div>
  `;
  document.getElementById('chatMessages').append(msg);
  scrollToBottom();
  return msg.querySelector('.assistant-content');
}

function addLoadingIndicator(container) {
  const el = document.createElement('div');
  el.className = 'loading-indicator';
  el.id = 'loadingIndicator';
  el.innerHTML = '<div class="loading-dot"></div><div class="loading-dot"></div><div class="loading-dot"></div><span>Agent is thinking...</span>';
  container.append(el);
  scrollToBottom();
}
function removeLoadingIndicator() {
  const el = document.getElementById('loadingIndicator');
  if (el) el.remove();
}
function updateLoadingText(text) {
  const el = document.getElementById('loadingIndicator');
  if (el) el.querySelector('span').textContent = text;
}

// --- Event Handlers ---
function handleEvent(event, container) {
  switch (event.type) {
    case 'status':
      updateLoadingText(event.data.message);
      break;

    case 'thinking': {
      removeLoadingIndicator();
      const card = document.createElement('div');
      card.className = 'step-card thinking-card';
      const content = event.data.content || '';
      const preview = content.substring(0, 80) + (content.length > 80 ? '...' : '');
      card.innerHTML = `
        <div class="thinking-header" onclick="this.parentElement.classList.toggle('open')">
          <span class="arrow">\u25b8</span>
          <span>Thinking</span>
          <span style="color:var(--text-muted);font-weight:400;font-size:12px;margin-left:auto;max-width:400px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escapeHtml(preview)}</span>
        </div>
        <div class="thinking-body">${escapeHtml(content)}</div>
      `;
      container.append(card);
      addLoadingIndicator(container);
      scrollToBottom();
      break;
    }

    case 'assistant_text': {
      removeLoadingIndicator();
      const div = document.createElement('div');
      div.className = 'assistant-text';
      div.innerHTML = renderMarkdown(event.data.content);
      container.append(div);
      container.querySelectorAll('pre code').forEach(el => hljs.highlightElement(el));
      addLoadingIndicator(container);
      scrollToBottom();
      break;
    }

    case 'tool_call': {
      removeLoadingIndicator();
      if (event.data.name === 'finish') {
        break;
      }
      const card = document.createElement('div');
      card.className = 'step-card toolcall-card';
      const code = event.data.code || event.data.arguments || '';
      const highlighted = code ? hljs.highlight(code, {language: 'python'}).value : '';
      card.innerHTML = `
        <div class="toolcall-label">
          <span class="fn-icon">\u26a1</span>
          <span class="fn-badge">${escapeHtml(event.data.name)}</span>
        </div>
        <div class="toolcall-code">
          <button class="copy-btn" onclick="copyCode(this)">Copy</button>
          <pre><code>${highlighted}</code></pre>
        </div>
      `;
      container.append(card);
      updateLoadingText('Running code...');
      addLoadingIndicator(container);
      scrollToBottom();
      break;
    }

    case 'tool_result': {
      removeLoadingIndicator();
      const card = document.createElement('div');
      card.className = 'step-card result-card' + (event.data.is_error ? ' error' : '');
      const label = event.data.is_error ? 'Error' : 'Output';
      let html = `
        <div class="result-label">
          <span class="res-badge">${label}</span>
        </div>
      `;
      if (event.data.output) {
        html += `<div class="result-output">${escapeHtml(event.data.output)}</div>`;
      }
      if (event.data.images && event.data.images.length > 0) {
        html += '<div class="result-images">';
        for (const src of event.data.images) {
          html += `<img src="${src}" onclick="openLightbox(this.src)" loading="lazy">`;
        }
        html += '</div>';
      }
      card.innerHTML = html;
      container.append(card);
      addLoadingIndicator(container);
      scrollToBottom();
      break;
    }

    case 'finish': {
      removeLoadingIndicator();
      const card = document.createElement('div');
      card.className = 'finish-card';
      card.innerHTML = `
        <div class="finish-label">\u2713 Final Answer</div>
        <div class="finish-text">${renderMarkdown(event.data.answer)}</div>
      `;
      container.append(card);
      container.querySelectorAll('.finish-text pre code').forEach(el => hljs.highlightElement(el));
      scrollToBottom();
      break;
    }

    case 'error': {
      removeLoadingIndicator();
      const card = document.createElement('div');
      card.className = 'error-card';
      card.textContent = event.data.message;
      container.append(card);
      scrollToBottom();
      break;
    }

    case 'done':
      removeLoadingIndicator();
      break;
  }
}

function copyCode(btn) {
  const code = btn.parentElement.querySelector('code').textContent;
  navigator.clipboard.writeText(code).then(() => {
    btn.textContent = 'Copied!';
    setTimeout(() => btn.textContent = 'Copy', 1500);
  });
}

// --- Send Message ---
async function sendMessage() {
  const input = document.getElementById('promptInput');
  const prompt = input.value.trim();
  if (!prompt || isProcessing) return;

  isProcessing = true;
  document.getElementById('sendBtn').disabled = true;

  const settings = loadSettings();
  const filesToSend = [...uploadedFiles];

  addUserMessage(prompt, filesToSend);

  input.value = '';
  input.style.height = 'auto';
  uploadedFiles = [];
  renderPreviews();

  const container = createAssistantMessage();
  addLoadingIndicator(container);

  const formData = new FormData();
  formData.append('prompt', prompt);
  formData.append('model', settings.model);
  formData.append('api_key', settings.apiKey);
  formData.append('base_url', settings.baseUrl);
  formData.append('reasoning', settings.reasoning);
  formData.append('max_iterations', settings.maxIter);
  for (const f of filesToSend) formData.append('images', f);

  try {
    const resp = await fetch('/api/chat', { method: 'POST', body: formData });
    if (!resp.ok) {
      const err = await resp.json();
      handleEvent({type: 'error', data: {message: err.error || 'Request failed'}}, container);
      isProcessing = false;
      document.getElementById('sendBtn').disabled = false;
      return;
    }
    const { session_id } = await resp.json();

    const es = new EventSource(`/api/stream/${session_id}`);
    currentEventSource = es;

    es.onmessage = (e) => {
      const event = JSON.parse(e.data);
      handleEvent(event, container);
      if (event.type === 'finish' || event.type === 'error' || event.type === 'done') {
        es.close();
        currentEventSource = null;
        isProcessing = false;
        document.getElementById('sendBtn').disabled = false;
      }
    };
    es.onerror = () => {
      es.close();
      currentEventSource = null;
      if (isProcessing) {
        handleEvent({type: 'error', data: {message: 'Connection lost. The agent may still be running in the background.'}}, container);
        isProcessing = false;
        document.getElementById('sendBtn').disabled = false;
      }
    };
  } catch (e) {
    handleEvent({type: 'error', data: {message: 'Network error: ' + e.message}}, container);
    isProcessing = false;
    document.getElementById('sendBtn').disabled = false;
  }
}

// --- Init ---
document.getElementById('promptInput').focus();
</script>
</body>
</html>
"""


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SWE-Vision Web App")
    parser.add_argument("--port", "-p", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"\n  SWE-Vision Web App")
    print(f"  Agent available: {AGENT_AVAILABLE}")
    if not AGENT_AVAILABLE:
        print(f"  Agent import error: {AGENT_IMPORT_ERROR}")
    print(f"  Sessions dir: {SESSION_BASE}")
    print(f"  Open: http://localhost:{args.port}\n")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

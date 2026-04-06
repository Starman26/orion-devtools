"""
test_server.py — ORION DevTools: Local testing interface.

Imports the agent graph from ../Orion/ and exposes a WebSocket bridge endpoint
(/ws/robot) for external bridge processes to connect to.
No auth, no Supabase, no Cloud Run — just the graph + MemorySaver.

Run:
    cd C:\\Products\\FINAL_PRODUCTS\\orion-devtools
    ..\\Orion\\.venv\\Scripts\\Activate.ps1
    pip install -r requirements.txt
    python test_server.py

Opens at http://localhost:8000
"""

import os
import sys
import io
import json
import asyncio
import logging
import uuid
from datetime import datetime
from typing import AsyncGenerator, Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# Path setup: add ../Orion so we can import the agent graph
# ---------------------------------------------------------------------------
ORION_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Orion'))
sys.path.insert(0, ORION_ROOT)

# Load .env from the Orion project (has ANTHROPIC_API_KEY, etc.)
from dotenv import load_dotenv
load_dotenv(os.path.join(ORION_ROOT, '.env'))

# Fix Windows console encoding (cp1252 can't handle Unicode from LLM output)
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from langchain_core.messages import HumanMessage, AIMessage

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("orion_devtools")

# ---------------------------------------------------------------------------
# Graph import (from ../Orion/src/agent/)
# ---------------------------------------------------------------------------

_graph = None
_loaded_nodes: List[str] = []


def _build_graph():
    global _graph, _loaded_nodes
    from langgraph.checkpoint.memory import MemorySaver

    try:
        from src.agent.graph import create_graph_with_checkpointer, ALL_NODES
        checkpointer = MemorySaver()
        _graph = create_graph_with_checkpointer(checkpointer=checkpointer, enable_verification=False)
        _loaded_nodes = sorted(ALL_NODES)
        logger.info(f"Graph compiled with MemorySaver — nodes: {_loaded_nodes}")
    except Exception as e:
        logger.error(f"FATAL: could not build graph: {e}", exc_info=True)
        sys.exit(1)


def get_graph():
    if _graph is None:
        _build_graph()
    return _graph


# ---------------------------------------------------------------------------
# Pydantic models (mirrors api_server.py, no auth)
# ---------------------------------------------------------------------------

class Attachment(BaseModel):
    name: str
    type: str
    data: str


class ChatRequest(BaseModel):
    message: str = ""
    user_id: Optional[str] = "test-local"
    user_name: Optional[str] = "Test User"
    session_id: Optional[str] = None
    interaction_mode: Optional[str] = "chat"
    llm_model: Optional[str] = ""
    automation_id: Optional[str] = None
    automation_md_content: Optional[str] = None
    automation_step: Optional[int] = None
    robot_ids: Optional[List[str]] = None
    equipment_id: Optional[str] = None
    attachments: Optional[List[Attachment]] = None


class ConfirmRequest(BaseModel):
    session_id: str
    answers: Union[dict, list]
    completed: bool = True
    cancelled: bool = False


# ---------------------------------------------------------------------------
# Helpers (adapted from Orion/test_server.py)
# ---------------------------------------------------------------------------

MAX_ATTACHMENT_SIZE = 10 * 1024 * 1024  # 10 MB


def _validate_attachments(attachments: Optional[List[Attachment]]):
    for att in (attachments or []):
        decoded_size = len(att.data) * 3 // 4
        if decoded_size > MAX_ATTACHMENT_SIZE:
            raise HTTPException(400, f"File {att.name} exceeds 10 MB limit")


def _clean_base64(data: str) -> str:
    return data.replace("\n", "").replace("\r", "").replace(" ", "")


def build_human_message(text: str, attachments: Optional[List[Attachment]] = None) -> HumanMessage:
    import base64
    if not attachments:
        return HumanMessage(content=text)

    content_blocks: List[dict] = []
    for att in attachments:
        if att.type.startswith("image/"):
            clean_data = _clean_base64(att.data)
            content_blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:{att.type};base64,{clean_data}"},
            })
        elif att.type == "application/pdf":
            try:
                import fitz
                pdf_bytes = base64.b64decode(att.data)
                pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                pdf_text = "\n".join(page.get_text() for page in pdf_doc)
                content_blocks.append({
                    "type": "text",
                    "text": f"[Attached PDF: {att.name}]\n\n{pdf_text[:50000]}",
                })
            except Exception as e:
                logger.warning(f"PDF extraction failed for {att.name}: {e}")
                content_blocks.append({
                    "type": "text",
                    "text": f"[Attached PDF: {att.name} — could not extract text]",
                })
        else:
            try:
                file_text = base64.b64decode(att.data).decode("utf-8")
                content_blocks.append({
                    "type": "text",
                    "text": f"[Attached file: {att.name}]\n\n{file_text[:20000]}",
                })
            except (UnicodeDecodeError, Exception):
                content_blocks.append({
                    "type": "text",
                    "text": f"[Attached file: {att.name} — binary file, cannot read as text]",
                })

    content_blocks.append({"type": "text", "text": text})
    return HumanMessage(content=content_blocks)


def sse_event(event_type: str, data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=True, default=str)
    return f"event: {event_type}\ndata: {payload}\n\n"


from src.agent.graph import ALL_NODES as _ALL_NODES_SET
_ALL_NODES = sorted(_ALL_NODES_SET)


def extract_response(event: dict) -> Optional[str]:
    if "synthesize" in event:
        msg = _extract_ai(event["synthesize"])
        if msg:
            return msg

    for node_name in ("chat", "tutor", "research", "troubleshooting",
                      "robot_operator", "analysis", "summarizer"):
        if node_name in event:
            msg = _extract_ai(event[node_name])
            if msg and len(msg) > 50:
                return msg
    return None


def _extract_ai(node_data: dict) -> Optional[str]:
    if not isinstance(node_data, dict):
        return None
    for msg in node_data.get("messages", []):
        if isinstance(msg, AIMessage):
            c = msg.content
            if isinstance(c, str):
                text = c.strip()
            elif isinstance(c, list):
                text = " ".join(
                    b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text"
                ).strip()
            else:
                text = str(c).strip() if c else ""
            if text:
                return text
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            text = (msg.get("content") or "").strip()
            if text:
                return text
    return None


def extract_suggestions(event: dict) -> list:
    for node_name in _ALL_NODES:
        if node_name in event and isinstance(event[node_name], dict):
            sugs = event[node_name].get("follow_up_suggestions", [])
            if sugs:
                return sugs
    return []


def extract_events_from_node(event: dict) -> list:
    events = []
    for node_name in _ALL_NODES:
        if node_name in event and isinstance(event[node_name], dict):
            for evt in event[node_name].get("events", []):
                events.append(evt)
    return events


def extract_chart_data(event: dict) -> Optional[dict]:
    for node_name in _ALL_NODES:
        if node_name in event and isinstance(event[node_name], dict):
            pc = event[node_name].get("pending_context")
            if isinstance(pc, dict) and "chart_data" in pc:
                return pc["chart_data"]
    return None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="ORION DevTools", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# WebSocket bridge infrastructure
# ---------------------------------------------------------------------------

BRIDGE_TOKEN = os.getenv("BRIDGE_TOKEN", "dev-bridge-token")
ROBOT_CONNECTIONS: Dict[str, WebSocket] = {}
ROBOT_METADATA: Dict[str, dict] = {}
PENDING_COMMANDS: Dict[str, dict] = {}
ROBOT_ACTION_LOG: Dict[str, list] = {}
_active_session: str = ""
TELEMETRY_LATEST: Dict[str, dict] = {}
TELEMETRY_LOG: list = []
TELEMETRY_RECORDING: bool = False
_main_loop: asyncio.AbstractEventLoop = None


@app.on_event("startup")
async def _capture_loop():
    global _main_loop
    _main_loop = asyncio.get_running_loop()


def get_main_loop() -> asyncio.AbstractEventLoop:
    return _main_loop


async def send_robot_command(robot_id: str, command: str, params: dict = None, timeout: float = 10.0) -> dict:
    """Send a command to a connected robot via WebSocket bridge and wait for response."""
    ws = ROBOT_CONNECTIONS.get(robot_id)

    # Match by device_type
    if not ws and isinstance(params, dict):
        target_type = params.get("_device_type", "")
        if target_type:
            for rid, meta in ROBOT_METADATA.items():
                if meta.get("type") == target_type and rid in ROBOT_CONNECTIONS:
                    robot_id = rid
                    ws = ROBOT_CONNECTIONS[rid]
                    break

    # Match by IP
    if not ws:
        search_ip = robot_id if "." in robot_id else ""
        if not search_ip and isinstance(params, dict):
            search_ip = params.get("plc_ip", "") or params.get("ip", "")
        if search_ip:
            for rid, meta in ROBOT_METADATA.items():
                if search_ip in meta.get("ips", []) and rid in ROBOT_CONNECTIONS:
                    robot_id = rid
                    ws = ROBOT_CONNECTIONS[rid]
                    break

    # Fallback to any connected device
    if not ws and ROBOT_CONNECTIONS:
        actual_id = next(iter(ROBOT_CONNECTIONS))
        logger.info(f"Robot '{robot_id}' not found, using '{actual_id}'")
        robot_id = actual_id
        ws = ROBOT_CONNECTIONS[actual_id]

    if not ws:
        return {
            "status": "error",
            "error": "No devices connected. Start the bridge: python bridge.py --config lab_config_dev.json",
            "connected_robots": [],
        }

    cmd_id = str(uuid.uuid4())
    event = asyncio.Event()
    PENDING_COMMANDS[cmd_id] = {"event": event, "result": None}

    try:
        await ws.send_json({
            "id": cmd_id,
            "command": command,
            "params": params or {},
        })
        await asyncio.wait_for(event.wait(), timeout=timeout)
        result = PENDING_COMMANDS[cmd_id]["result"]
        if result and isinstance(result, dict) and result.get("status") == "ok":
            entry = {
                "device_id": robot_id,
                "command": command,
                "params": params or {},
                "data": result.get("data", {}),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            if _active_session:
                ROBOT_ACTION_LOG.setdefault(_active_session, []).append(entry)
        return result if result else {"status": "error", "error": "No response received"}
    except asyncio.TimeoutError:
        return {"status": "error", "error": f"Command '{command}' timed out after {timeout}s"}
    except Exception as e:
        return {"status": "error", "error": str(e)}
    finally:
        PENDING_COMMANDS.pop(cmd_id, None)


async def notify_bridge(robot_id: str, message: dict) -> bool:
    """One-way notification to a connected bridge."""
    ws = ROBOT_CONNECTIONS.get(robot_id)
    if not ws:
        return False
    try:
        await ws.send_json(message)
        return True
    except Exception as e:
        logger.error(f"notify_bridge failed for '{robot_id}': {e}")
        return False


# Register for worker access (automation_worker_node imports from here)
try:
    from src.agent.utils.robot_commands import register as _register_robot_cmds
    _register_robot_cmds(send_robot_command, get_main_loop)
except ImportError:
    pass


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "nodes": _loaded_nodes,
        "timestamp": datetime.utcnow().isoformat(),
        "default_model": os.getenv("DEFAULT_MODEL", "(not set)"),
        "connected_robots": len(ROBOT_CONNECTIONS),
        "robots": list(ROBOT_METADATA.keys()),
    }


# ---------------------------------------------------------------------------
# POST /api/chat — SSE streaming
# ---------------------------------------------------------------------------

@app.post("/api/chat")
async def chat_stream(req: ChatRequest):
    graph = get_graph()

    session_id = req.session_id or f"test-{uuid.uuid4().hex[:12]}"
    global _active_session
    _active_session = session_id
    config = {"configurable": {"thread_id": session_id}}

    _validate_attachments(req.attachments)
    if req.attachments:
        for att in req.attachments:
            logger.info(f"[Chat] Attachment: {att.name} ({att.type}, ~{len(att.data) * 3 // 4 // 1024}KB)")

    human_msg = build_human_message(req.message, req.attachments)

    payload: Dict[str, Any] = {
        "messages": [human_msg],
        "user_name": req.user_name or "Test User",
        "user_id": req.user_id or "test-local",
        "interaction_mode": req.interaction_mode or "chat",
        "llm_model": req.llm_model or "",
        "image_attachments": [],
    }
    if req.robot_ids:
        payload["robot_ids"] = req.robot_ids
    if req.automation_id:
        payload["automation_id"] = req.automation_id
    if req.automation_md_content:
        payload["automation_md_content"] = req.automation_md_content
    if req.automation_step is not None:
        payload["automation_step"] = req.automation_step
    if req.equipment_id:
        payload["pending_context"] = {"equipment_id": req.equipment_id}

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            yield sse_event("session", {"session_id": session_id})
            yield sse_event("thinking", {"node": "start", "message": "Processing\u2026"})

            final_response = None
            all_suggestions: list = []
            chart_payload = None
            interrupted = False
            interrupt_payload = None

            loop = asyncio.get_running_loop()
            event_queue: asyncio.Queue = asyncio.Queue()

            def run_graph():
                try:
                    for event in graph.stream(payload, config=config, stream_mode="updates"):
                        loop.call_soon_threadsafe(event_queue.put_nowait, event)
                    loop.call_soon_threadsafe(event_queue.put_nowait, None)
                except Exception as exc:
                    err_msg = str(exc).encode("utf-8", errors="replace").decode("utf-8")
                    loop.call_soon_threadsafe(event_queue.put_nowait, {"__error__": err_msg})
                    loop.call_soon_threadsafe(event_queue.put_nowait, None)

            thread = loop.run_in_executor(None, run_graph)

            while True:
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=180)
                except asyncio.TimeoutError:
                    yield sse_event("error", {"message": "Timeout (3 min)"})
                    break

                if event is None:
                    break

                if isinstance(event, dict) and event.get("__error__"):
                    yield sse_event("error", {"message": event["__error__"]})
                    break

                # Interrupt (HITL)
                if isinstance(event, dict) and "__interrupt__" in event:
                    interrupted = True
                    int_data = event.get("__interrupt__", ())
                    if isinstance(int_data, (list, tuple)):
                        for item in int_data:
                            if hasattr(item, "value") and item.value:
                                interrupt_payload = item.value
                    elif isinstance(int_data, dict):
                        interrupt_payload = int_data
                    continue

                # Node events -> node_update
                node_events = extract_events_from_node(event)
                for evt in node_events:
                    yield sse_event("node_update", evt)

                # Which node just ran (for thinking indicator)
                for node_name in _ALL_NODES:
                    if node_name in event:
                        yield sse_event("thinking", {"node": node_name, "message": f"Running {node_name}\u2026"})
                        break

                sugs = extract_suggestions(event)
                if sugs:
                    all_suggestions = sugs

                cd = extract_chart_data(event)
                if cd:
                    chart_payload = cd

                response = extract_response(event)
                if response:
                    final_response = response

            await thread

            # HITL questions
            if interrupted and interrupt_payload:
                yield sse_event("questions", {
                    **(interrupt_payload if isinstance(interrupt_payload, dict) else {"prompt": str(interrupt_payload)}),
                    "session_id": session_id,
                })

            if all_suggestions:
                yield sse_event("suggestions", {"suggestions": all_suggestions})

            if final_response:
                yield sse_event("response", {"content": final_response, "session_id": session_id})

            if chart_payload:
                yield sse_event("chart", chart_payload)

            # Token usage (best-effort)
            try:
                final_state = graph.get_state(config)
                if final_state and hasattr(final_state, "values"):
                    tokens = final_state.values.get("token_usage", 0) or 0
                    yield sse_event("tokens", {"used": tokens})
            except Exception:
                pass

            for act in ROBOT_ACTION_LOG.pop(session_id, []):
                yield sse_event("robot_action", act)

            yield sse_event("done", {"session_id": session_id})

        except Exception as exc:
            safe_err = str(exc).encode("utf-8", errors="replace").decode("utf-8")
            logger.error(f"Stream error: {safe_err}", exc_info=True)
            yield sse_event("error", {"message": safe_err})
            yield sse_event("done", {"session_id": session_id})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# POST /api/confirm — HITL resume
# ---------------------------------------------------------------------------

@app.post("/api/confirm")
async def confirm_interrupt(req: ConfirmRequest):
    graph = get_graph()
    config = {"configurable": {"thread_id": req.session_id}}
    from langgraph.types import Command

    resume_data = {
        "answers": req.answers,
        "completed": req.completed,
        "cancelled": req.cancelled,
    }

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            yield sse_event("thinking", {"node": "start", "message": "Resuming\u2026"})

            final_response = None
            all_suggestions: list = []

            loop = asyncio.get_running_loop()
            event_queue: asyncio.Queue = asyncio.Queue()

            def run_resume():
                try:
                    for event in graph.stream(Command(resume=resume_data), config=config, stream_mode="updates"):
                        loop.call_soon_threadsafe(event_queue.put_nowait, event)
                    loop.call_soon_threadsafe(event_queue.put_nowait, None)
                except Exception as exc:
                    err_msg = str(exc).encode("utf-8", errors="replace").decode("utf-8")
                    loop.call_soon_threadsafe(event_queue.put_nowait, {"__error__": err_msg})
                    loop.call_soon_threadsafe(event_queue.put_nowait, None)

            thread = loop.run_in_executor(None, run_resume)

            while True:
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=180)
                except asyncio.TimeoutError:
                    yield sse_event("error", {"message": "Timeout"})
                    break

                if event is None:
                    break
                if isinstance(event, dict) and event.get("__error__"):
                    yield sse_event("error", {"message": event["__error__"]})
                    break

                for evt in extract_events_from_node(event):
                    yield sse_event("node_update", evt)

                sugs = extract_suggestions(event)
                if sugs:
                    all_suggestions = sugs

                response = extract_response(event)
                if response:
                    final_response = response

            await thread

            if all_suggestions:
                yield sse_event("suggestions", {"suggestions": all_suggestions})
            if final_response:
                yield sse_event("response", {"content": final_response, "session_id": req.session_id})

            yield sse_event("done", {"session_id": req.session_id})

        except Exception as exc:
            yield sse_event("error", {"message": str(exc)})
            yield sse_event("done", {"session_id": req.session_id})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# WebSocket robot bridge endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/robot")
async def ws_robot(ws: WebSocket):
    """WebSocket endpoint for robot bridge clients."""
    await ws.accept()

    try:
        init = await asyncio.wait_for(ws.receive_json(), timeout=10)
    except (asyncio.TimeoutError, Exception):
        await ws.close(code=4001, reason="Auth timeout")
        return

    token = init.get("token", "")
    robot_id = init.get("robot_id", "")

    if token != BRIDGE_TOKEN or not robot_id:
        await ws.close(code=4003, reason="Invalid token or missing robot_id")
        return

    ROBOT_CONNECTIONS[robot_id] = ws
    now_iso = datetime.utcnow().isoformat() + "Z"
    meta = {
        "type": init.get("type", init.get("robot_type", "unknown")),
        "model": init.get("model", ""),
        "protocol": init.get("protocol", "websocket"),
        "capabilities": init.get("capabilities", []),
        "ips": init.get("ips", []),
        "last_heartbeat": now_iso,
    }
    ROBOT_METADATA[robot_id] = meta

    # Mirror to shared_state for worker access
    try:
        from src.agent.shared_state import register_robot
        register_robot(robot_id, ws, meta)
    except ImportError:
        pass

    logger.info(f"Bridge connected: {robot_id} (type={meta['type']})")
    await ws.send_json({"type": "registered", "robot_id": robot_id})

    try:
        while True:
            data = await ws.receive_json()

            if robot_id in ROBOT_METADATA:
                ROBOT_METADATA[robot_id]["last_heartbeat"] = datetime.utcnow().isoformat() + "Z"

            cmd_id = data.get("id")
            if cmd_id and cmd_id in PENDING_COMMANDS:
                PENDING_COMMANDS[cmd_id]["result"] = data
                PENDING_COMMANDS[cmd_id]["event"].set()

            if data.get("type") == "status_update":
                logger.info(f"Robot {robot_id} status: {data}")

            if data.get("type") == "telemetry":
                TELEMETRY_LATEST[robot_id] = data
                if TELEMETRY_RECORDING:
                    TELEMETRY_LOG.append({
                        "robot_id": robot_id,
                        "timestamp": data.get("timestamp", ""),
                        **data.get("data", {}),
                    })
                continue

    except WebSocketDisconnect:
        logger.info(f"Bridge disconnected: {robot_id}")
    except Exception as e:
        logger.warning(f"Bridge error ({robot_id}): {e}")
    finally:
        ROBOT_CONNECTIONS.pop(robot_id, None)
        ROBOT_METADATA.pop(robot_id, None)
        try:
            from src.agent.shared_state import unregister_robot
            unregister_robot(robot_id)
        except ImportError:
            pass


@app.get("/api/robots")
async def list_robots():
    robots = []
    for rid in ROBOT_CONNECTIONS:
        meta = ROBOT_METADATA.get(rid, {})
        robots.append({
            "robot_id": rid,
            "connected": True,
            "type": meta.get("type", "unknown"),
            "model": meta.get("model", ""),
            "capabilities": meta.get("capabilities", []),
            "last_heartbeat": meta.get("last_heartbeat"),
        })
    return {"robots": robots, "count": len(robots)}


@app.get("/api/telemetry/latest")
async def telemetry_latest():
    return TELEMETRY_LATEST


@app.post("/api/telemetry/record")
async def telemetry_record_toggle():
    global TELEMETRY_RECORDING
    TELEMETRY_RECORDING = not TELEMETRY_RECORDING
    return {"recording": TELEMETRY_RECORDING, "samples": len(TELEMETRY_LOG)}


@app.get("/api/telemetry/export")
async def telemetry_export():
    return JSONResponse({"samples": TELEMETRY_LOG, "count": len(TELEMETRY_LOG)})


@app.post("/api/telemetry/clear")
async def telemetry_clear():
    global TELEMETRY_LOG
    TELEMETRY_LOG = []
    return {"cleared": True}


# ---------------------------------------------------------------------------
# Inline HTML UI
# ---------------------------------------------------------------------------

_HTML_UI = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ORION DevTools</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, system-ui, sans-serif;
  background: #fff; color: #111;
  height: 100vh; display: flex; flex-direction: column; overflow: hidden;
}

.top {
  height: 36px; border-bottom: 1px solid #e5e5e5;
  display: flex; align-items: center; padding: 0 12px; gap: 10px; font-size: 13px;
  flex-shrink: 0;
}
.top b { font-weight: 600; }
.top select, .top input {
  font-size: 12px; border: 1px solid #ddd; border-radius: 4px;
  padding: 2px 6px; background: #fff; color: #111; outline: none;
}
.top .spacer { flex: 1; }
.top button {
  font-size: 12px; border: 1px solid #ddd; border-radius: 4px;
  padding: 2px 10px; background: #fff; cursor: pointer;
}
.top button:hover { background: #f5f5f5; }
.status { font-size: 11px; color: #888; }
.status.on { color: #16a34a; }

.layout { flex: 1; display: flex; overflow: hidden; }

/* left sidebar */
.left-sb {
  width: 260px; flex-shrink: 0; border-right: 1px solid #e5e5e5;
  display: flex; flex-direction: column; overflow: hidden; font-size: 12px;
}
.left-sb.hidden { display: none; }
.left-sb-header {
  padding: 8px 12px; border-bottom: 1px solid #e5e5e5;
  display: flex; align-items: center; justify-content: space-between;
}
.left-sb-header span { font-size: 11px; font-weight: 600; color: #999; }
.left-sb-header button {
  font-size: 10px; border: 1px solid #e5e5e5; border-radius: 4px;
  padding: 2px 8px; background: #fff; cursor: pointer; color: #888;
}
.left-sb-header button:hover { background: #f5f5f5; color: #111; }
.left-sb-stats {
  padding: 6px 12px; border-bottom: 1px solid #e5e5e5;
  display: flex; gap: 12px; font-size: 11px; color: #888;
}
.left-sb-stats b { color: #111; font-weight: 600; }
.move-list { flex: 1; overflow-y: auto; }
.move-list::-webkit-scrollbar { width: 3px; }
.move-list::-webkit-scrollbar-thumb { background: #ddd; }
.move-item {
  padding: 5px 12px; border-bottom: 1px solid #f5f5f5;
  display: flex; gap: 8px; align-items: flex-start;
}
.move-item:hover { background: #fafafa; }
.move-num { font-family: monospace; font-size: 10px; color: #bbb; min-width: 18px; padding-top: 1px; }
.move-info { flex: 1; min-width: 0; }
.move-joint { font-weight: 600; font-size: 11px; }
.move-detail { font-size: 10px; color: #888; margin-top: 1px; }
.move-ts { font-family: monospace; font-size: 9px; color: #bbb; white-space: nowrap; padding-top: 1px; }
.empty-state { padding: 20px 12px; text-align: center; color: #bbb; font-size: 11px; }
.left-sb-export {
  padding: 6px 12px; border-top: 1px solid #e5e5e5; display: flex; gap: 4px; flex-shrink: 0;
}
.left-sb-export button {
  flex: 1; padding: 4px; border: 1px solid #e5e5e5; border-radius: 4px;
  background: #fff; font-size: 10px; cursor: pointer; color: #888;
}
.left-sb-export button:hover { background: #f5f5f5; color: #111; }

/* chat */
.chat { flex: 1; display: flex; flex-direction: column; min-width: 0; }
.messages { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
.msg-u { align-self: flex-end; background: #111; color: #fff; padding: 8px 14px; border-radius: 14px 14px 4px 14px; font-size: 13px; max-width: 75%; white-space: pre-wrap; word-break: break-word; }
.msg-a { font-size: 13px; line-height: 1.6; white-space: pre-wrap; word-break: break-word; max-width: 85%; }
.msg-a-label { font-size: 10px; font-weight: 600; color: #999; letter-spacing: 0.1em; margin-bottom: 2px; }
.msg-err { font-size: 12px; color: #dc2626; font-family: monospace; background: #fef2f2; padding: 6px 10px; border-radius: 6px; }
.msg-narr { font-size: 11px; color: #888; font-style: italic; border-left: 2px solid #e5e5e5; padding-left: 8px; }
.thinking { display: none; align-items: center; gap: 8px; padding: 4px 16px 8px; font-size: 12px; color: #888; }
.thinking.on { display: flex; }
.dot-pulse { width: 6px; height: 6px; background: #999; border-radius: 50%; animation: pulse 1s infinite; }
@keyframes pulse { 0%,100% { opacity:.3; } 50% { opacity:1; } }
.sugs { display: flex; flex-wrap: wrap; gap: 6px; padding: 0 16px 6px; }
.sug { padding: 4px 12px; border: 1px solid #e5e5e5; border-radius: 14px; font-size: 11px; color: #666; cursor: pointer; background: #fff; }
.sug:hover { border-color: #999; color: #111; }
.input-area { padding: 8px 16px 12px; }
.input-box { display: flex; border: 1px solid #ddd; border-radius: 10px; background: #fff; }
.input-box:focus-within { border-color: #999; }
.input-box textarea {
  flex: 1; border: none; padding: 10px 12px; font-size: 13px; font-family: inherit;
  background: transparent; color: #111; resize: none; outline: none;
  min-height: 40px; max-height: 120px; line-height: 1.5;
}
.input-box textarea::placeholder { color: #bbb; }
.input-box button { padding: 0 12px; border: none; background: transparent; cursor: pointer; font-size: 16px; color: #bbb; }
.input-box button.active { color: #111; }
#fileInput { display: none; }
.attach-row { display: flex; flex-wrap: wrap; gap: 4px; padding: 0 16px 4px; }
.att { font-size: 10px; color: #666; background: #f5f5f5; padding: 2px 8px; border-radius: 8px; }
.att span { color: #dc2626; cursor: pointer; margin-left: 4px; font-weight: 700; }
.hitl { display: none; padding: 12px 16px; border-top: 2px solid #f59e0b; }
.hitl.on { display: block; }
.hitl h3 { font-size: 13px; margin-bottom: 8px; }
.hitl label { display: block; font-size: 11px; color: #666; margin: 6px 0 2px; }
.hitl input, .hitl select { width: 100%; padding: 6px 8px; border: 1px solid #ddd; border-radius: 4px; font-size: 12px; }
.hitl .btns { margin-top: 10px; display: flex; gap: 6px; }
.hitl .btns button { padding: 5px 14px; border: 1px solid #ddd; border-radius: 4px; font-size: 12px; cursor: pointer; background: #fff; }
.hitl .btns button.p { background: #111; color: #fff; border-color: #111; }

/* right sidebar */
.sidebar {
  width: 340px; flex-shrink: 0; border-left: 1px solid #e5e5e5;
  display: flex; flex-direction: column; overflow: hidden; font-size: 12px;
}
.sidebar.hidden { display: none; }
.sb-section { border-bottom: 1px solid #e5e5e5; padding: 10px 12px; flex-shrink: 0; }
.sb-title { font-size: 11px; font-weight: 600; color: #999; margin-bottom: 6px; }
.sb-row { display: flex; justify-content: space-between; padding: 2px 0; }
.sb-row .k { color: #888; }
.sb-row .v { font-weight: 500; font-family: monospace; font-size: 11px; }
.sb-tabs { display: flex; border-bottom: 1px solid #e5e5e5; flex-shrink: 0; }
.sb-tab {
  flex: 1; padding: 6px 0; text-align: center; font-size: 11px; font-weight: 500;
  color: #999; cursor: pointer; border-bottom: 2px solid transparent;
}
.sb-tab:hover { color: #666; }
.sb-tab.on { color: #111; border-bottom-color: #111; }
.sb-panel { display: none; flex: 1; flex-direction: column; overflow: hidden; }
.sb-panel.on { display: flex; }
.sb-trace { flex: 1; overflow-y: auto; padding: 10px 12px; }
.sb-trace::-webkit-scrollbar { width: 3px; }
.sb-trace::-webkit-scrollbar-thumb { background: #ddd; }
.trace-run { margin-bottom: 12px; }
.trace-run-header { font-weight: 600; font-size: 12px; margin-bottom: 4px; display: flex; justify-content: space-between; }
.trace-run-header .ms { font-weight: 400; color: #888; font-family: monospace; }
.trace-node { display: flex; gap: 8px; padding: 3px 0; border-bottom: 1px solid #f5f5f5; }
.trace-node:last-child { border-bottom: none; }
.tn-bar { width: 3px; border-radius: 1px; flex-shrink: 0; }
.tn-body { flex: 1; min-width: 0; }
.tn-top { display: flex; justify-content: space-between; align-items: center; }
.tn-name { font-weight: 500; font-size: 11px; }
.tn-ms { font-family: monospace; font-size: 10px; color: #888; }
.tn-detail { font-size: 10px; color: #999; line-height: 1.4; margin-top: 1px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; cursor: default; }
.tn-detail:hover { white-space: normal; }
.tn-latbar { height: 3px; background: #f0f0f0; border-radius: 1px; margin-top: 3px; overflow: hidden; }
.tn-latfill { height: 100%; border-radius: 1px; }
.sb-raw { flex: 1; overflow-y: auto; padding: 6px 12px; font-family: monospace; font-size: 10px; }
.sb-raw::-webkit-scrollbar { width: 3px; }
.sb-raw::-webkit-scrollbar-thumb { background: #ddd; }
.raw-line { padding: 2px 0; border-bottom: 1px solid #fafafa; color: #888; display: flex; gap: 6px; }
.raw-line .rt { color: #bbb; white-space: nowrap; }
.raw-line .re { font-weight: 600; min-width: 60px; }
.raw-line .re.thinking { color: #f59e0b; }
.raw-line .re.response { color: #16a34a; }
.raw-line .re.error { color: #dc2626; }
.raw-line .re.node_update { color: #2563eb; }
.raw-line .rd { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
.raw-line:hover .rd { white-space: normal; word-break: break-all; }
.sb-export { padding: 8px 12px; border-top: 1px solid #e5e5e5; display: flex; gap: 4px; flex-shrink: 0; }
.sb-export button {
  flex: 1; padding: 4px; border: 1px solid #e5e5e5; border-radius: 4px;
  background: #fff; font-size: 10px; cursor: pointer; color: #888;
}
.sb-export button:hover { background: #f5f5f5; color: #111; }
</style>
</head>
<body>

<div class="top">
  <b>ORION DevTools</b>
  <span class="status" id="bridgeStatus">checking...</span>
  <div class="spacer"></div>
  <select id="cfgModel"><option value="">default model</option><option value="claude-sonnet-4-5-20250929">sonnet-4.5</option><option value="claude-sonnet-4-20250514">sonnet-4</option><option value="claude-haiku-4-5-20251001">haiku-4.5</option><option value="gpt-4o">gpt-4o</option></select>
  <select id="cfgMode"><option value="automation">automation</option><option value="chat">chat</option><option value="practice">practice</option><option value="troubleshoot">troubleshoot</option></select>
  <input id="cfgEquipment" placeholder="equipment_id" style="width:90px">
  <button onclick="clearAll()">clear</button>
  <button onclick="toggleLeft()">movements</button>
  <button onclick="toggleSidebar()">trace</button>
</div>

<div class="layout">

  <div class="left-sb" id="leftSb">
    <div class="left-sb-header">
      <span>Robot Data</span>
      <div style="display:flex;gap:4px">
        <button onclick="toggleRecord()" id="recBtn">record</button>
        <button onclick="clearMoves()">clear</button>
      </div>
    </div>
    <div id="telemLive" style="padding:4px 12px;font-family:monospace;font-size:10px;border-bottom:1px solid #e5e5e5;color:#888;">waiting for telemetry...</div>
    <div class="left-sb-stats">
      <span>actions: <b id="moveCount">0</b></span>
      <span>samples: <b id="telemCount">0</b></span>
      <span>joints: <b id="moveJoints">-</b></span>
    </div>
    <div class="move-list" id="moveList">
      <div class="empty-state">no movements yet</div>
    </div>
    <div class="left-sb-export">
      <button onclick="exportMovesCSV()">actions csv</button>
      <button onclick="exportTelemCSV()">telem csv</button>
      <button onclick="exportMovesJSON()">json</button>
    </div>
  </div>

  <div class="chat">
    <div class="messages" id="msgs"></div>
    <div class="sugs" id="sugs"></div>
    <div class="thinking" id="thinkBar"><div class="dot-pulse"></div><span id="thinkText">thinking...</span></div>
    <div class="hitl" id="hitl">
      <h3 id="hitlTitle">Input required</h3>
      <div id="hitlQs"></div>
      <div class="btns"><button class="p" onclick="submitHITL()">Submit</button><button onclick="cancelHITL()">Cancel</button></div>
    </div>
    <div class="attach-row" id="attRow"></div>
    <div class="input-area">
      <div class="input-box">
        <textarea id="input" rows="1" placeholder="message..." onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send()}" oninput="autoGrow(this)"></textarea>
        <button onclick="document.getElementById('fileInput').click()">+</button>
        <button id="sendBtn" onclick="send()">^</button>
        <input type="file" id="fileInput" multiple onchange="addFiles(this.files)">
      </div>
    </div>
  </div>

  <div class="sidebar" id="sidebar">
    <div class="sb-section">
      <div class="sb-title">Session</div>
      <div class="sb-row"><span class="k">id</span><span class="v" id="sId">-</span></div>
      <div class="sb-row"><span class="k">messages</span><span class="v" id="sMsgs">0</span></div>
      <div class="sb-row"><span class="k">errors</span><span class="v" id="sErrs">0</span></div>
      <div class="sb-row"><span class="k">tokens (last)</span><span class="v" id="sTokens">-</span></div>
      <div class="sb-row"><span class="k">avg latency</span><span class="v" id="sAvg">-</span></div>
      <div class="sb-row"><span class="k">bridge</span><span class="v" id="sBridge">-</span></div>
    </div>
    <div class="sb-tabs">
      <div class="sb-tab on" onclick="sbTab('trace',this)">Trace</div>
      <div class="sb-tab" onclick="sbTab('raw',this)">Raw</div>
    </div>
    <div class="sb-panel on" id="pTrace"><div class="sb-trace" id="traceContainer"></div></div>
    <div class="sb-panel" id="pRaw"><div class="sb-raw" id="rawContainer"></div></div>
    <div class="sb-export">
      <button onclick="exportJSON()">export json</button>
      <button onclick="exportCSV()">export csv</button>
      <button onclick="copyLast()">copy last trace</button>
    </div>
  </div>
</div>

<script>
var $ = function(id) { return document.getElementById(id); };
var BASE = 'http://localhost:8000';
var esc = function(s) { var d = document.createElement('div'); d.textContent = s; return d.innerHTML; };

var sessionId = '', busy = false, atts = [], hitlSid = '';
var msgCount = 0, errCount = 0, lastTokens = 0;
var runs = [], cur = null;
var movements = [];

function addMovement(mv) {
  mv.n = movements.length + 1;
  mv.time = new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'});
  mv.timestamp = new Date().toISOString();
  movements.push(mv);
  renderMoves();
}

function renderMoves() {
  $('moveCount').textContent = movements.length;
  var joints = {};
  movements.forEach(function(m) { if (m.joint !== '-') joints[m.joint] = true; });
  var jk = Object.keys(joints);
  $('moveJoints').textContent = jk.length > 0 ? jk.sort().join(', ') : '-';
  if (movements.length === 0) { $('moveList').innerHTML = '<div class="empty-state">no movements yet</div>'; return; }
  var html = '';
  for (var i = movements.length - 1; i >= 0; i--) {
    var m = movements[i];
    var detail = '';
    if(m.command==='move_joint'&&m.from!=null)detail=m.from+' -> '+m.to+' deg';
    else if(m.command==='move_joint')detail=m.to+' deg';
    else if(m.final_angles)detail=m.final_angles.map(function(a,i){return'J'+(i+1)+':'+Math.round(a*10)/10;}).join(' ');
    else if(m.name)detail=m.name;
    else detail=m.command||'';
    var color = m.status === 'ok' ? '#16a34a' : m.status === 'error' ? '#dc2626' : m.status === 'noop' ? '#888' : '#2563eb';
    html += '<div class="move-item"><div class="move-num">' + m.n + '</div><div class="move-info"><div class="move-joint" style="color:' + color + '">' + esc(m.joint + (m.name ? ' (' + m.name + ')' : '')) + '</div><div class="move-detail">' + esc(detail) + '</div></div><div class="move-ts">' + m.time + '</div></div>';
  }
  $('moveList').innerHTML = html;
}

function clearMoves() { movements = []; renderMoves(); }
function exportMovesCSV(){
  var csv='n,time,timestamp,device,command,joint,name,from,to,j1,j2,j3,j4,j5,j6,status\n';
  movements.forEach(function(m){var fa=m.final_angles||[];csv+=m.n+','+m.time+','+m.timestamp+','+(m.device||'')+','+(m.command||'')+','+m.joint+',"'+(m.name||'')+'",'+(m.from!=null?m.from:'')+','+(m.to!=null?m.to:'')+','+(fa[0]!=null?fa[0]:'')+','+(fa[1]!=null?fa[1]:'')+','+(fa[2]!=null?fa[2]:'')+','+(fa[3]!=null?fa[3]:'')+','+(fa[4]!=null?fa[4]:'')+','+(fa[5]!=null?fa[5]:'')+','+m.status+'\n';});
  dl('movements.csv',csv,'text/csv');
}
function exportMovesJSON() { dl('movements.json', JSON.stringify(movements, null, 2), 'application/json'); }
function copyMoves() {
  var t = movements.map(function(m) { return m.n+'. ['+m.time+'] '+m.joint+(m.name?' ('+m.name+')':'')+' -> '+(m.to!==null?m.to+'deg':'?')+(m.status==='error'?' FAILED':''); }).join('\n');
  navigator.clipboard.writeText(t);
}

// --- sidebars ---
function toggleLeft() { $('leftSb').classList.toggle('hidden'); }
function toggleSidebar() { $('sidebar').classList.toggle('hidden'); }
function sbTab(name, el) {
  document.querySelectorAll('.sb-tab').forEach(function(t){t.classList.remove('on');});
  document.querySelectorAll('.sb-panel').forEach(function(p){p.classList.remove('on');});
  el.classList.add('on');
  $('p'+name.charAt(0).toUpperCase()+name.slice(1)).classList.add('on');
}

// --- chat ---
function scrollDown() { var m = $('msgs'); requestAnimationFrame(function(){m.scrollTop=m.scrollHeight;}); }
function ts() { return new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'}); }
function addChat(type, text) {
  var d = document.createElement('div');
  if (type==='user') d.innerHTML='<div class="msg-u">'+esc(text)+'</div>';
  else if (type==='ai') d.innerHTML='<div class="msg-a-label">ORION</div><div class="msg-a">'+esc(text)+'</div>';
  else if (type==='err') d.innerHTML='<div class="msg-err">'+esc(text)+'</div>';
  else if (type==='narr') d.innerHTML='<div class="msg-narr">'+esc(text)+'</div>';
  $('msgs').appendChild(d); scrollDown();
}
function setThink(on,msg){$('thinkBar').classList.toggle('on',on);if(msg)$('thinkText').textContent=msg;}
function autoGrow(el){el.style.height='auto';el.style.height=Math.min(el.scrollHeight,120)+'px';$('sendBtn').classList.toggle('active',el.value.trim().length>0);}
function addFiles(files){for(var i=0;i<files.length;i++){(function(f){var r=new FileReader();r.onload=function(){atts.push({name:f.name,type:f.type||'application/octet-stream',data:r.result.split(',')[1]});renderAtts();};r.readAsDataURL(f);})(files[i]);}$('fileInput').value='';}
function renderAtts(){$('attRow').innerHTML=atts.map(function(a,i){return'<span class="att">'+esc(a.name)+'<span onclick="atts.splice('+i+',1);renderAtts()">x</span></span>';}).join('');}
function showSugs(list){$('sugs').innerHTML=list.map(function(s){var t=typeof s==='string'?s:s.text||JSON.stringify(s);return'<span class="sug" onclick="$(\'input\').value=this.textContent;send()">'+esc(t)+'</span>';}).join('');}

// --- trace ---
function startRun(p){cur={id:Date.now(),prompt:p,response:'',nodes:[],totalMs:0,t0:performance.now()};}
function addNode(n,d){if(!cur)return;var now=performance.now();var prev=cur.nodes.length?cur.nodes[cur.nodes.length-1]._t:cur.t0;cur.nodes.push({name:n,detail:d||'',ms:Math.round(now-prev),_t:now,wall:ts()});}
function endRun(r){if(!cur)return;cur.totalMs=Math.round(performance.now()-cur.t0);cur.response=r||'';runs.push(cur);renderTrace();updateStats();cur=null;}
function renderTrace(){
  var c=$('traceContainer');c.innerHTML='';
  for(var i=runs.length-1;i>=0;i--){var r=runs[i];var mx=Math.max.apply(null,r.nodes.map(function(n){return n.ms;}));if(mx<1)mx=1;
  var h='<div class="trace-run"><div class="trace-run-header"><span>'+esc(r.prompt.substring(0,60))+'</span><span class="ms">'+(r.totalMs/1000).toFixed(1)+'s</span></div>';
  r.nodes.forEach(function(n){var p=Math.min((n.ms/mx)*100,100);var co=n.ms<2000?'#16a34a':n.ms<8000?'#f59e0b':'#dc2626';
  h+='<div class="trace-node"><div class="tn-bar" style="background:'+co+'"></div><div class="tn-body"><div class="tn-top"><span class="tn-name">'+esc(n.name)+'</span><span class="tn-ms">'+n.ms+'ms</span></div>';
  if(n.detail)h+='<div class="tn-detail">'+esc(n.detail)+'</div>';
  h+='<div class="tn-latbar"><div class="tn-latfill" style="width:'+p+'%;background:'+co+'"></div></div></div></div>';});
  h+='</div>';c.innerHTML+=h;}
}
function updateStats(){$('sMsgs').textContent=msgCount;$('sErrs').textContent=errCount;if(lastTokens)$('sTokens').textContent=lastTokens;if(runs.length){var avg=Math.round(runs.reduce(function(s,r){return s+r.totalMs;},0)/runs.length);$('sAvg').textContent=(avg/1000).toFixed(1)+'s';}}
function logRaw(type,data){var d=document.createElement('div');d.className='raw-line';d.innerHTML='<span class="rt">'+ts()+'</span><span class="re '+type+'">'+type+'</span><span class="rd">'+esc(JSON.stringify(data)).substring(0,300)+'</span>';$('rawContainer').appendChild(d);$('rawContainer').scrollTop=$('rawContainer').scrollHeight;}

// --- send ---
async function send(){
  if(busy)return;var text=$('input').value.trim();if(!text&&!atts.length)return;
  busy=true;if(text){addChat('user',text);msgCount++;}
  $('input').value='';$('input').style.height='auto';$('sendBtn').classList.remove('active');$('sugs').innerHTML='';setThink(true,'thinking...');startRun(text);
  var body={message:text,user_name:'DevTools',user_id:'devtools',interaction_mode:$('cfgMode').value,llm_model:$('cfgModel').value};
  var eq=$('cfgEquipment').value.trim();if(eq)body.equipment_id=eq;
  if(sessionId)body.session_id=sessionId;if(atts.length)body.attachments=atts;atts=[];renderAtts();
  var f='';try{f=await doStream(BASE+'/api/chat',body);}catch(e){addChat('err',e.message);errCount++;}
  endRun(f);setThink(false);busy=false;
}
async function doStream(url,body){
  var resp=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(!resp.ok){addChat('err','HTTP '+resp.status);errCount++;return'';}
  var reader=resp.body.getReader(),dec=new TextDecoder();var buf='',et='',final='';
  while(true){var chunk=await reader.read();if(chunk.done)break;buf+=dec.decode(chunk.value,{stream:true});var lines=buf.split('\n');buf=lines.pop();
  for(var li=0;li<lines.length;li++){var ln=lines[li];if(ln.startsWith('event: '))et=ln.slice(7).trim();else if(ln.startsWith('data: ')&&et){try{final=handleSSE(et,JSON.parse(ln.slice(6)),final);}catch(e){}et='';}}}
  return final;
}
function handleSSE(type,data,final){
  logRaw(type,data);
  switch(type){
    case'session':sessionId=data.session_id||sessionId;$('sId').textContent=sessionId.substring(0,20);break;
    case'thinking':setThink(true,data.message||(data.node+'...'));if(data.node&&data.node!=='start')addNode(data.node,data.message||'');break;
    case'node_update':if(data.type==='narration'&&data.content){addChat('narr',data.content);addNode(data.source||'narration',data.content);}else if(data.content)addNode(data.source||data.node||'?',data.content.substring(0,200));break;
    case'response':setThink(false);if(data.content){addChat('ai',data.content);final=data.content;}break;
    case'suggestions':showSugs(data.suggestions||[]);break;
    case'questions':showHITL(data);break;
    case'tokens':lastTokens=data.total||data.used||0;$('sTokens').textContent=lastTokens+(data.model?' ('+data.model+')':'');break;
    case 'robot_action':
      var rd=data.data||{};var cmd=data.command||'';
      var mv={n:movements.length+1,time:ts(),timestamp:data.timestamp||new Date().toISOString(),device:data.device_id||'',command:cmd,status:'ok'};
      if(cmd==='move_joint'){mv.joint='J'+(rd.target_joint||'?');mv.name=rd.joint_name||'';mv.from=rd.previous_angle!=null?rd.previous_angle:null;mv.to=rd.target_angle!=null?rd.target_angle:null;mv.final_angles=rd.final_angles||null;}
      else if(cmd==='home'){mv.joint='ALL';mv.name='home';mv.to=0;mv.final_angles=[0,0,0,0,0,0];}
      else if(cmd==='go_to_pose'){mv.joint='ALL';mv.name=rd.pose||'';mv.final_angles=rd.final_angles||null;}
      else if(cmd==='say_hi'){mv.joint='ALL';mv.name='wave';}
      else if(cmd==='get_position'||cmd==='get_full_status'){mv.joint='ALL';mv.name=cmd;mv.status='query';mv.final_angles=rd.joints||null;}
      else{mv.joint='-';mv.name=cmd;}
      movements.push(mv);renderMoves();break;
    case'error':setThink(false);addChat('err',data.message||'error');errCount++;addNode('error',data.message||'');break;
    case'done':setThink(false);break;
  }
  return final;
}

// --- hitl ---
function showHITL(data){setThink(false);hitlSid=data.session_id||sessionId;$('hitlTitle').textContent=data.title||'Input required';var c=$('hitlQs');c.innerHTML='';var qs=data.questions||[];if(!qs.length&&data.prompt){addChat('ai',data.prompt);return;}
qs.forEach(function(q,i){var d=document.createElement('div');d.innerHTML='<label>'+esc(q.question||q.label||'Q'+(i+1))+'</label>';if(q.options&&q.options.length){var sel=document.createElement('select');sel.dataset.key=q.key||('q'+i);q.options.forEach(function(o){var opt=document.createElement('option');opt.value=typeof o==='string'?o:o.value;opt.textContent=typeof o==='string'?o:o.label;sel.appendChild(opt);});d.appendChild(sel);}else{var inp=document.createElement('input');inp.dataset.key=q.key||('q'+i);inp.placeholder=q.placeholder||'';d.appendChild(inp);}c.appendChild(d);});$('hitl').classList.add('on');}
async function submitHITL(){var ans={};$('hitlQs').querySelectorAll('input,select').forEach(function(el){ans[el.dataset.key]=el.value;});$('hitl').classList.remove('on');setThink(true,'submitting...');busy=true;try{await doStream(BASE+'/api/confirm',{session_id:hitlSid,answers:ans,completed:true,cancelled:false});}catch(e){addChat('err',e.message);}setThink(false);busy=false;}
function cancelHITL(){$('hitl').classList.remove('on');}

// --- exports (trace) ---
function exportJSON(){var d={runs:runs.map(function(r){return{prompt:r.prompt,response:r.response,totalMs:r.totalMs,nodes:r.nodes.map(function(n){return{name:n.name,ms:n.ms,detail:n.detail,wall:n.wall};})};})};dl('trace.json',JSON.stringify(d,null,2),'application/json');}
function exportCSV(){var csv='run,prompt,node,ms,detail\n';runs.forEach(function(r,i){r.nodes.forEach(function(n){csv+=(i+1)+',"'+r.prompt.replace(/"/g,'""')+'","'+n.name+'",'+n.ms+',"'+(n.detail||'').replace(/"/g,'""')+'"\n';});});dl('trace.csv',csv,'text/csv');}
function dl(name,content,mime){var a=document.createElement('a');a.href=URL.createObjectURL(new Blob([content],{type:mime}));a.download=name;a.click();}
function copyLast(){var r=runs[runs.length-1];if(!r)return;navigator.clipboard.writeText(r.nodes.map(function(n){return'['+n.wall+'] '+n.name+' ('+n.ms+'ms) '+(n.detail||'');}).join('\n'));}

// --- bridge ---
async function pollBridge(){try{var r=await fetch(BASE+'/api/robots',{signal:AbortSignal.timeout(3000)});var d=await r.json();var n=d.count||0;$('bridgeStatus').textContent=n>0?n+' device'+(n>1?'s':'')+' connected':'no bridge';$('bridgeStatus').className='status'+(n>0?' on':'');$('sBridge').textContent=n>0?d.robots.map(function(r){return r.robot_id;}).join(', '):'none';}catch(e){$('bridgeStatus').textContent='offline';$('bridgeStatus').className='status';}}
setInterval(pollBridge,3000);pollBridge();

function clearAll(){$('msgs').innerHTML='';$('rawContainer').innerHTML='';$('traceContainer').innerHTML='';$('sugs').innerHTML='';runs=[];cur=null;msgCount=0;errCount=0;lastTokens=0;sessionId='';$('sId').textContent='-';$('sTokens').textContent='-';updateStats();clearMoves();}
var telemData=[];var telemRecording=false;
function pollTelemetry(){
  fetch(BASE+'/api/telemetry/latest',{signal:AbortSignal.timeout(2000)}).then(function(r){return r.json();}).then(function(d){
    var keys=Object.keys(d);if(!keys.length)return;
    var dev=keys[0];var t=d[dev];var dd=t.data||{};
    var j=dd.joints_deg||[];
    var html='<b>'+esc(dev)+'</b> '+(dd.state||'')+' ';
    for(var i=0;i<6;i++){html+='J'+(i+1)+':'+(j[i]!=null?j[i].toFixed(1):'-')+' ';if(i===2)html+='<br>';}
    $('telemLive').innerHTML=html;
    if(telemRecording&&j.length){
      var row={t:new Date().toISOString(),device:dev};
      j.forEach(function(v,i){row['j'+(i+1)]=Math.round(v*100)/100;});
      (dd.velocities||[]).forEach(function(v,i){row['v'+(i+1)]=Math.round(v*1000)/1000;});
      (dd.efforts||[]).forEach(function(v,i){row['e'+(i+1)]=Math.round(v*1000)/1000;});
      if(dd.temperatures)(dd.temperatures).forEach(function(v,i){row['temp'+(i+1)]=v;});
      telemData.push(row);$('telemCount').textContent=telemData.length;
    }
  }).catch(function(){});
}
setInterval(pollTelemetry,200);

function toggleRecord(){
  telemRecording=!telemRecording;
  $('recBtn').textContent=telemRecording?'stop':'record';
  $('recBtn').style.color=telemRecording?'#dc2626':'';
  fetch(BASE+'/api/telemetry/record',{method:'POST'});
}

function exportTelemCSV(){
  if(!telemData.length)return;
  var keys=Object.keys(telemData[0]);
  var csv=keys.join(',')+'\n';
  telemData.forEach(function(r){csv+=keys.map(function(k){return r[k]!=null?r[k]:'';}).join(',')+'\n';});
  dl('telemetry.csv',csv,'text/csv');
}

$('input').focus();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return _HTML_UI


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _build_graph()
    print(f"\n  ORION DevTools - Local Test Server")
    print(f"  -----------------------------------")
    print(f"  Agent source: {ORION_ROOT}")
    print(f"  Bridge: SimXArmBridge")
    print(f"  DEFAULT_MODEL = {os.getenv('DEFAULT_MODEL', '(not set)')}")
    print(f"  Open \033[4mhttp://localhost:8000\033[0m\n", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

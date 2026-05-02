"""
test_server.py

Local dev server for ORION. Spins up the agent graph with an in-memory
checkpointer and a WS endpoint so the bridge can connect without needing
Cloud Run or Supabase.

Supports all interaction modes:
  - chat / automation / troubleshoot: as-is
  - practice: loads automation MDs from lab.automations (Supabase) at runtime,
    tracks step progress in RAM, surfaces a stepper panel in the UI when
    this mode is selected

Usage:
    cd C:\\Products\\FINAL_PRODUCTS\\orion-devtools
    ..\\Orion\\.venv\\Scripts\\Activate.ps1
    pip install -r requirements.txt
    python test_server.py
"""

import os
import re
import sys
import io
import json
import asyncio
import logging
import uuid
from datetime import datetime
from typing import AsyncGenerator, Any, Dict, List, Optional, Union

ORION_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Orion'))
sys.path.insert(0, ORION_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ORION_ROOT, '.env'))

CLOUD_AGENT_URL: str = os.getenv("CLOUD_AGENT_URL", "").rstrip("/")
CLOUD_AGENT_TOKEN: str = os.getenv("CLOUD_AGENT_TOKEN", "")
BACKEND_MODE: str = "local"
_http_client = None  # httpx.AsyncClient, lazy-init on first cloud request

# windows terminal chokes on unicode from LLM output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import httpx

from langchain_core.messages import HumanMessage, AIMessage

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("orion_devtools")

# ── STREAM CALLBACK REGISTRY ──────────────────────────────────────────────────
from src.agent.utils.stream_utils import (
    register_stream_callback,
    unregister_stream_callback,
    get_stream_callback,
)
from src.agent.services import init_services, get_supabase


# ── GRAPH SETUP ───────────────────────────────────────────────────────────────

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
        logger.info(f"graph ready — nodes: {_loaded_nodes}")
    except Exception as e:
        logger.error(f"couldn't build graph: {e}", exc_info=True)
        sys.exit(1)


def get_graph():
    if _graph is None:
        _build_graph()
    return _graph


# ── MODELS ────────────────────────────────────────────────────────────────────

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
    automation_md_content: Optional[str] = None  # optional override
    automation_step: Optional[int] = None
    robot_ids: Optional[List[str]] = None
    equipment_id: Optional[str] = None
    attachments: Optional[List[Attachment]] = None


class ConfirmRequest(BaseModel):
    session_id: str
    answers: Union[dict, list]
    completed: bool = True
    cancelled: bool = False


class BackendModeRequest(BaseModel):
    mode: str


# ── PRACTICE LOADING (Supabase) ───────────────────────────────────────────────

_AUTOMATIONS_CACHE: Dict[str, dict] = {}

# In-RAM practice progress — {session_id: {automation_id, current_step, ...}}
PRACTICE_SESSIONS: Dict[str, dict] = {}


def fetch_automations_list() -> List[dict]:
    """Metadata for all automations (no md_content — keeps payload small)."""
    sb = get_supabase()
    if sb is None:
        logger.warning("Supabase not available — returning empty automations list")
        return []
    try:
        resp = (
            sb.schema("lab")
            .table("automations")
            .select("id, title, description, type, difficulty, sort_order")
            .order("sort_order")
            .execute()
        )
        return resp.data or []
    except Exception as e:
        logger.error(f"fetch_automations_list failed: {e}")
        return []


def fetch_automation(automation_id: str) -> Optional[dict]:
    """Single automation with md_content. Cached in-process."""
    if automation_id in _AUTOMATIONS_CACHE:
        return _AUTOMATIONS_CACHE[automation_id]
    sb = get_supabase()
    if sb is None:
        return None
    try:
        resp = (
            sb.schema("lab")
            .table("automations")
            .select("id, title, description, type, difficulty, md_content, sort_order")
            .eq("id", automation_id)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            return None
        row = rows[0]
        _AUTOMATIONS_CACHE[automation_id] = row
        return row
    except Exception as e:
        logger.error(f"fetch_automation({automation_id}) failed: {e}")
        return None


def parse_practice_steps(md_content: str) -> List[dict]:
    """
    Parse MD into ordered steps.

    Strategy:
      1. Prefer headings that match the `PASO N` / `STEP N` / `PASO N:` pattern
         (case-insensitive, tolerates acentos). These are the actual steps.
         Everything else at `## ` level (CONTEXTO, TONO, REGLAS, AL FINALIZAR...)
         is treated as meta and ignored.
      2. If no headings match that pattern, fall back to treating every `## `
         heading as a step — keeps compatibility with practices using a
         different convention.
    """
    if not md_content:
        return []

    lines = md_content.splitlines()

    # Capture every `## ` heading with its line index and raw title
    heading_re = re.compile(r"^##\s+(.+?)\s*$")
    step_re = re.compile(r"^\s*(paso|step)\s+(\d+)\b", re.IGNORECASE)

    headings: List[dict] = []  # {line_idx, title, is_step, step_num}
    for i, line in enumerate(lines):
        if line.startswith("###"):
            continue
        m = heading_re.match(line)
        if not m:
            continue
        title = m.group(1).strip()
        sm = step_re.match(title)
        headings.append({
            "line_idx": i,
            "title": title,
            "is_step": sm is not None,
            "step_num": int(sm.group(2)) if sm else None,
        })

    if not headings:
        return []

    # Pick which headings actually count as steps
    step_headings = [h for h in headings if h["is_step"]]
    if not step_headings:
        step_headings = headings  # fallback: no explicit pattern → treat all

    # For each chosen heading, body = lines between that heading and the NEXT
    # heading (of any kind), so meta-sections interleaved between steps don't
    # leak into the step body.
    all_heading_lines = [h["line_idx"] for h in headings] + [len(lines)]

    steps: List[dict] = []
    for idx, h in enumerate(step_headings):
        start = h["line_idx"] + 1
        # Find the first heading boundary strictly after this one
        end = next((ln for ln in all_heading_lines if ln > h["line_idx"]), len(lines))
        body = "\n".join(lines[start:end]).strip()

        summary = ""
        for ln in body.splitlines():
            t = ln.strip()
            if not t or t.startswith("#"):
                continue
            # Drop leading markdown bold markers like `**Qué hacer:**`
            clean = re.sub(r"^\*+[^*]+?\*+\s*:?\s*", "", t)
            summary = (clean or t)[:140]
            break

        steps.append({
            "index": idx + 1,
            "title": h["title"],
            "body": body,
            "summary": summary,
        })

    return steps


# ── HELPERS ───────────────────────────────────────────────────────────────────

MAX_ATTACHMENT_SIZE = 10 * 1024 * 1024


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
                logger.warning(f"couldn't extract PDF {att.name}: {e}")
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

_RESPONSE_NODES = ("chat", "tutor", "research", "troubleshooting",
                    "analysis", "summarizer")


def extract_response(event: dict) -> Optional[str]:
    if "synthesize" in event:
        msg = _extract_ai(event["synthesize"])
        if msg:
            return msg
    for node_name in _RESPONSE_NODES:
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


def extract_automation_step(event: dict) -> Optional[int]:
    """Scan every node slice for an automation_step update."""
    for node_name in _ALL_NODES:
        if node_name in event and isinstance(event[node_name], dict):
            step = event[node_name].get("automation_step")
            if step is not None:
                try:
                    return int(step)
                except (TypeError, ValueError):
                    pass
    return None


def extract_chart_data(event: dict) -> Optional[dict]:
    for node_name in _ALL_NODES:
        if node_name in event and isinstance(event[node_name], dict):
            pc = event[node_name].get("pending_context")
            if isinstance(pc, dict) and "chart_data" in pc:
                return pc["chart_data"]
    return None


# ── APP ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="ORION DevTools", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── WEBSOCKET BRIDGE ──────────────────────────────────────────────────────────

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

# ── Recording sessions (student practice) ────────────────────────────────────
# Structure per session:
#   {
#     "session_id": str,
#     "device_id": str,
#     "started_at": iso,
#     "stopped_at": iso | None,
#     "active": bool,
#     "events": [bridge_event, ...],        # discrete movements
#     "stream": [telemetry_stream, ...],    # 10Hz rich telemetry
#     "summary": dict | None,               # computed at stop
#   }
RECORDING_SESSIONS: Dict[str, dict] = {}
# Lab indexing — lab_id → set of robot_ids in that lab
LAB_DEVICES: Dict[str, set] = {}


def _get_lab_token(lab_id: str) -> str:
    """
    Resolve the bridge token for a given lab_id.
    Looks up BRIDGE_TOKEN_<LAB_UPPER>. Falls back to BRIDGE_TOKEN
    so legacy bridges (no lab_id) keep working.
    """
    if not lab_id or lab_id == "default":
        return BRIDGE_TOKEN
    env_var = f"BRIDGE_TOKEN_{lab_id.upper().replace('-', '_')}"
    return os.getenv(env_var, BRIDGE_TOKEN)

@app.on_event("startup")
async def _capture_loop():
    global _main_loop
    _main_loop = asyncio.get_running_loop()
    init_services()


@app.on_event("shutdown")
async def _shutdown_http_client():
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient()
    return _http_client


def _cloud_headers() -> dict:
    """Build headers for cloud agent requests. Omits Authorization if no token set."""
    if CLOUD_AGENT_TOKEN:
        return {"Authorization": f"Bearer {CLOUD_AGENT_TOKEN}"}
    return {}


def get_main_loop() -> asyncio.AbstractEventLoop:
    return _main_loop


async def send_robot_command(robot_id: str, command: str, params: dict = None, timeout: float = 10.0) -> dict:
    ws = ROBOT_CONNECTIONS.get(robot_id)

    if not ws and isinstance(params, dict):
        target_type = params.get("_device_type", "")
        if target_type:
            for rid, meta in ROBOT_METADATA.items():
                if meta.get("type") == target_type and rid in ROBOT_CONNECTIONS:
                    robot_id = rid
                    ws = ROBOT_CONNECTIONS[rid]
                    break

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

    if not ws and ROBOT_CONNECTIONS:
        actual_id = next(iter(ROBOT_CONNECTIONS))
        logger.info(f"'{robot_id}' not found, falling back to '{actual_id}'")
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
        return {"status": "error", "error": f"'{command}' timed out ({timeout}s)"}
    except Exception as e:
        return {"status": "error", "error": str(e)}
    finally:
        PENDING_COMMANDS.pop(cmd_id, None)


async def notify_bridge(robot_id: str, message: dict) -> bool:
    ws = ROBOT_CONNECTIONS.get(robot_id)
    if not ws:
        return False
    try:
        await ws.send_json(message)
        return True
    except Exception as e:
        logger.error(f"notify_bridge failed for '{robot_id}': {e}")
        return False


async def send_bridge_context_update(
    robot_id: str,
    *,
    reactive_enabled: bool,
    session_id: str,
    thread_id: str = "",
    user_id: str = "",
    practice: Optional[dict] = None,
) -> bool:
    """
    Notify the bridge to enable/disable its MovementObserver + telemetry stream.
    Consumed by ReactiveContextManager.update() on the bridge side.
    """
    return await notify_bridge(robot_id, {
        "type": "bridge_context_update",
        "mode": "practice",
        "reactive_enabled": reactive_enabled,
        "session_id": session_id,
        "thread_id": thread_id or session_id,
        "user_id": user_id,
        "device_id": robot_id,
        "practice": practice or {},
    })


try:
    from src.agent.utils.robot_commands import register as _register_robot_cmds
    _register_robot_cmds(send_robot_command, get_main_loop)
except ImportError:
    pass


@app.get("/api/backend-mode")
async def get_backend_mode():
    return {
        "mode": BACKEND_MODE,
        "cloud_url": CLOUD_AGENT_URL,
        "cloud_configured": bool(CLOUD_AGENT_URL),
    }


@app.post("/api/backend-mode")
async def set_backend_mode(req: BackendModeRequest):
    global BACKEND_MODE
    if req.mode not in ("local", "cloud"):
        raise HTTPException(400, f"Invalid mode: {req.mode}")
    if req.mode == "cloud" and not CLOUD_AGENT_URL:
        raise HTTPException(400, "CLOUD_AGENT_URL not set")
    BACKEND_MODE = req.mode
    return {
        "mode": BACKEND_MODE,
        "cloud_url": CLOUD_AGENT_URL,
        "cloud_configured": bool(CLOUD_AGENT_URL),
    }


@app.get("/health")
async def health():
    sb = get_supabase()
    return {
        "status": "ok",
        "nodes": _loaded_nodes,
        "timestamp": datetime.utcnow().isoformat(),
        "default_model": os.getenv("DEFAULT_MODEL", "(not set)"),
        "supabase_connected": sb is not None,
        "connected_robots": len(ROBOT_CONNECTIONS),
        "robots": list(ROBOT_METADATA.keys()),
    }


# ── Practice REST endpoints ───────────────────────────────────────────────────

@app.get("/api/practices")
async def list_practices():
    rows = fetch_automations_list()
    return {"practices": rows, "count": len(rows)}


@app.get("/api/practices/{practice_id}")
async def get_practice(practice_id: str):
    row = fetch_automation(practice_id)
    if not row:
        raise HTTPException(404, f"Practice {practice_id} not found")
    steps = parse_practice_steps(row.get("md_content") or "")
    return {**row, "steps": steps, "steps_count": len(steps)}


@app.post("/api/practices/cache/clear")
async def clear_practice_cache():
    _AUTOMATIONS_CACHE.clear()
    return {"cleared": True}


def _summarize_recording(rec: dict) -> dict:
    """
    Build a compact summary the agent can reason over.
    Extracts joint deltas, gripper changes, duration, and movement events.
    """
    events = rec.get("events", []) or []
    stream = rec.get("stream", []) or []

    # Duration
    started_at = rec.get("started_at")
    stopped_at = rec.get("stopped_at")
    try:
        from datetime import datetime as _dt
        dt_start = _dt.fromisoformat(started_at.replace("Z", "+00:00")) if started_at else None
        dt_stop = _dt.fromisoformat(stopped_at.replace("Z", "+00:00")) if stopped_at else None
        duration_s = round((dt_stop - dt_start).total_seconds(), 2) if (dt_start and dt_stop) else None
    except Exception:
        duration_s = None

    # Joint deltas (first vs last stream sample, >1° counts)
    joint_changes = []
    if stream:
        first = stream[0].get("data", {}) or {}
        last = stream[-1].get("data", {}) or {}
        j0 = first.get("joints_deg") or []
        jN = last.get("joints_deg") or []
        n = min(len(j0), len(jN))
        for i in range(n):
            delta = jN[i] - j0[i]
            if abs(delta) > 1.0:
                joint_changes.append({
                    "joint": i + 1,
                    "from_deg": round(j0[i], 2),
                    "to_deg": round(jN[i], 2),
                    "delta_deg": round(delta, 2),
                })

    # Gripper open/close events (crossing midpoint 400)
    gripper_changes = []
    prev_pos = None
    for s in stream:
        pos = (s.get("data") or {}).get("gripper_position")
        if pos is None:
            continue
        if prev_pos is None:
            prev_pos = pos
            continue
        if prev_pos <= 400 < pos:
            gripper_changes.append({"action": "open", "at": s.get("timestamp")})
        elif prev_pos >= 400 > pos:
            gripper_changes.append({"action": "close", "at": s.get("timestamp")})
        prev_pos = pos

    # TCP summary (first vs last)
    tcp_first = (stream[0].get("data", {}) or {}).get("tcp") if stream else None
    tcp_last = (stream[-1].get("data", {}) or {}).get("tcp") if stream else None

    # Peak effort / temperature
    peak_effort = None
    peak_temp = None
    for s in stream:
        d = s.get("data", {}) or {}
        efforts = d.get("efforts") or []
        temps = d.get("temperatures") or []
        if efforts:
            m = max(abs(e) for e in efforts)
            peak_effort = max(peak_effort, m) if peak_effort is not None else m
        if temps:
            m = max(temps)
            peak_temp = max(peak_temp, m) if peak_temp is not None else m

    # Discrete movement events from MovementObserver
    movement_events = []
    for ev in events:
        d = ev.get("data", {}) or {}
        movement_events.append({
            "kind": d.get("movement_kind"),
            "summary": d.get("movement_summary"),
            "at": ev.get("timestamp"),
            "within_tolerance": (d.get("evaluation") or {}).get("within_tolerance"),
            "position_error_mm": (d.get("evaluation") or {}).get("position_error_mm"),
        })

    return {
        "duration_s": duration_s,
        "device_id": rec.get("device_id", ""),
        "samples_count": len(stream),
        "events_count": len(events),
        "joint_changes": joint_changes,
        "gripper_changes": gripper_changes,
        "tcp_first": tcp_first,
        "tcp_last": tcp_last,
        "peak_effort": round(peak_effort, 4) if peak_effort is not None else None,
        "peak_temperature": round(peak_temp, 1) if peak_temp is not None else None,
        "movement_events": movement_events,
    }


# ── /api/chat ─────────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat_stream(req: ChatRequest):
    if BACKEND_MODE == "cloud":
        async def cloud_generator() -> AsyncGenerator[str, None]:
            async with _get_http_client().stream(
                "POST",
                f"{CLOUD_AGENT_URL}/api/chat",
                json=req.model_dump(exclude_none=True),
                headers=_cloud_headers(),
                timeout=None,
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    yield sse_event("error", {
                        "error": f"cloud agent {resp.status_code}: {body[:200].decode(errors='replace')}"
                    })
                    return
                async for chunk in resp.aiter_text():
                    yield chunk
        return StreamingResponse(
            cloud_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    graph = get_graph()

    session_id = req.session_id or f"test-{uuid.uuid4().hex[:12]}"
    global _active_session
    _active_session = session_id
    config = {"configurable": {"thread_id": session_id}}

    _validate_attachments(req.attachments)
    human_msg = build_human_message(req.message, req.attachments)

    payload: Dict[str, Any] = {
        "messages": [human_msg],
        "user_name": req.user_name or "Test User",
        "user_id": req.user_id or "test-local",
        "interaction_mode": req.interaction_mode or "chat",
        "llm_model": req.llm_model or "",
        "image_attachments": [],
        "_stream_session_id": session_id,
    }
    if req.robot_ids:
        payload["robot_ids"] = req.robot_ids
    if req.equipment_id:
        payload["pending_context"] = {"equipment_id": req.equipment_id}

    # ── Inject connected-device capability cards for any mode that needs them.
    # The bridge sends capability_card on WS registration (see transport.py
    # _get_device_metadata), and we persist it in ROBOT_METADATA. Making it
    # available in state lets tutor_node/automation_worker render accurate
    # device specs (num_joints, joint_limits, actions, etc.) without relying
    # on whatever the practice MD hardcoded.
    if ROBOT_METADATA:
        devices_snapshot = {}
        for rid, meta in ROBOT_METADATA.items():
            devices_snapshot[rid] = {
                "type": meta.get("type", "unknown"),
                "model": meta.get("model", ""),
                "capabilities": meta.get("capabilities", []),
                "capability_card": meta.get("capability_card", {}),
                "num_joints": meta.get("num_joints"),
                "ips": meta.get("ips", []),
            }
        payload["connected_devices"] = devices_snapshot

    # ── Practice mode: hydrate automation_md_content from Supabase ──
    is_practice = (req.interaction_mode or "").lower() == "practice"
    practice_meta: Optional[dict] = None
    if is_practice:
        if not req.automation_id:
            raise HTTPException(400, "Practice mode requires automation_id")
        practice_meta = fetch_automation(req.automation_id)
        if not practice_meta:
            raise HTTPException(404, f"Practice {req.automation_id} not found in lab.automations")

        md_content = req.automation_md_content or practice_meta.get("md_content") or ""

        sess = PRACTICE_SESSIONS.setdefault(session_id, {
            "automation_id": req.automation_id,
            "automation_title": practice_meta.get("title", ""),
            "current_step": req.automation_step or 1,
            "started_at": datetime.utcnow().isoformat() + "Z",
            "steps_completed": [],
        })
        # Handle practice swap mid-session
        if sess["automation_id"] != req.automation_id:
            sess.update({
                "automation_id": req.automation_id,
                "automation_title": practice_meta.get("title", ""),
                "current_step": req.automation_step or 1,
                "started_at": datetime.utcnow().isoformat() + "Z",
                "steps_completed": [],
            })
        # Client-supplied override
        if req.automation_step is not None and req.automation_step != sess["current_step"]:
            sess["current_step"] = req.automation_step

        payload["automation_id"] = req.automation_id
        payload["automation_md_content"] = md_content
        payload["automation_step"] = sess["current_step"]
    else:
        # Non-practice: pass through any client-supplied automation fields as-is
        if req.automation_id:
            payload["automation_id"] = req.automation_id
        if req.automation_md_content:
            payload["automation_md_content"] = req.automation_md_content
        if req.automation_step is not None:
            payload["automation_step"] = req.automation_step

    # ── Attach student_recording to payload if available for this session ──
    # The tutor_node consumes this to compare the student's practice attempt
    # against the step's objective and give feedback.
    if is_practice and session_id in RECORDING_SESSIONS:
        rec = RECORDING_SESSIONS[session_id]
        summary = rec.get("summary")
        # If still active when the user sends a chat message, snapshot now
        if rec.get("active") and not summary:
            summary = _summarize_recording(rec)
        if summary:
            payload["student_recording"] = {
                "summary": summary,
                "started_at": rec.get("started_at"),
                "stopped_at": rec.get("stopped_at"),
                "active": rec.get("active", False),
            }
            logger.info(f"chat: attaching student_recording to payload (session={session_id})")

    async def event_generator() -> AsyncGenerator[str, None]:
        loop = asyncio.get_running_loop()
        event_queue: asyncio.Queue = asyncio.Queue()

        def emit_stream_chunk(chunk_data: dict):
            sse_payload = {"__stream_chunk__": True, **chunk_data}
            loop.call_soon_threadsafe(event_queue.put_nowait, sse_payload)

        register_stream_callback(session_id, emit_stream_chunk)

        try:
            session_event = {"session_id": session_id}
            if is_practice and practice_meta:
                session_event["automation_id"] = req.automation_id
                session_event["current_step"] = PRACTICE_SESSIONS[session_id]["current_step"]
            yield sse_event("session", session_event)
            yield sse_event("thinking", {"node": "start", "message": "Processing…"})

            final_response = None
            all_suggestions: list = []
            chart_payload = None
            interrupted = False
            interrupt_payload = None

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

                if isinstance(event, dict) and event.get("__stream_chunk__"):
                    chunk = {k: v for k, v in event.items() if k != "__stream_chunk__"}
                    yield sse_event("stream_chunk", chunk)
                    continue

                if isinstance(event, dict) and event.get("__error__"):
                    yield sse_event("error", {"message": event["__error__"]})
                    break

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

                # Practice step tracking (only when in practice mode)
                if is_practice:
                    new_step = extract_automation_step(event)
                    sess = PRACTICE_SESSIONS.get(session_id)
                    if new_step is not None and sess and new_step != sess["current_step"]:
                        prev = sess["current_step"]
                        sess["current_step"] = new_step
                        if prev not in sess["steps_completed"] and new_step > prev:
                            sess["steps_completed"].append(prev)
                        yield sse_event("step_update", {
                            "previous_step": prev,
                            "current_step": new_step,
                            "steps_completed": sess["steps_completed"],
                        })

                node_events = extract_events_from_node(event)
                for evt in node_events:
                    yield sse_event("node_update", evt)
                    if evt.get("type") == "narration" and evt.get("content"):
                        yield sse_event("narration", {
                            "content": evt["content"],
                            "source": evt.get("source", ""),
                            "phase": evt.get("phase", "thinking"),
                        })

                for node_name in _ALL_NODES:
                    if node_name in event:
                        yield sse_event("thinking", {"node": node_name, "message": f"Running {node_name}…"})
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

            try:
                final_state = graph.get_state(config)
                if final_state and hasattr(final_state, "values"):
                    tokens = final_state.values.get("token_usage", 0) or 0
                    # Practice: final check for automation_step in case agent wrote it only at end
                    if is_practice:
                        sess = PRACTICE_SESSIONS.get(session_id)
                        final_step = final_state.values.get("automation_step")
                        if sess and final_step is not None and int(final_step) != sess["current_step"]:
                            prev = sess["current_step"]
                            sess["current_step"] = int(final_step)
                            if prev not in sess["steps_completed"] and int(final_step) > prev:
                                sess["steps_completed"].append(prev)
                            yield sse_event("step_update", {
                                "previous_step": prev,
                                "current_step": sess["current_step"],
                                "steps_completed": sess["steps_completed"],
                            })
                    yield sse_event("tokens", {"used": tokens})
            except Exception:
                pass

            for act in ROBOT_ACTION_LOG.pop(session_id, []):
                yield sse_event("robot_action", act)

            yield sse_event("done", {"session_id": session_id})

        except Exception as exc:
            safe_err = str(exc).encode("utf-8", errors="replace").decode("utf-8")
            logger.error(f"stream error: {safe_err}", exc_info=True)
            yield sse_event("error", {"message": safe_err})
            yield sse_event("done", {"session_id": session_id})
        finally:
            unregister_stream_callback(session_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ── /api/confirm ──────────────────────────────────────────────────────────────

@app.post("/api/confirm")
async def confirm_interrupt(req: ConfirmRequest):
    if BACKEND_MODE == "cloud":
        async def cloud_confirm_generator() -> AsyncGenerator[str, None]:
            async with _get_http_client().stream(
                "POST",
                f"{CLOUD_AGENT_URL}/api/confirm",
                json=req.model_dump(exclude_none=True),
                headers=_cloud_headers(),
                timeout=None,
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    yield sse_event("error", {
                        "error": f"cloud agent {resp.status_code}: {body[:200].decode(errors='replace')}"
                    })
                    return
                async for chunk in resp.aiter_text():
                    yield chunk
        return StreamingResponse(
            cloud_confirm_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    graph = get_graph()
    config = {"configurable": {"thread_id": req.session_id}}
    from langgraph.types import Command

    resume_data = {
        "answers": req.answers,
        "completed": req.completed,
        "cancelled": req.cancelled,
    }

    sess = PRACTICE_SESSIONS.get(req.session_id)

    async def event_generator() -> AsyncGenerator[str, None]:
        loop = asyncio.get_running_loop()
        event_queue: asyncio.Queue = asyncio.Queue()

        def emit_stream_chunk(chunk_data: dict):
            sse_payload = {"__stream_chunk__": True, **chunk_data}
            loop.call_soon_threadsafe(event_queue.put_nowait, sse_payload)

        register_stream_callback(req.session_id, emit_stream_chunk)

        try:
            yield sse_event("thinking", {"node": "start", "message": "Resuming…"})

            final_response = None
            all_suggestions: list = []

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

                if isinstance(event, dict) and event.get("__stream_chunk__"):
                    chunk = {k: v for k, v in event.items() if k != "__stream_chunk__"}
                    yield sse_event("stream_chunk", chunk)
                    continue

                if isinstance(event, dict) and event.get("__error__"):
                    yield sse_event("error", {"message": event["__error__"]})
                    break

                # If the session is a practice session, also track step updates here
                if sess:
                    new_step = extract_automation_step(event)
                    if new_step is not None and new_step != sess["current_step"]:
                        prev = sess["current_step"]
                        sess["current_step"] = new_step
                        if prev not in sess["steps_completed"] and new_step > prev:
                            sess["steps_completed"].append(prev)
                        yield sse_event("step_update", {
                            "previous_step": prev,
                            "current_step": new_step,
                            "steps_completed": sess["steps_completed"],
                        })

                for evt in extract_events_from_node(event):
                    yield sse_event("node_update", evt)
                    if evt.get("type") == "narration" and evt.get("content"):
                        yield sse_event("narration", {
                            "content": evt["content"],
                            "source": evt.get("source", ""),
                            "phase": evt.get("phase", "thinking"),
                        })

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
        finally:
            unregister_stream_callback(req.session_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── WebSocket bridge ──────────────────────────────────────────────────────────

@app.post("/api/approve")
async def approve_joint(req: dict):
    """Receive operator approval/rejection for a joint demo step."""
    if BACKEND_MODE == "cloud" and CLOUD_AGENT_URL:
        r = await _get_http_client().post(
            f"{CLOUD_AGENT_URL}/api/approve",
            json=req,
            headers=_cloud_headers(),
            timeout=10.0,
        )
        return JSONResponse(content=r.json(), status_code=r.status_code)

    session_id = req.get("session_id", "")
    approved = req.get("approved", False)
    if session_id:
        try:
            from src.agent.utils.stream_utils import resolve_approval
            resolve_approval(session_id, approved)
        except Exception as e:
            logger.warning(f"resolve_approval failed: {e}")
    return {"ok": True, "session_id": session_id, "approved": approved}


@app.websocket("/ws/robot")
async def ws_robot(ws: WebSocket):
    await ws.accept()

    try:
        init = await asyncio.wait_for(ws.receive_json(), timeout=10)
    except (asyncio.TimeoutError, Exception):
        await ws.close(code=4001, reason="Auth timeout")
        return

    token = init.get("token", "")
    robot_id = init.get("robot_id", "")
    lab_id = init.get("lab_id", "default")          # ← NUEVO
    bridge_id = init.get("bridge_id", "")           # ← NUEVO

    expected_token = _get_lab_token(lab_id)         # ← CAMBIADO
    if token != expected_token or not robot_id:
        await ws.close(code=4003, reason="Invalid token or missing robot_id")
        return

    if robot_id in ROBOT_CONNECTIONS:
        logger.warning(f"[{robot_id}] already registered — evicting stale connection")
        try:
            await ROBOT_CONNECTIONS[robot_id].close()
        except Exception:
            pass
        # Clean up the OLD lab index entry before re-registering
        old_meta = ROBOT_METADATA.get(robot_id, {})
        old_lab = old_meta.get("lab_id")
        if old_lab and old_lab in LAB_DEVICES:
            LAB_DEVICES[old_lab].discard(robot_id)
            if not LAB_DEVICES[old_lab]:
                del LAB_DEVICES[old_lab]
        ROBOT_CONNECTIONS.pop(robot_id, None)
        ROBOT_METADATA.pop(robot_id, None)

    ROBOT_CONNECTIONS[robot_id] = ws
    now_iso = datetime.utcnow().isoformat() + "Z"
    meta = {
        "type": init.get("type", init.get("robot_type", "unknown")),
        "model": init.get("model", ""),
        "lab_id": lab_id,                            # ← NUEVO
        "bridge_id": bridge_id,                      # ← NUEVO
        "protocol": init.get("protocol", "websocket"),
        "capabilities": init.get("capabilities", []),
        "capability_card": init.get("capability_card", {}),
        "ips": init.get("ips", []),
        "num_joints": init.get("num_joints"),
        "last_heartbeat": now_iso,
    }
    ROBOT_METADATA[robot_id] = meta

    # Index by lab — supports multi-lab queries
    LAB_DEVICES.setdefault(lab_id, set()).add(robot_id)

    try:
        from src.agent.shared_state import register_robot
        register_robot(robot_id, ws, meta)
    except ImportError:
        pass

    logger.info(f"bridge connected: {robot_id} (lab={lab_id}, type={meta['type']})")  # ← agregar lab al log
    await ws.send_json({"type": "registered", "robot_id": robot_id, "lab_id": lab_id})  # ← devolver lab_id

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
                logger.info(f"{robot_id} status: {data}")

            # ── Practice recording capture ─────────────────────────────────
            # The bridge emits these two types while MovementObserver is
            # active (reactive_enabled=True). We buffer them per session_id
            # so /api/record/* endpoints can expose them.
            evt_type = data.get("type")
            if evt_type in ("bridge_event", "telemetry_stream"):
                sid = data.get("session_id", "")
                if sid and sid in RECORDING_SESSIONS:
                    rec = RECORDING_SESSIONS[sid]
                    if rec.get("active"):
                        if evt_type == "bridge_event":
                            rec["events"].append(data)
                        else:
                            rec["stream"].append(data)
                continue

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
        pass
    except Exception as e:
        logger.warning(f"bridge error ({robot_id}): {e}")
    finally:
        # Remove from lab index BEFORE deleting metadata
        meta_at_disconnect = ROBOT_METADATA.get(robot_id, {})
        old_lab = meta_at_disconnect.get("lab_id")
        if old_lab and old_lab in LAB_DEVICES:
            LAB_DEVICES[old_lab].discard(robot_id)
            if not LAB_DEVICES[old_lab]:
                del LAB_DEVICES[old_lab]

        ROBOT_CONNECTIONS.pop(robot_id, None)
        ROBOT_METADATA.pop(robot_id, None)
        logger.info(f"bridge disconnected: {robot_id}")
        try:
            from src.agent.shared_state import unregister_robot
            unregister_robot(robot_id)
        except ImportError:
            pass


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/api/robots")
async def list_robots():
    if BACKEND_MODE == "cloud":
        r = await _get_http_client().get(
            f"{CLOUD_AGENT_URL}/api/robots",
            headers=_cloud_headers(),
            timeout=10.0,
        )
        return JSONResponse(content=r.json(), status_code=r.status_code)

    robots = []
    for rid in ROBOT_CONNECTIONS:
        meta = ROBOT_METADATA.get(rid, {})
        robots.append({
            "robot_id": rid,
            "connected": True,
            "type": meta.get("type", "unknown"),
            "model": meta.get("model", ""),
            "lab_id": meta.get("lab_id", "default"),
            "bridge_id": meta.get("bridge_id", ""),
            "capabilities": meta.get("capabilities", []),
            "last_heartbeat": meta.get("last_heartbeat"),
        })
    return {"robots": robots, "count": len(robots)}


@app.get("/api/labs")
async def list_labs():
    """List all labs with at least one connected device."""
    if BACKEND_MODE == "cloud":
        r = await _get_http_client().get(
            f"{CLOUD_AGENT_URL}/api/labs",
            headers=_cloud_headers(),
            timeout=10.0,
        )
        return JSONResponse(content=r.json(), status_code=r.status_code)

    labs = []
    for lab_id, robot_ids in LAB_DEVICES.items():
        device_types = sorted({
            ROBOT_METADATA.get(rid, {}).get("type", "unknown")
            for rid in robot_ids
        })
        labs.append({
            "lab_id": lab_id,
            "device_count": len(robot_ids),
            "device_types": device_types,
        })
    return {"labs": labs, "count": len(labs)}


@app.get("/api/labs/{lab_id}/devices")
async def list_lab_devices(lab_id: str):
    """List devices for a specific lab."""
    if BACKEND_MODE == "cloud":
        r = await _get_http_client().get(
            f"{CLOUD_AGENT_URL}/api/labs/{lab_id}/devices",
            headers=_cloud_headers(),
            timeout=10.0,
        )
        return JSONResponse(content=r.json(), status_code=r.status_code)

    robot_ids = LAB_DEVICES.get(lab_id, set())
    if not robot_ids:
        return {"lab_id": lab_id, "devices": [], "count": 0}

    devices = []
    for rid in robot_ids:
        meta = ROBOT_METADATA.get(rid, {})
        devices.append({
            "device_id": rid,
            "type": meta.get("type"),
            "model": meta.get("model"),
            "bridge_id": meta.get("bridge_id"),
            "ips": meta.get("ips", []),
            "capability_card": meta.get("capability_card", {}),
            "last_heartbeat": meta.get("last_heartbeat"),
            "connected": rid in ROBOT_CONNECTIONS,
        })

    return {
        "lab_id": lab_id,
        "devices": devices,
        "count": len(devices),
    }


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


# ── Practice recording ────────────────────────────────────────────────────────

class RecordStartRequest(BaseModel):
    session_id: str
    device_id: str
    user_id: Optional[str] = "practice-local"


@app.post("/api/record/start")
async def record_start(req: RecordStartRequest):
    """Start a practice recording session. Tells the bridge to go reactive."""
    if req.device_id not in ROBOT_CONNECTIONS:
        raise HTTPException(400, f"Device '{req.device_id}' not connected")

    now = datetime.utcnow().isoformat() + "Z"
    RECORDING_SESSIONS[req.session_id] = {
        "session_id": req.session_id,
        "device_id": req.device_id,
        "started_at": now,
        "stopped_at": None,
        "active": True,
        "events": [],
        "stream": [],
        "summary": None,
    }

    ok = await send_bridge_context_update(
        req.device_id,
        reactive_enabled=True,
        session_id=req.session_id,
        thread_id=req.session_id,
        user_id=req.user_id or "practice-local",
    )
    logger.info(f"record_start: session={req.session_id}, bridge_notified={ok}")
    return {"recording": True, "session_id": req.session_id, "started_at": now}


@app.post("/api/record/stop")
async def record_stop(req: RecordStartRequest):
    """Stop the recording. Bridge goes quiet. Summary is computed."""
    rec = RECORDING_SESSIONS.get(req.session_id)
    if not rec:
        raise HTTPException(404, f"No recording for session {req.session_id}")

    rec["active"] = False
    rec["stopped_at"] = datetime.utcnow().isoformat() + "Z"
    rec["summary"] = _summarize_recording(rec)

    await send_bridge_context_update(
        req.device_id,
        reactive_enabled=False,
        session_id=req.session_id,
        thread_id=req.session_id,
        user_id=req.user_id or "practice-local",
    )
    logger.info(f"record_stop: session={req.session_id}, summary={rec['summary']}")
    return {
        "recording": False,
        "session_id": req.session_id,
        "summary": rec["summary"],
        "events_count": len(rec["events"]),
        "samples_count": len(rec["stream"]),
    }


@app.get("/api/record/download/{session_id}")
async def record_download(session_id: str):
    """Download the raw recording (events + stream + summary) as JSON."""
    rec = RECORDING_SESSIONS.get(session_id)
    if not rec:
        raise HTTPException(404, f"No recording for session {session_id}")
    return JSONResponse(rec)


@app.get("/api/record/download_csv/{session_id}")
async def record_download_csv(session_id: str):
    """Download recording stream as CSV (one row per telemetry sample)."""
    rec = RECORDING_SESSIONS.get(session_id)
    if not rec:
        raise HTTPException(404, f"No recording for session {session_id}")

    import csv
    from fastapi.responses import Response
    from datetime import datetime as _dt

    stream = rec.get("stream", []) or []
    device_id = rec.get("device_id", "")

    # Determine max number of joints across the stream for column count
    max_joints = 0
    for s in stream:
        joints = (s.get("data") or {}).get("joints_deg") or []
        if len(joints) > max_joints:
            max_joints = len(joints)
    max_joints = max_joints or 7

    # Build headers
    headers = ["timestamp", "elapsed_s", "device_id", "state", "mode"]
    headers += [f"j{i+1}_deg" for i in range(max_joints)]
    headers += [f"v{i+1}" for i in range(max_joints)]
    headers += [f"e{i+1}" for i in range(max_joints)]
    headers += [f"t{i+1}_c" for i in range(max_joints)]
    headers += ["tcp_x", "tcp_y", "tcp_z", "tcp_roll", "tcp_pitch", "tcp_yaw"]
    headers += ["gripper_pos"]

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)

    # Compute elapsed_s from first sample
    t0 = None
    if stream:
        try:
            t0 = _dt.fromisoformat(stream[0]["timestamp"].replace("Z", "+00:00"))
        except Exception:
            t0 = None

    def _pad(arr, n):
        arr = list(arr or [])
        return arr + [""] * (n - len(arr))

    for s in stream:
        d = s.get("data") or {}
        ts = s.get("timestamp", "")
        elapsed = ""
        if t0:
            try:
                t = _dt.fromisoformat(ts.replace("Z", "+00:00"))
                elapsed = round((t - t0).total_seconds(), 3)
            except Exception:
                pass
        tcp = d.get("tcp") or {}
        row = [
            ts, elapsed, device_id, d.get("state", ""), d.get("mode", ""),
            *_pad(d.get("joints_deg"), max_joints),
            *_pad(d.get("velocities"), max_joints),
            *_pad(d.get("efforts"), max_joints),
            *_pad(d.get("temperatures"), max_joints),
            tcp.get("x", ""), tcp.get("y", ""), tcp.get("z", ""),
            tcp.get("roll", ""), tcp.get("pitch", ""), tcp.get("yaw", ""),
            d.get("gripper_position", ""),
        ]
        writer.writerow(row)

    csv_bytes = buf.getvalue().encode("utf-8")
    filename = f"recording_{session_id[:12]}.csv"
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/record/download_events_csv/{session_id}")
async def record_download_events_csv(session_id: str):
    """Download discrete movement events (not the full stream) as CSV."""
    rec = RECORDING_SESSIONS.get(session_id)
    if not rec:
        raise HTTPException(404, f"No recording for session {session_id}")

    import csv
    from fastapi.responses import Response

    events = rec.get("events", []) or []
    device_id = rec.get("device_id", "")

    headers = [
        "timestamp", "device_id", "kind", "summary",
        "tcp_x", "tcp_y", "tcp_z", "tcp_roll", "tcp_pitch", "tcp_yaw",
        "state", "joints_deg", "within_tolerance", "position_error_mm",
    ]
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)

    for ev in events:
        d = ev.get("data") or {}
        tcp = d.get("tcp") or {}
        evaluation = d.get("evaluation") or {}
        joints = d.get("joints") or []
        row = [
            ev.get("timestamp", ""),
            device_id,
            d.get("movement_kind", ""),
            d.get("movement_summary", ""),
            tcp.get("x", ""), tcp.get("y", ""), tcp.get("z", ""),
            tcp.get("roll", ""), tcp.get("pitch", ""), tcp.get("yaw", ""),
            d.get("state", ""),
            ";".join(f"{j:.3f}" for j in joints),
            evaluation.get("within_tolerance", ""),
            evaluation.get("position_error_mm", ""),
        ]
        writer.writerow(row)

    csv_bytes = buf.getvalue().encode("utf-8")
    filename = f"recording_events_{session_id[:12]}.csv"
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/record/summary/{session_id}")
async def record_summary(session_id: str):
    """Lightweight summary for the UI (doesn't include raw stream samples)."""
    rec = RECORDING_SESSIONS.get(session_id)
    if not rec:
        raise HTTPException(404, f"No recording for session {session_id}")
    return {
        "session_id": session_id,
        "active": rec.get("active", False),
        "started_at": rec.get("started_at"),
        "stopped_at": rec.get("stopped_at"),
        "events_count": len(rec.get("events", [])),
        "samples_count": len(rec.get("stream", [])),
        "summary": rec.get("summary"),
    }


# ── HTML UI ───────────────────────────────────────────────────────────────────

_HTML_UI = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ORION DevTools</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, system-ui, 'Segoe UI', sans-serif;
  background: #fafafa; color: #111;
  height: 100vh; display: flex; flex-direction: column; overflow: hidden;
}
.top {
  border-bottom: 1px solid #e0e0e0;
  display: flex; align-items: center; padding: 6px 14px; gap: 8px; font-size: 13px;
  flex-shrink: 0; background: #fff; flex-wrap: wrap; min-height: 40px;
}
.top b { font-weight: 600; letter-spacing: -0.01em; }
.top select, .top input {
  font-size: 11px; border: 1px solid #ddd; border-radius: 5px;
  padding: 3px 8px; background: #fff; color: #111; outline: none;
}
.top select.practice-sel { min-width: 220px; display: none; }
.top select.practice-sel.on { display: inline-block; }
.top select#robotSel { min-width: 130px; }
.top .spacer { flex: 1; }
.top button {
  font-size: 11px; border: 1px solid #ddd; border-radius: 5px;
  padding: 3px 12px; background: #fff; cursor: pointer; transition: all 0.15s;
}
.top button:hover { background: #f0f0f0; border-color: #bbb; }
.top button.active { background: #111; color: #fff; border-color: #111; }
.status { font-size: 10px; color: #aaa; font-weight: 500; }
.status.on { color: #16a34a; }
.layout { flex: 1; display: flex; overflow: hidden; }

/* ── Left sidebar — shared shell ── */
.left-sb {
  width: 260px; flex-shrink: 0; border-right: 1px solid #e0e0e0;
  display: flex; flex-direction: column; overflow: hidden; font-size: 12px; background: #fff;
}
.left-sb.hidden { display: none; }
.left-sb-header {
  padding: 10px 12px; border-bottom: 1px solid #e0e0e0;
  display: flex; align-items: center; justify-content: space-between;
}
.left-sb-header span { font-size: 10px; font-weight: 600; color: #999; text-transform: uppercase; letter-spacing: 0.05em; }
.left-sb-header button {
  font-size: 10px; border: 1px solid #e5e5e5; border-radius: 4px;
  padding: 2px 8px; background: #fff; cursor: pointer; color: #999;
}
.left-sb-header button:hover { background: #f5f5f5; color: #111; }

/* Movements panel (visible when mode ≠ practice) */
.moves-panel { display: flex; flex-direction: column; flex: 1; overflow: hidden; }
.moves-panel.hidden { display: none; }
.left-sb-stats { padding: 8px 12px; border-bottom: 1px solid #e0e0e0; display: flex; gap: 14px; font-size: 10px; color: #999; }
.left-sb-stats b { color: #111; font-weight: 600; }
.move-list { flex: 1; overflow-y: auto; }
.move-list::-webkit-scrollbar { width: 3px; }
.move-list::-webkit-scrollbar-thumb { background: #ddd; border-radius: 2px; }
.move-item { padding: 6px 12px; border-bottom: 1px solid #f0f0f0; display: flex; gap: 8px; align-items: flex-start; }
.move-item:hover { background: #f8f8f8; }
.move-num { font-family: monospace; font-size: 10px; color: #ccc; min-width: 18px; padding-top: 1px; }
.move-info { flex: 1; min-width: 0; }
.move-joint { font-weight: 600; font-size: 11px; }
.move-detail { font-size: 10px; color: #999; margin-top: 2px; }
.move-ts { font-family: monospace; font-size: 9px; color: #ccc; white-space: nowrap; padding-top: 1px; }
.empty-state { padding: 32px 12px; text-align: center; color: #ccc; font-size: 11px; }
.left-sb-export { padding: 8px 12px; border-top: 1px solid #e0e0e0; display: flex; gap: 4px; flex-shrink: 0; }
.left-sb-export button {
  flex: 1; padding: 4px; border: 1px solid #e5e5e5; border-radius: 4px;
  background: #fff; font-size: 10px; cursor: pointer; color: #999;
}
.left-sb-export button:hover { background: #f5f5f5; color: #111; }

/* Practice stepper panel (visible when mode = practice) */
.practice-panel { display: none; flex-direction: column; flex: 1; overflow: hidden; }
.practice-panel.on { display: flex; }
.practice-header { padding: 12px 14px; border-bottom: 1px solid #e0e0e0; flex-shrink: 0; }
.practice-title { font-size: 13px; font-weight: 600; color: #111; line-height: 1.3; }
.practice-meta { font-size: 10px; color: #999; margin-top: 4px; display: flex; gap: 10px; flex-wrap: wrap; }
.practice-meta span { text-transform: uppercase; letter-spacing: 0.04em; }
.practice-desc { font-size: 11px; color: #666; margin-top: 6px; line-height: 1.4; }
.progress-bar { height: 3px; background: #f0f0f0; margin: 0; }
.progress-fill { height: 100%; background: #2563eb; transition: width 0.3s; }
.stepper { flex: 1; overflow-y: auto; padding: 10px 0; }
.stepper::-webkit-scrollbar { width: 3px; }
.stepper::-webkit-scrollbar-thumb { background: #ddd; border-radius: 2px; }
.step-item { padding: 10px 14px; border-left: 3px solid transparent; display: flex; gap: 10px; align-items: flex-start; transition: all 0.15s; }
.step-item .step-num {
  width: 20px; height: 20px; border-radius: 50%;
  background: #e5e5e5; color: #888; font-size: 10px; font-weight: 600;
  display: flex; align-items: center; justify-content: center; flex-shrink: 0; margin-top: 1px;
}
.step-item .step-info { flex: 1; min-width: 0; }
.step-item .step-title { font-size: 12px; font-weight: 500; color: #333; line-height: 1.35; }
.step-item .step-summary { font-size: 10px; color: #999; margin-top: 3px; line-height: 1.4; }
.step-item.done { background: #f9fdfa; }
.step-item.done .step-num { background: #16a34a; color: #fff; }
.step-item.done .step-title { color: #555; }
.step-item.current { background: #eff6ff; border-left-color: #2563eb; }
.step-item.current .step-num { background: #2563eb; color: #fff; }
.step-item.current .step-title { color: #111; font-weight: 600; }
.step-empty { padding: 40px 14px; text-align: center; color: #ccc; font-size: 11px; }

/* ── Chat ── */
.chat { flex: 1; display: flex; flex-direction: column; min-width: 0; background: #fafafa; }
.messages { flex: 1; overflow-y: auto; padding: 24px 0; display: flex; flex-direction: column; gap: 16px; align-items: center; }
.messages::-webkit-scrollbar { width: 4px; }
.messages::-webkit-scrollbar-thumb { background: #ddd; border-radius: 2px; }
.msg-wrap { width: 100%; max-width: 680px; padding: 0 24px; }
.msg-u {
  margin-left: auto; background: #111; color: #fff;
  padding: 10px 16px; border-radius: 16px 16px 4px 16px; font-size: 13px;
  max-width: 80%; width: fit-content; white-space: pre-wrap; word-break: break-word; line-height: 1.5;
}
.msg-a-wrap { max-width: 100%; }
.msg-a { font-size: 13px; line-height: 1.7; white-space: pre-wrap; word-break: break-word; color: #222; }
.msg-a-label { font-size: 9px; font-weight: 600; color: #bbb; letter-spacing: 0.08em; margin-bottom: 4px; text-transform: uppercase; }
.msg-err { font-size: 12px; color: #dc2626; font-family: monospace; background: #fef2f2; padding: 8px 12px; border-radius: 8px; border: 1px solid #fecaca; }
.msg-narr { font-size: 11px; color: #999; font-style: italic; border-left: 2px solid #e0e0e0; padding-left: 10px; }
.msg-step { font-size: 11px; color: #2563eb; font-weight: 600; background: #eff6ff; padding: 6px 12px; border-radius: 8px; border: 1px solid #dbeafe; display: inline-block; }
.msg-stream { font-size: 12px; color: #555; background: #f8f9fa; padding: 10px 14px; border-radius: 10px; border: 1px solid #e8e8e8; line-height: 1.6; word-break: break-word; min-width: 260px; }
.msg-stream .stream-label { font-size: 9px; font-weight: 600; color: #bbb; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 8px; }
.stream-joint-row { display: flex; align-items: baseline; gap: 8px; padding: 4px 0; border-bottom: 1px solid #f0f0f0; font-size: 12px; }
.stream-joint-row:last-child { border-bottom: none; }
.stream-joint-name { font-weight: 600; color: #333; min-width: 140px; }
.stream-joint-status { font-size: 11px; color: #16a34a; white-space: nowrap; }
.stream-joint-status.pending { color: #f59e0b; }
.stream-joint-status.error { color: #dc2626; }
.thinking { display: none; align-items: center; gap: 8px; padding: 4px 24px 8px; font-size: 12px; color: #999; justify-content: center; }
.thinking.on { display: flex; }
.dot-pulse { width: 5px; height: 5px; background: #bbb; border-radius: 50%; animation: pulse 1s infinite; }
@keyframes pulse { 0%,100% { opacity:.2; } 50% { opacity:1; } }
.sugs { display: flex; flex-wrap: wrap; gap: 6px; padding: 0 0 8px; justify-content: center; }
.sug { padding: 5px 14px; border: 1px solid #e0e0e0; border-radius: 16px; font-size: 11px; color: #777; cursor: pointer; background: #fff; transition: all 0.15s; }
.sug:hover { border-color: #999; color: #111; background: #f8f8f8; }
.input-area { padding: 8px 24px 16px; display: flex; justify-content: center; }
.input-box { display: flex; border: 1px solid #ddd; border-radius: 12px; background: #fff; width: 100%; max-width: 680px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
.input-box:focus-within { border-color: #bbb; }
.input-box textarea { flex: 1; border: none; padding: 10px 14px; font-size: 13px; font-family: inherit; background: transparent; color: #111; resize: none; outline: none; min-height: 40px; max-height: 120px; line-height: 1.5; }
.input-box textarea::placeholder { color: #ccc; }
.input-box textarea:disabled { background: #f8f8f8; cursor: not-allowed; }
.input-box button { padding: 0 12px; border: none; background: transparent; cursor: pointer; font-size: 15px; color: #ccc; }
.input-box button:hover { color: #888; }
.input-box button.active { color: #111; }
#fileInput { display: none; }
.attach-row { display: flex; flex-wrap: wrap; gap: 4px; padding: 0 24px 4px; justify-content: center; }
.att { font-size: 10px; color: #777; background: #f0f0f0; padding: 3px 10px; border-radius: 10px; }
.att span { color: #dc2626; cursor: pointer; margin-left: 4px; font-weight: 700; }
.hitl { display: none; padding: 14px 24px; border-top: 2px solid #f59e0b; background: #fffbeb; }
.hitl.on { display: block; }
.hitl h3 { font-size: 13px; margin-bottom: 10px; font-weight: 600; }
.hitl label { display: block; font-size: 11px; color: #666; margin: 8px 0 3px; }
.hitl input, .hitl select { width: 100%; padding: 7px 10px; border: 1px solid #ddd; border-radius: 6px; font-size: 12px; outline: none; }
.hitl .btns { margin-top: 12px; display: flex; gap: 6px; }
.hitl .btns button { padding: 6px 16px; border: 1px solid #ddd; border-radius: 6px; font-size: 12px; cursor: pointer; background: #fff; }
.hitl .btns button.p { background: #111; color: #fff; border-color: #111; }
.approval-card { display: none; margin: 0 24px 12px; max-width: 680px; background: #fff; border: 1.5px solid #2563eb; border-radius: 12px; padding: 16px 20px; box-shadow: 0 2px 8px rgba(37,99,235,0.08); }
.approval-card.on { display: block; }
.approval-header { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.approval-badge { background: #2563eb; color: #fff; font-size: 10px; font-weight: 600; padding: 2px 8px; border-radius: 10px; letter-spacing: 0.05em; text-transform: uppercase; white-space: nowrap; }
.approval-title { font-size: 14px; font-weight: 600; color: #111; }
.approval-desc { font-size: 12px; color: #666; margin-bottom: 14px; line-height: 1.5; }
.approval-progress { font-size: 11px; color: #999; margin-bottom: 12px; }
.approval-btns { display: flex; gap: 8px; }
.approval-btns button { flex: 1; padding: 9px 0; border-radius: 8px; font-size: 13px; font-weight: 500; cursor: pointer; border: 1.5px solid; transition: all 0.15s; }
.btn-yes { background: #111; color: #fff; border-color: #111; }
.btn-yes:hover { background: #333; }
.btn-no { background: #fff; color: #666; border-color: #ddd; }
.btn-no:hover { background: #f5f5f5; border-color: #bbb; color: #333; }

/* ── Right sidebar (trace) ── */
.sidebar { width: 320px; flex-shrink: 0; border-left: 1px solid #e0e0e0; display: flex; flex-direction: column; overflow: hidden; font-size: 12px; background: #fff; }
.sidebar.hidden { display: none; }
.sb-section { border-bottom: 1px solid #e0e0e0; padding: 12px 14px; flex-shrink: 0; }
.sb-title { font-size: 10px; font-weight: 600; color: #999; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.05em; }
.sb-row { display: flex; justify-content: space-between; padding: 3px 0; }
.sb-row .k { color: #999; font-size: 11px; }
.sb-row .v { font-weight: 500; font-family: monospace; font-size: 11px; color: #333; }
.sb-tabs { display: flex; border-bottom: 1px solid #e0e0e0; flex-shrink: 0; }
.sb-tab { flex: 1; padding: 8px 0; text-align: center; font-size: 11px; font-weight: 500; color: #bbb; cursor: pointer; border-bottom: 2px solid transparent; }
.sb-tab:hover { color: #888; }
.sb-tab.on { color: #111; border-bottom-color: #111; }
.sb-panel { display: none; flex: 1; flex-direction: column; overflow: hidden; }
.sb-panel.on { display: flex; }
.sb-trace { flex: 1; overflow-y: auto; padding: 12px 14px; }
.trace-run { margin-bottom: 16px; }
.trace-run-header { font-weight: 600; font-size: 12px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: baseline; padding-bottom: 6px; border-bottom: 1px solid #f0f0f0; }
.trace-run-header .ms { font-weight: 400; color: #999; font-family: monospace; font-size: 11px; }
.trace-node { padding: 6px 0; border-bottom: 1px solid #f5f5f5; }
.trace-node:last-child { border-bottom: none; }
.tn-top { display: flex; justify-content: space-between; align-items: center; }
.tn-name { font-weight: 500; font-size: 11px; color: #333; }
.tn-ms { font-family: monospace; font-size: 10px; color: #999; }
.tn-detail { font-size: 10px; color: #aaa; line-height: 1.4; margin-top: 2px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.tn-detail:hover { white-space: normal; }
.tn-latbar { height: 3px; background: #f0f0f0; border-radius: 2px; margin-top: 4px; overflow: hidden; }
.tn-latfill { height: 100%; border-radius: 2px; min-width: 6px; }
.sb-raw { flex: 1; overflow-y: auto; padding: 8px 14px; font-family: monospace; font-size: 10px; }
.sb-raw::-webkit-scrollbar { width: 3px; }
.sb-raw::-webkit-scrollbar-thumb { background: #ddd; border-radius: 2px; }
.raw-line { padding: 3px 0; border-bottom: 1px solid #f8f8f8; color: #999; display: flex; gap: 8px; }
.raw-line .rt { color: #ccc; white-space: nowrap; }
.raw-line .re { font-weight: 600; min-width: 80px; }
.raw-line .re.thinking { color: #f59e0b; }
.raw-line .re.response { color: #16a34a; }
.raw-line .re.error { color: #dc2626; }
.raw-line .re.stream_chunk { color: #7c3aed; }
.raw-line .re.narration { color: #0891b2; }
.raw-line .re.step_update { color: #2563eb; }
.raw-line .rd { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
.raw-line:hover .rd { white-space: normal; word-break: break-all; }
.sb-export { padding: 8px 14px; border-top: 1px solid #e0e0e0; display: flex; gap: 4px; flex-shrink: 0; }
.sb-export button { flex: 1; padding: 5px; border: 1px solid #e5e5e5; border-radius: 5px; background: #fff; font-size: 10px; cursor: pointer; color: #999; }
.sb-export button:hover { background: #f5f5f5; color: #111; }
</style>
</head>
<body>

<div class="top">
  <b>ORION DevTools</b>
  <span class="status" id="bridgeStatus">checking...</span>
  <div class="spacer"></div>
  <select id="cfgModel">
    <option value="">default model</option>
    <option value="claude-sonnet-4-5-20250929">sonnet-4.5</option>
    <option value="claude-sonnet-4-20250514">sonnet-4</option>
    <option value="claude-haiku-4-5-20251001">haiku-4.5</option>
    <option value="gpt-4o">gpt-4o</option>
  </select>
  <select id="cfgMode" onchange="onModeChange()">
    <option value="automation">automation</option>
    <option value="chat">chat</option>
    <option value="practice">practice</option>
    <option value="troubleshoot">troubleshoot</option>
  </select>
  <select class="practice-sel" id="practiceSel" onchange="onPracticeChange()">
    <option value="">— elige práctica —</option>
  </select>
  <select class="practice-sel" id="robotSel" onchange="onRobotChange()">
    <option value="">— sin robot —</option>
  </select>
  <input id="cfgEquipment" placeholder="equipment_id" style="width:90px">
  <button id="btnBackend" onclick="toggleBackend()" title="">🟢 Local</button>
  <button onclick="clearAll()">clear</button>
  <button id="btnLeft" onclick="toggleLeft()">&#9654; panel</button>
  <button id="btnRight" onclick="toggleSidebar()">trace &#9664;</button>
</div>

<div class="layout">

  <div class="left-sb" id="leftSb">
    <!-- Movements panel (non-practice modes) -->
    <div class="moves-panel" id="movesPanel">
      <div class="left-sb-header">
        <span>Robot Data</span>
        <div style="display:flex;gap:4px">
          <button id="autoRecordBtn" onclick="togglePracticeRecord()" title="Record rich telemetry (bridge stream)">● rec</button>
          <button onclick="downloadRecordJSON()" title="Download rich recording (JSON)">↓ json</button>
          <button onclick="downloadRecordCSV()" title="Download rich recording (CSV)">↓ csv</button>
          <button onclick="toggleRecord()" id="recBtn" title="Cheap per-frame telemetry capture">telem</button>
          <button onclick="clearMoves()">clear</button>
        </div>
      </div>
      <div id="autoRecordStatus" style="padding:6px 12px;font-family:monospace;font-size:10px;border-bottom:1px solid #e0e0e0;color:#999;">not recording</div>
      <div id="telemLive" style="padding:6px 12px;font-family:monospace;font-size:10px;border-bottom:1px solid #e0e0e0;color:#999;">waiting for telemetry...</div>
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

    <!-- Practice stepper panel (practice mode) -->
    <div class="practice-panel" id="practicePanel">
      <div class="left-sb-header">
        <span>Práctica</span>
        <div style="display:flex;gap:4px">
          <button id="practiceRecordBtn" onclick="togglePracticeRecord()" title="Start/stop manual recording">● rec</button>
          <button onclick="sendRecordToAgent()" title="Send recording to agent">send</button>
          <button onclick="downloadRecord()" title="Download JSON">↓</button>
          <button onclick="resetPracticeSession()">reset</button>
        </div>
      </div>
      <div id="recordStatus" style="padding:6px 12px;font-family:monospace;font-size:10px;border-bottom:1px solid #e0e0e0;color:#999;">not recording</div>
      <div class="practice-header" id="practiceHeader">
        <div class="practice-title" id="practiceTitle">Selecciona una práctica</div>
        <div class="practice-meta" id="practiceMeta"></div>
        <div class="practice-desc" id="practiceDesc"></div>
      </div>
      <div class="progress-bar"><div class="progress-fill" id="progressFill" style="width:0%"></div></div>
      <div class="stepper" id="stepper">
        <div class="step-empty">Elige una práctica para ver los pasos</div>
      </div>
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
    <div class="approval-card" id="approvalCard">
      <div class="approval-header">
        <span class="approval-badge">Autorización requerida</span>
        <span class="approval-title" id="approvalTitle">J1 — Base</span>
      </div>
      <div class="approval-desc" id="approvalDesc">¿Autoriza mover este joint?</div>
      <div class="approval-progress" id="approvalProgress"></div>
      <div class="approval-btns">
        <button class="btn-yes" onclick="answerApproval(true)">Sí, mover</button>
        <button class="btn-no" onclick="answerApproval(false)">No, saltar</button>
      </div>
    </div>
    <div class="attach-row" id="attRow"></div>
    <div class="input-area">
      <div class="input-box">
        <textarea id="input" rows="1" placeholder="message..."
          onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send()}"
          oninput="autoGrow(this)"></textarea>
        <button onclick="document.getElementById('fileInput').click()">+</button>
        <button id="sendBtn" onclick="send()">&#x2191;</button>
        <input type="file" id="fileInput" multiple onchange="addFiles(this.files)">
      </div>
    </div>
  </div>

  <div class="sidebar" id="sidebar">
    <div class="sb-section">
      <div class="sb-title">Session</div>
      <div class="sb-row"><span class="k">id</span><span class="v" id="sId">-</span></div>
      <div class="sb-row"><span class="k">mode</span><span class="v" id="sMode">automation</span></div>
      <div class="sb-row"><span class="k">practice</span><span class="v" id="sPract">-</span></div>
      <div class="sb-row"><span class="k">step</span><span class="v" id="sStep">-</span></div>
      <div class="sb-row"><span class="k">messages</span><span class="v" id="sMsgs">0</span></div>
      <div class="sb-row"><span class="k">errors</span><span class="v" id="sErrs">0</span></div>
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
var msgCount = 0, errCount = 0;
var runs = [], cur = null;
var movements = [];
var backendMode = 'local';

async function loadBackendMode() {
  try {
    var r = await fetch(BASE + '/api/backend-mode');
    var d = await r.json();
    backendMode = d.mode;
    updateBackendBtn(d);
  } catch(e) {}
}

function updateBackendBtn(d) {
  var btn = $('btnBackend');
  if (!btn) return;
  btn.textContent = backendMode === 'cloud' ? '☁️ Cloud' : '🟢 Local';
  btn.title = d.cloud_url || 'CLOUD_AGENT_URL not set';
}

async function toggleBackend() {
  var newMode = backendMode === 'local' ? 'cloud' : 'local';
  try {
    var r = await fetch(BASE + '/api/backend-mode', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({mode: newMode}),
    });
    var d = await r.json();
    if (!r.ok) {
      alert('Cannot switch to ' + newMode + ': ' + (d.detail || JSON.stringify(d)));
      return;
    }
    backendMode = d.mode;
    sessionId = '';
    $('sId').textContent = '-';
    updateBackendBtn(d);
  } catch(e) {
    alert('Backend toggle failed: ' + e.message);
  }
}

// Practice state
var practicesList = [];
var currentPractice = null;
var currentStep = 1;
var stepsCompleted = [];
var practicesLoaded = false;
var selectedRobotId = '';
var availableRobots = [];

// ── Live stream bubble ─────────────────────────────────────────────────────
var _liveWrap = null, _liveList = null, _currentJointRow = null;
function _ensureLiveBubble() {
  if (_liveWrap) return;
  _liveWrap = document.createElement('div');
  _liveWrap.className = 'msg-wrap';
  _liveWrap.innerHTML =
    '<div class="msg-stream">' +
      '<div class="stream-label">ORION — live</div>' +
      '<div id="_liveList"></div>' +
    '</div>';
  $('msgs').appendChild(_liveWrap);
  _liveList = _liveWrap.querySelector('#_liveList');
  scrollDown();
}
function _removeLiveBubble() {
  if (_liveWrap && _liveWrap.parentNode) _liveWrap.parentNode.removeChild(_liveWrap);
  _liveWrap = null; _liveList = null; _currentJointRow = null; hideApprovalCard();
}
function handleStreamChunk(chunk) {
  var type = chunk.type || '';
  var content = chunk.content || '';
  if (type === 'partial' || type === 'thinking') {
    _ensureLiveBubble();
    if (!content) return;
    var row = document.createElement('div');
    row.className = 'stream-joint-row';
    row.innerHTML = '<span class="stream-joint-name">' + esc(content) + '</span><span class="stream-joint-status pending">⟳</span>';
    _liveList.appendChild(row);
    _currentJointRow = row;
    scrollDown();
  } else if (type === 'tool_status' && chunk.status === 'completed') {
    if (_currentJointRow) {
      var st = _currentJointRow.querySelector('.stream-joint-status');
      if (st) {
        var clean = content.replace(/xarm_move_joint:\s*/i, '').replace(/\{[^}]*\}/g, '').trim();
        if (content.indexOf('⚠') !== -1 || content.indexOf('outside') !== -1 || content.indexOf('error') !== -1) {
          st.className = 'stream-joint-status error';
          st.textContent = '⚠ ' + clean;
        } else {
          st.className = 'stream-joint-status';
          st.textContent = clean || '✓';
        }
      }
      _currentJointRow = null;
    }
    scrollDown();
  } else if (type === 'tool_status' && chunk.status === 'executing') {
    if (content.indexOf('xarm_move_joint') !== -1) return;
    _ensureLiveBubble();
    var row2 = document.createElement('div');
    row2.className = 'stream-joint-row';
    row2.innerHTML = '<span class="stream-joint-name">' + esc(content) + '</span><span class="stream-joint-status pending">⟳</span>';
    _liveList.appendChild(row2);
    _currentJointRow = row2;
    scrollDown();
  }
}

// ── Mode switching ─────────────────────────────────────────────────────────
function currentMode() { return $('cfgMode').value; }

function onModeChange() {
  var mode = currentMode();
  $('sMode').textContent = mode;
  var isPractice = mode === 'practice';
  // Robot selector is useful in practice AND automation (recording picks device).
  var showRobotSel = isPractice || mode === 'automation';
  $('practiceSel').classList.toggle('on', isPractice);
  $('robotSel').classList.toggle('on', showRobotSel);
  $('movesPanel').classList.toggle('hidden', isPractice);
  $('practicePanel').classList.toggle('on', isPractice);
  if (isPractice && !practicesLoaded) {
    loadPracticesList();
  }
  updateSendEnabled();
}

function updateSendEnabled() {
  var mode = currentMode();
  if (mode === 'practice' && !currentPractice) {
    $('input').disabled = true;
    $('input').placeholder = 'Elige una práctica para empezar...';
  } else {
    $('input').disabled = false;
    $('input').placeholder = 'message...';
  }
}

// ── Practice loading ───────────────────────────────────────────────────────
async function loadPracticesList() {
  try {
    var r = await fetch(BASE + '/api/practices');
    var d = await r.json();
    practicesList = d.practices || [];
    var sel = $('practiceSel');
    sel.innerHTML = '<option value="">— elige práctica —</option>';
    practicesList.forEach(function(p) {
      var o = document.createElement('option');
      o.value = p.id;
      var tag = p.difficulty ? ' [' + p.difficulty + ']' : '';
      o.textContent = p.title + tag;
      sel.appendChild(o);
    });
    practicesLoaded = true;
  } catch(e) {
    addChat('err', 'No se pudo cargar prácticas de Supabase: ' + e.message);
  }
}

async function onPracticeChange() {
  var id = $('practiceSel').value;
  if (!id) {
    currentPractice = null;
    renderStepper(); renderPracticeHeader();
    updateSendEnabled();
    return;
  }
  try {
    var r = await fetch(BASE + '/api/practices/' + id);
    if (!r.ok) throw new Error('HTTP ' + r.status);
    var d = await r.json();
    currentPractice = d;
    currentStep = 1; stepsCompleted = [];
    sessionId = '';  // new practice → new session
    $('sId').textContent = '-';
    renderPracticeHeader(); renderStepper();
    updateSendEnabled();
    $('input').focus();
  } catch(e) {
    addChat('err', 'No se pudo cargar la práctica: ' + e.message);
  }
}

function renderPracticeHeader() {
  if (!currentPractice) {
    $('practiceTitle').textContent = 'Selecciona una práctica';
    $('practiceMeta').innerHTML = '';
    $('practiceDesc').textContent = '';
    $('sPract').textContent = '-';
    return;
  }
  $('practiceTitle').textContent = currentPractice.title || '-';
  var meta = [];
  if (currentPractice.type) meta.push('<span>' + esc(currentPractice.type) + '</span>');
  if (currentPractice.difficulty) meta.push('<span>' + esc(currentPractice.difficulty) + '</span>');
  meta.push('<span>' + (currentPractice.steps_count || 0) + ' pasos</span>');
  $('practiceMeta').innerHTML = meta.join('');
  $('practiceDesc').textContent = currentPractice.description || '';
  $('sPract').textContent = (currentPractice.title || '').substring(0, 22);
}

function renderStepper() {
  var c = $('stepper');
  if (!currentPractice || !currentPractice.steps || !currentPractice.steps.length) {
    c.innerHTML = '<div class="step-empty">Sin pasos — elige una práctica</div>';
    $('progressFill').style.width = '0%';
    $('sStep').textContent = '-';
    return;
  }
  var html = '';
  currentPractice.steps.forEach(function(s) {
    var cls = 'step-item';
    if (stepsCompleted.indexOf(s.index) !== -1) cls += ' done';
    if (s.index === currentStep) cls += ' current';
    html += '<div class="' + cls + '">' +
      '<div class="step-num">' + s.index + '</div>' +
      '<div class="step-info">' +
        '<div class="step-title">' + esc(s.title) + '</div>' +
        (s.summary ? '<div class="step-summary">' + esc(s.summary) + '</div>' : '') +
      '</div></div>';
  });
  c.innerHTML = html;
  var total = currentPractice.steps.length;
  var progress = total > 0 ? ((currentStep - 1) / total) * 100 : 0;
  $('progressFill').style.width = Math.min(progress, 100) + '%';
  $('sStep').textContent = currentStep + ' / ' + total;
}

function handleStepUpdate(prev, next) {
  currentStep = next;
  if (prev && stepsCompleted.indexOf(prev) === -1 && next > prev) {
    stepsCompleted.push(prev);
  }
  renderStepper();
  addChat('step', 'Paso ' + prev + ' → ' + next);
  var items = $('stepper').querySelectorAll('.step-item');
  if (items[next - 1]) items[next - 1].scrollIntoView({behavior:'smooth', block:'center'});
}

function resetPracticeSession() {
  sessionId = '';
  currentStep = 1;
  stepsCompleted = [];
  $('sId').textContent = '-';
  renderStepper();
  // Reset recording state
  stopRecordPoll();
  practiceRecordingActive = false;
  if ($('practiceRecordBtn')) updatePracticeRecordUI();
  if ($('recordStatus')) $('recordStatus').textContent = 'not recording';
}

// ── Practice recording (manual operation by student) ───────────────────────
var practiceRecordingActive = false;
var recordingPollTimer = null;

async function togglePracticeRecord() {
  // Works in both practice and automation modes. Practice gate only for UX
  // safety when in practice mode (avoid recording before practice is picked).
  var mode = currentMode();
  if (mode === 'practice' && !currentPractice) {
    addChat('err', 'Selecciona una práctica primero.');
    return;
  }
  if (!selectedRobotId) { addChat('err', 'Selecciona un robot primero.'); return; }
  // Auto-generate sessionId if none yet (automation "record-then-send" flow).
  if (!sessionId) {
    sessionId = 'rec-' + Math.random().toString(36).substring(2, 14);
    $('sId').textContent = sessionId.substring(0, 20);
  }

  var endpoint = practiceRecordingActive ? '/api/record/stop' : '/api/record/start';
  try {
    var r = await fetch(BASE + endpoint, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        session_id: sessionId,
        device_id: selectedRobotId,
        user_id: 'devtools',
      }),
    });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    var d = await r.json();
    practiceRecordingActive = d.recording;
    updatePracticeRecordUI();
    if (practiceRecordingActive) {
      startRecordPoll();
      addChat('narr', 'Recording started — move the robot manually.');
    } else {
      stopRecordPoll();
      addChat('narr', 'Recording stopped. ' + (d.events_count || 0) + ' events, ' + (d.samples_count || 0) + ' samples.');
    }
  } catch (e) {
    addChat('err', 'Record toggle failed: ' + e.message);
  }
}

function updatePracticeRecordUI() {
  // Update both practice and automation rec buttons since the state is shared.
  var btns = [$('practiceRecordBtn'), $('autoRecordBtn')];
  btns.forEach(function(btn){
    if (!btn) return;
    if (practiceRecordingActive) {
      btn.textContent = '■ stop';
      btn.style.color = '#dc2626';
    } else {
      btn.textContent = '● rec';
      btn.style.color = '';
    }
  });
  // Update status line in whichever panel is visible
  var statusEls = [$('recordStatus'), $('autoRecordStatus')];
  statusEls.forEach(function(el){
    if (!el) return;
    if (!practiceRecordingActive) el.textContent = 'not recording';
  });
}

function startRecordPoll() {
  stopRecordPoll();
  recordingPollTimer = setInterval(pollRecordSummary, 1000);
}

function stopRecordPoll() {
  if (recordingPollTimer) {
    clearInterval(recordingPollTimer);
    recordingPollTimer = null;
  }
}

async function pollRecordSummary() {
  if (!sessionId) return;
  try {
    var r = await fetch(BASE + '/api/record/summary/' + sessionId);
    if (!r.ok) return;
    var d = await r.json();
    var text = (d.active ? '● recording · ' : '○ stopped · ')
      + (d.events_count || 0) + ' events · ' + (d.samples_count || 0) + ' samples';
    if ($('recordStatus')) $('recordStatus').textContent = text;
    if ($('autoRecordStatus')) $('autoRecordStatus').textContent = text;
  } catch (e) {}
}

async function sendRecordToAgent() {
  if (!sessionId) { addChat('err', 'No session.'); return; }
  // If still recording, stop first so we get a clean summary
  if (practiceRecordingActive) {
    await togglePracticeRecord();
  }
  // Send an implicit message — test_server attaches student_recording to payload
  $('input').value = "I'm done. Here's what I did — how did I do?";
  send();
}

async function downloadRecord() {
  // Legacy alias — used by the practice panel "↓" button. Calls JSON version.
  return downloadRecordJSON();
}

async function downloadRecordJSON() {
  if (!sessionId) { addChat('err', 'No session — start a recording first.'); return; }
  try {
    var r = await fetch(BASE + '/api/record/download/' + sessionId);
    if (!r.ok) throw new Error('HTTP ' + r.status);
    var text = await r.text();
    dl('recording_' + sessionId.substring(0, 12) + '.json', text, 'application/json');
  } catch (e) {
    addChat('err', 'Download failed: ' + e.message);
  }
}

async function downloadRecordCSV() {
  if (!sessionId) { addChat('err', 'No session — start a recording first.'); return; }
  try {
    var r = await fetch(BASE + '/api/record/download_csv/' + sessionId);
    if (!r.ok) throw new Error('HTTP ' + r.status);
    var text = await r.text();
    dl('recording_' + sessionId.substring(0, 12) + '.csv', text, 'text/csv');
  } catch (e) {
    addChat('err', 'CSV download failed: ' + e.message);
  }
}

function onRobotChange() {
  selectedRobotId = $('robotSel').value;
}

function refreshRobotDropdown(robots) {
  // robots = [{robot_id, type, ...}, ...]
  availableRobots = robots || [];
  var sel = $('robotSel');
  var prevValue = sel.value;
  // Only rebuild if the set of robot_ids actually changed, to avoid
  // wiping the user's selection on every 3s poll.
  var newIds = availableRobots.map(function(r){return r.robot_id;}).sort().join('|');
  if (sel.dataset.lastIds === newIds) return;
  sel.dataset.lastIds = newIds;

  sel.innerHTML = '<option value="">— sin robot —</option>';
  availableRobots.forEach(function(r) {
    var o = document.createElement('option');
    o.value = r.robot_id;
    var typeTag = r.type ? ' (' + r.type + ')' : '';
    var labTag = r.lab_id && r.lab_id !== 'default' ? ' [' + r.lab_id + ']' : '';
    o.textContent = r.robot_id + typeTag + labTag;
    sel.appendChild(o);
  });
  // Restore previous selection if still valid
  if (prevValue && availableRobots.some(function(r){return r.robot_id === prevValue;})) {
    sel.value = prevValue;
  } else {
    selectedRobotId = '';
  }
}

// ── Movements ──────────────────────────────────────────────────────────────
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
    if(m.command==='move_joint'&&m.from!=null) detail=m.from+' → '+m.to+'°';
    else if(m.command==='move_joint') detail=m.to+'°';
    else if(m.final_angles) detail=m.final_angles.map(function(a,i){return'J'+(i+1)+':'+Math.round(a*10)/10;}).join(' ');
    else if(m.name) detail=m.name;
    else detail=m.command||'';
    var color = m.status==='ok'?'#16a34a':m.status==='error'?'#dc2626':m.status==='noop'?'#999':'#2563eb';
    html += '<div class="move-item"><div class="move-num">'+m.n+'</div><div class="move-info"><div class="move-joint" style="color:'+color+'">'+esc(m.joint+(m.name?' ('+m.name+')':''))+'</div><div class="move-detail">'+esc(detail)+'</div></div><div class="move-ts">'+m.time+'</div></div>';
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

// ── Sidebar toggles ────────────────────────────────────────────────────────
function toggleLeft() { $('leftSb').classList.toggle('hidden'); $('btnLeft').classList.toggle('active', !$('leftSb').classList.contains('hidden')); }
function toggleSidebar() { $('sidebar').classList.toggle('hidden'); $('btnRight').classList.toggle('active', !$('sidebar').classList.contains('hidden')); }
function sbTab(name, el) {
  document.querySelectorAll('.sb-tab').forEach(function(t){t.classList.remove('on');});
  document.querySelectorAll('.sb-panel').forEach(function(p){p.classList.remove('on');});
  el.classList.add('on');
  $('p'+name.charAt(0).toUpperCase()+name.slice(1)).classList.add('on');
}

function scrollDown() { var m=$('msgs'); requestAnimationFrame(function(){m.scrollTop=m.scrollHeight;}); }
function ts() { return new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit'}); }

function addChat(type, text) {
  var wrap = document.createElement('div');
  wrap.className = 'msg-wrap';
  if (type==='user')  wrap.innerHTML='<div class="msg-u">'+esc(text)+'</div>';
  else if (type==='ai') wrap.innerHTML='<div class="msg-a-wrap"><div class="msg-a-label">ORION</div><div class="msg-a">'+esc(text)+'</div></div>';
  else if (type==='err') wrap.innerHTML='<div class="msg-err">'+esc(text)+'</div>';
  else if (type==='narr') wrap.innerHTML='<div class="msg-narr">'+esc(text)+'</div>';
  else if (type==='step') wrap.innerHTML='<div class="msg-step">'+esc(text)+'</div>';
  $('msgs').appendChild(wrap); scrollDown();
}

function setThink(on,msg){$('thinkBar').classList.toggle('on',on);if(msg)$('thinkText').textContent=msg;}
function autoGrow(el){el.style.height='auto';el.style.height=Math.min(el.scrollHeight,120)+'px';$('sendBtn').classList.toggle('active',el.value.trim().length>0);}
function addFiles(files){for(var i=0;i<files.length;i++){(function(f){var r=new FileReader();r.onload=function(){atts.push({name:f.name,type:f.type||'application/octet-stream',data:r.result.split(',')[1]});renderAtts();};r.readAsDataURL(f);})(files[i]);}$('fileInput').value='';}
function renderAtts(){$('attRow').innerHTML=atts.map(function(a,i){return'<span class="att">'+esc(a.name)+'<span onclick="atts.splice('+i+',1);renderAtts()">&times;</span></span>';}).join('');}
function showSugs(list){$('sugs').innerHTML=list.map(function(s){var t=typeof s==='string'?s:s.text||JSON.stringify(s);return'<span class="sug" onclick="$(\'input\').value=this.textContent;send()">'+esc(t)+'</span>';}).join('');}

function startRun(p){cur={id:Date.now(),prompt:p,response:'',nodes:[],totalMs:0,t0:performance.now()};}
function addNode(n,d){if(!cur)return;var now=performance.now();var prev=cur.nodes.length?cur.nodes[cur.nodes.length-1]._t:cur.t0;cur.nodes.push({name:n,detail:d||'',ms:Math.round(now-prev),_t:now,wall:ts()});}
function endRun(r){if(!cur)return;cur.totalMs=Math.round(performance.now()-cur.t0);cur.response=r||'';runs.push(cur);renderTrace();updateStats();cur=null;}

function renderTrace(){
  var c=$('traceContainer'); c.innerHTML='';
  for(var i=runs.length-1;i>=0;i--){
    var r=runs[i];
    var mx=Math.max.apply(null,r.nodes.map(function(n){return n.ms;}));
    if(mx<1)mx=1;
    var h='<div class="trace-run"><div class="trace-run-header"><span>'+esc(r.prompt.substring(0,50))+'</span><span class="ms">'+(r.totalMs/1000).toFixed(1)+'s</span></div>';
    r.nodes.forEach(function(n){
      var p=Math.min((n.ms/mx)*100,100);
      var co=n.ms<2000?'#16a34a':n.ms<8000?'#f59e0b':'#dc2626';
      h+='<div class="trace-node"><div class="tn-top"><span class="tn-name">'+esc(n.name)+'</span><span class="tn-ms">'+n.ms+'ms</span></div>';
      if(n.detail)h+='<div class="tn-detail">'+esc(n.detail)+'</div>';
      h+='<div class="tn-latbar"><div class="tn-latfill" style="width:'+p+'%;background:'+co+'"></div></div></div>';
    });
    h+='</div>';
    c.innerHTML+=h;
  }
}

function updateStats(){$('sMsgs').textContent=msgCount;$('sErrs').textContent=errCount;if(runs.length){var avg=Math.round(runs.reduce(function(s,r){return s+r.totalMs;},0)/runs.length);$('sAvg').textContent=(avg/1000).toFixed(1)+'s';}}

function logRaw(type,data){
  var d=document.createElement('div');
  d.className='raw-line';
  d.innerHTML='<span class="rt">'+ts()+'</span><span class="re '+type+'">'+type+'</span><span class="rd">'+esc(JSON.stringify(data)).substring(0,300)+'</span>';
  $('rawContainer').appendChild(d);
  $('rawContainer').scrollTop=$('rawContainer').scrollHeight;
}

async function send(){
  if(busy)return;
  var mode = currentMode();
  if(mode === 'practice' && !currentPractice){
    addChat('err','Selecciona una práctica primero.');
    return;
  }
  var text=$('input').value.trim();
  if(!text&&!atts.length)return;
  busy=true;
  if(text){addChat('user',text);msgCount++;}
  $('input').value='';$('input').style.height='auto';$('sendBtn').classList.remove('active');
  $('sugs').innerHTML='';
  setThink(true,'thinking...');
  startRun(text);

  var body={
    message:text,user_name:'DevTools',user_id:'devtools',
    interaction_mode: mode,
    llm_model:$('cfgModel').value
  };
  var eq=$('cfgEquipment').value.trim();if(eq)body.equipment_id=eq;
  if(sessionId)body.session_id=sessionId;
  if(atts.length)body.attachments=atts;
  if(mode === 'practice' && currentPractice){
    body.automation_id = currentPractice.id;
  }
  if(selectedRobotId){
    body.robot_ids = [selectedRobotId];
  }
  atts=[];renderAtts();

  var f='';
  try{f=await doStream(BASE+'/api/chat',body);}
  catch(e){addChat('err',e.message);errCount++;}

  _removeLiveBubble();
  endRun(f);
  setThink(false);
  busy=false;
}

async function doStream(url,body){
  var resp=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(!resp.ok){
    var errText=await resp.text().catch(function(){return'';});
    addChat('err','HTTP '+resp.status+(errText?' — '+errText.substring(0,200):''));
    errCount++;return'';
  }
  var reader=resp.body.getReader(),dec=new TextDecoder();
  var buf='',et='',final='';
  while(true){
    var chunk=await reader.read();
    if(chunk.done)break;
    buf+=dec.decode(chunk.value,{stream:true});
    var lines=buf.split('\n');
    buf=lines.pop();
    for(var li=0;li<lines.length;li++){
      var ln=lines[li];
      if(ln.startsWith('event: ')) et=ln.slice(7).trim();
      else if(ln.startsWith('data: ')&&et){
        try{final=handleSSE(et,JSON.parse(ln.slice(6)),final);}catch(e){}
        et='';
      }
    }
  }
  return final;
}

function handleSSE(type,data,final){
  logRaw(type,data);
  switch(type){
    case 'session':
      sessionId=data.session_id||sessionId;
      $('sId').textContent=sessionId.substring(0,20);
      if(data.current_step){currentStep=data.current_step;renderStepper();}
      break;
    case 'thinking':
      setThink(true,data.message||(data.node+'...'));
      if(data.node&&data.node!=='start') addNode(data.node,data.message||'');
      break;
    case 'stream_chunk':
    case 'practice_chunk':
      if (data.type === 'approval_request') showApprovalCard(data);
      else handleStreamChunk(data);
      break;
    case 'narration':
      if(data.content) addChat('narr', data.content);
      if(data.source||data.phase) addNode(data.source||'narration', data.content||'');
      break;
    case 'node_update':
      if(data.type==='narration'&&data.content){
        addChat('narr',data.content);
        addNode(data.source||'narration',data.content);
      } else if(data.content){
        addNode(data.source||data.node||'?',data.content.substring(0,200));
      }
      break;
    case 'step_update':
      handleStepUpdate(data.previous_step, data.current_step);
      break;
    case 'response':
      setThink(false);
      if(data.content){addChat('ai',data.content);final=data.content;}
      break;
    case 'suggestions':
      showSugs(data.suggestions||[]);
      break;
    case 'questions':
      showHITL(data);
      break;
    case 'robot_action':
      var rd=data.data||{};var cmd=data.command||'';
      // Capture full context: what the agent requested (params), what the
      // bridge returned (raw), and a live telemetry snapshot at this instant.
      // Display fields (joint, name, from, to) are set below for the sidebar.
      var mv={n:movements.length+1,time:ts(),timestamp:data.timestamp||new Date().toISOString(),
              device:data.device_id||'',command:cmd,status:'ok',
              params:data.params||{},raw:rd,telemetry_snapshot:_lastTelemSnapshot};
      if(cmd==='move_joint'){mv.joint='J'+(rd.target_joint||'?');mv.name=rd.joint_name||'';mv.from=rd.previous_angle!=null?rd.previous_angle:null;mv.to=rd.target_angle!=null?rd.target_angle:null;mv.final_angles=rd.final_angles||null;}
      else if(cmd==='home'){mv.joint='ALL';mv.name='home';mv.to=0;mv.final_angles=[0,0,0,0,0,0];}
      else if(cmd==='go_to_pose'){mv.joint='ALL';mv.name=rd.pose||'';mv.final_angles=rd.final_angles||null;}
      else if(cmd==='say_hi'){mv.joint='ALL';mv.name='wave';}
      else if(cmd==='get_position'||cmd==='get_full_status'){mv.joint='ALL';mv.name=cmd;mv.status='query';mv.final_angles=rd.joints||null;}
      else{mv.joint='-';mv.name=cmd;}
      movements.push(mv);renderMoves();
      break;
    case 'error':
      setThink(false);
      addChat('err',data.message||'error');
      errCount++;
      addNode('error',data.message||'');
      break;
    case 'done':
      setThink(false);
      break;
  }
  return final;
}

function showHITL(data){
  setThink(false);hitlSid=data.session_id||sessionId;
  $('hitlTitle').textContent=data.title||'Input required';
  var c=$('hitlQs');c.innerHTML='';
  var qs=data.questions||[];
  if(!qs.length&&data.prompt){addChat('ai',data.prompt);return;}
  qs.forEach(function(q,i){
    var d=document.createElement('div');
    d.innerHTML='<label>'+esc(q.question||q.label||'Q'+(i+1))+'</label>';
    if(q.options&&q.options.length){
      var sel=document.createElement('select');sel.dataset.key=q.key||('q'+i);
      q.options.forEach(function(o){var opt=document.createElement('option');opt.value=typeof o==='string'?o:o.value;opt.textContent=typeof o==='string'?o:o.label;sel.appendChild(opt);});
      d.appendChild(sel);
    } else {
      var inp=document.createElement('input');inp.dataset.key=q.key||('q'+i);inp.placeholder=q.placeholder||'';d.appendChild(inp);
    }
    c.appendChild(d);
  });
  $('hitl').classList.add('on');
}

async function submitHITL(){
  var ans={};
  $('hitlQs').querySelectorAll('input,select').forEach(function(el){ans[el.dataset.key]=el.value;});
  $('hitl').classList.remove('on');setThink(true,'submitting...');busy=true;
  try{await doStream(BASE+'/api/confirm',{session_id:hitlSid,answers:ans,completed:true,cancelled:false});}
  catch(e){addChat('err',e.message);}
  _removeLiveBubble();setThink(false);busy=false;
}
function cancelHITL(){$('hitl').classList.remove('on');}

// ── Approval card ──────────────────────────────────────────────────────────
var _approvalSessionId = '';
function showApprovalCard(data) {
  _approvalSessionId = sessionId;
  $('approvalTitle').textContent = 'J' + data.joint_id + ' — ' + (data.joint_name || '?');
  $('approvalDesc').textContent = (data.joint_desc || '') + (data.demo_range ? ' (rango: ±' + data.demo_range + '°)' : '');
  $('approvalProgress').textContent = data.joint_number ? 'Joint ' + data.joint_number + ' de ' + data.total_joints : '';
  $('approvalCard').classList.add('on');
  $('approvalCard').scrollIntoView({behavior:'smooth',block:'nearest'});
}
function hideApprovalCard() { $('approvalCard').classList.remove('on'); _approvalSessionId = ''; }

async function answerApproval(approved) {
  hideApprovalCard();
  if (!_approvalSessionId && !sessionId) return;
  try {
    await fetch(BASE + '/api/approve', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({session_id: sessionId, approved: approved}),
    });
  } catch(e) { console.error('answerApproval failed:', e); }
}

function exportJSON(){
  var d={runs:runs.map(function(r){return{prompt:r.prompt,response:r.response,totalMs:r.totalMs,nodes:r.nodes.map(function(n){return{name:n.name,ms:n.ms,detail:n.detail,wall:n.wall};})};})};
  dl('trace.json',JSON.stringify(d,null,2),'application/json');
}
function exportCSV(){
  var csv='run,prompt,node,ms,detail\n';
  runs.forEach(function(r,i){r.nodes.forEach(function(n){csv+=(i+1)+',"'+r.prompt.replace(/"/g,'""')+'","'+n.name+'",'+n.ms+',"'+(n.detail||'').replace(/"/g,'""')+'"\n';});});
  dl('trace.csv',csv,'text/csv');
}
function dl(name,content,mime){var a=document.createElement('a');a.href=URL.createObjectURL(new Blob([content],{type:mime}));a.download=name;a.click();}
function copyLast(){
  var r=runs[runs.length-1];if(!r)return;
  navigator.clipboard.writeText(r.nodes.map(function(n){return'['+n.wall+'] '+n.name+' ('+n.ms+'ms) '+(n.detail||'');}).join('\n'));
}

async function pollBridge(){
  try{
    var r=await fetch(BASE+'/api/robots',{signal:AbortSignal.timeout(3000)});
    var d=await r.json();var n=d.count||0;
    $('bridgeStatus').textContent=n>0?n+' device'+(n>1?'s':'')+' connected':'no bridge';
    $('bridgeStatus').className='status'+(n>0?' on':'');
    $('sBridge').textContent=n>0?d.robots.map(function(r){return r.robot_id;}).join(', '):'none';
    refreshRobotDropdown(d.robots || []);
  }catch(e){$('bridgeStatus').textContent='offline';$('bridgeStatus').className='status';refreshRobotDropdown([]);}
}
setInterval(pollBridge,3000);pollBridge();

function clearAll(){
  $('msgs').innerHTML='';$('rawContainer').innerHTML='';$('traceContainer').innerHTML='';
  $('sugs').innerHTML='';runs=[];cur=null;msgCount=0;errCount=0;sessionId='';
  $('sId').textContent='-';updateStats();clearMoves();
  currentStep=1;stepsCompleted=[];renderStepper();
  _liveWrap=null;_liveList=null;
  // Reset practice recording state
  stopRecordPoll();
  practiceRecordingActive=false;
  if($('practiceRecordBtn'))updatePracticeRecordUI();
  if($('recordStatus'))$('recordStatus').textContent='not recording';
}

var telemData=[];var telemRecording=false;var _lastTelemSnapshot=null;
function pollTelemetry(){
  fetch(BASE+'/api/telemetry/latest',{signal:AbortSignal.timeout(2000)}).then(function(r){return r.json();}).then(function(d){
    var keys=Object.keys(d);if(!keys.length)return;
    var dev=keys[0];var t=d[dev];var dd=t.data||{};
    // Store a deep snapshot so robot_action can attach live state at call time.
    _lastTelemSnapshot={device:dev,at:new Date().toISOString(),data:dd};
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

// ── Init ───────────────────────────────────────────────────────────────────
$('input').focus();
onModeChange();  // set initial panel visibility
loadBackendMode();
if (window.innerWidth <= 700) {
  $('leftSb').classList.add('hidden');
  $('sidebar').classList.add('hidden');
} else {
  $('btnLeft').classList.add('active');
  $('btnRight').classList.add('active');
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return _HTML_UI


if __name__ == "__main__":
    init_services()
    _build_graph()
    sb = get_supabase()
    sb_status = "connected" if sb else "NOT CONNECTED — practice mode disabled"
    print(f"\n  ORION DevTools")
    print(f"  ─────────────")
    print(f"  agent:        {ORION_ROOT}")
    print(f"  model:        {os.getenv('DEFAULT_MODEL', '(not set)')}")
    print(f"  supabase:     {sb_status}")
    print(f"  http://localhost:8000\n", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
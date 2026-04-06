# orion-devtools

Local dev server for testing the ORION agent graph without Cloud Run, Supabase, or any external infra. Includes a built-in chat UI with trace visualization, a WebSocket bridge endpoint for connecting robots/PLCs, and telemetry recording.

## Folder structure

This repo **must** sit next to the main `Orion` folder — it imports the agent graph, workers, and shared state directly from there:

```
FINAL_PRODUCTS/
├── Orion/              ← main agent repo (graph, workers, prompts, .env)
│   ├── src/
│   │   └── agent/
│   │       ├── graph.py
│   │       ├── shared_state.py
│   │       └── utils/
│   ├── .env            ← ANTHROPIC_API_KEY, DEFAULT_MODEL, etc.
│   ├── .venv/          ← shared virtual environment
│   └── requirements.txt
│
└── orion-devtools/     ← this repo
    ├── test_server.py
    ├── bridge.py
    ├── lab_config_dev.json
    ├── requirements.txt
    └── README.md
```

If the folders aren't siblings, the imports will fail. `test_server.py` resolves the path with `../Orion` at startup.

## Setup

### 1. Create a virtual environment (or use Orion's)

You can either create a dedicated venv or reuse the one from `Orion/`. Using Orion's is easier since it already has `langgraph`, `langchain`, etc.

**Option A — reuse Orion's venv:**

```powershell
cd orion-devtools
..\Orion\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Option B — fresh venv:**

```powershell
cd orion-devtools
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r ..\Orion\requirements.txt   # need the agent deps too
```

### 2. Check your .env

The server loads `.env` from `../Orion/.env`. Make sure it has at least:

```
ANTHROPIC_API_KEY=sk-ant-...
DEFAULT_MODEL=claude-sonnet-4-20250514
```

### 3. Run the server

```powershell
python test_server.py
```

Opens at **http://localhost:8000** — the chat UI loads inline, no frontend build step needed.

## Connecting the bridge

In a second terminal (same venv activated):

```powershell
python bridge.py --config lab_config_dev.json
```

The bridge connects to `ws://localhost:8000/ws/robot` and registers whatever devices are defined in your config (xArm, PLC, etc.). The DevTools UI shows connection status in the top bar.

## What's in the UI

- **Chat** — sends messages to the agent graph via SSE streaming (`POST /api/chat`)
- **Left sidebar** — robot movement log + telemetry recording (record/export as CSV)
- **Right sidebar** — execution trace (which nodes ran, latency per node) + raw SSE event log
- **HITL** — if the graph interrupts for human input, a form appears inline
- **Top bar** — switch interaction mode, model, equipment ID

## API endpoints

| Endpoint | Method | What it does |
|---|---|---|
| `/` | GET | Serves the inline HTML UI |
| `/health` | GET | Status check, lists loaded nodes and connected devices |
| `/api/chat` | POST | SSE stream — send a message, get back agent response |
| `/api/confirm` | POST | SSE stream — resume after a HITL interrupt |
| `/api/robots` | GET | List connected bridge devices |
| `/api/telemetry/latest` | GET | Last telemetry frame per device |
| `/api/telemetry/record` | POST | Toggle telemetry recording on/off |
| `/api/telemetry/export` | GET | Dump recorded telemetry as JSON |
| `/api/telemetry/clear` | POST | Clear recorded telemetry |
| `/ws/robot` | WS | Bridge connection endpoint |

## Troubleshooting

**"could not build graph"** — usually means `../Orion/src/agent/graph.py` can't be found. Check that the folders are siblings.

**"No module named src.agent..."** — the venv is missing Orion's dependencies. Install them with `pip install -r ../Orion/requirements.txt`.

**Bridge won't connect** — make sure `BRIDGE_TOKEN` matches between the server and bridge. Default is `dev-bridge-token` (set via env var or hardcoded).

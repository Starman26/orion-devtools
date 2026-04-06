# ORION DevTools

Local testing tooling for the ORION multi-agent system. Runs the agent graph from `../Orion/` with a pluggable robot bridge (simulated or Gazebo).

## Quick Start

```powershell
cd C:\Products\FINAL_PRODUCTS\orion-devtools

# Activate the Orion venv (has all agent dependencies)
..\Orion\.venv\Scripts\Activate.ps1

# Install devtools-specific dependencies
pip install -r requirements.txt

# Run
python test_server.py
```

Open [http://localhost:8000](http://localhost:8000)

## Architecture

```
orion-devtools/
├── test_server.py            # FastAPI server — imports graph from ../Orion
├── bridge/
│   ├── robot_interface.py    # Abstract RobotBridge + dataclasses
│   ├── sim_bridge.py         # SimXArmBridge — in-memory xArm 6 simulation
│   └── gazebo_bridge.py      # GazeboBridge — connects to Gazebo via rosbridge
├── requirements.txt
└── .gitignore
```

The server adds `../Orion` to `sys.path` and loads the `.env` from there. No files in the Orion repo are modified.

## Robot Bridges

### SimXArmBridge (default)
In-memory simulation of an xArm 6. Validates workspace limits, joint limits, and requires the real enable sequence (`motion_enable → set_mode(0) → set_state(0)`). No external dependencies.

### GazeboBridge
Connects to a Gazebo simulation via rosbridge WebSocket (roslibpy). Publishes `JointTrajectory` messages and subscribes to `/joint_states`.

**Gazebo in WSL2 setup:**

1. In WSL2, install ROS 2 + Gazebo + xarm packages
2. Launch the simulation:
   ```bash
   ros2 launch xarm_gazebo xarm6_beside_table_gazebo.launch.py
   ```
3. Start rosbridge:
   ```bash
   ros2 launch rosbridge_server rosbridge_websocket_launch.xml
   ```
4. From Windows, switch bridge in the UI or call:
   ```
   POST /api/robot/bridge
   {"bridge": "gazebo", "host": "localhost", "port": 9090}
   ```

### Real Robot (future)
Not yet implemented. Will connect to xArm SDK directly.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI |
| GET | `/health` | Status + loaded nodes + bridge info |
| POST | `/api/chat` | SSE streaming chat with the agent |
| POST | `/api/confirm` | Resume HITL (human-in-the-loop) |
| GET | `/api/robot/state` | Current robot state + action log |
| POST | `/api/robot/command` | Send manual commands to the robot |
| POST | `/api/robot/bridge` | Switch between sim/gazebo bridges |

"""
GazeboBridge — connects to Gazebo via rosbridge WebSocket (roslibpy).

Requires:
  - WSL2 with ROS 2 + Gazebo running the xarm6 simulation
  - rosbridge_server running on ws://localhost:9090
  - xarm6_traj_controller active

Usage:
    bridge = GazeboBridge(host="localhost", port=9090)
    bridge.connect()
"""

import math
import time
from datetime import datetime
from typing import List, Tuple

from .robot_interface import RobotBridge, RobotState, ActionLog

try:
    import roslibpy
except ImportError:
    roslibpy = None


HOME_JOINTS_RAD = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class GazeboBridge(RobotBridge):
    def __init__(self, host: str = "localhost", port: int = 9090):
        self._host = host
        self._port = port
        self._client = None
        self._state = RobotState(is_simulation=False)
        self._action_log: List[ActionLog] = []
        self._current_joints_rad: List[float] = [0.0] * 6
        self._joint_sub = None
        self._traj_pub = None

    def _log(self, command: str, params: dict, result: Tuple[int, str], duration_ms: float = 0):
        self._action_log.append(ActionLog(
            timestamp=datetime.utcnow().isoformat() + "Z",
            command=command,
            params=params,
            result=result,
            duration_ms=duration_ms,
        ))
        if len(self._action_log) > 200:
            self._action_log = self._action_log[-200:]

    @staticmethod
    def _deg2rad(deg: float) -> float:
        return deg * math.pi / 180.0

    @staticmethod
    def _rad2deg(rad: float) -> float:
        return rad * 180.0 / math.pi

    def _on_joint_state(self, msg):
        positions = msg.get("position", [])
        if len(positions) >= 6:
            self._current_joints_rad = list(positions[:6])
            self._state.joints = [self._rad2deg(r) for r in self._current_joints_rad]

    def connect(self) -> Tuple[int, str]:
        if roslibpy is None:
            r = (1, "roslibpy not installed. Run: pip install roslibpy")
            self._log("connect", {"host": self._host, "port": self._port}, r)
            return r

        try:
            self._client = roslibpy.Ros(host=self._host, port=self._port)
            self._client.run()

            # Subscribe to joint states
            self._joint_sub = roslibpy.Topic(
                self._client,
                "/joint_states",
                "sensor_msgs/JointState",
            )
            self._joint_sub.subscribe(self._on_joint_state)

            # Publisher for trajectory commands
            self._traj_pub = roslibpy.Topic(
                self._client,
                "/xarm6_traj_controller/command",
                "trajectory_msgs/JointTrajectory",
            )
            self._traj_pub.advertise()

            self._state.is_connected = True
            self._state.is_simulation = False
            r = (0, f"Connected to rosbridge at ws://{self._host}:{self._port}")
            self._log("connect", {"host": self._host, "port": self._port}, r)
            return r
        except Exception as e:
            r = (1, f"Failed to connect to rosbridge: {e}")
            self._log("connect", {"host": self._host, "port": self._port}, r)
            return r

    def disconnect(self) -> Tuple[int, str]:
        try:
            if self._joint_sub:
                self._joint_sub.unsubscribe()
            if self._traj_pub:
                self._traj_pub.unadvertise()
            if self._client:
                self._client.terminate()
        except Exception:
            pass
        self._state.is_connected = False
        self._state.is_enabled = False
        r = (0, "Disconnected from rosbridge")
        self._log("disconnect", {}, r)
        return r

    def get_state(self) -> RobotState:
        return self._state

    def get_action_log(self, limit=50) -> List[ActionLog]:
        return self._action_log[-limit:]

    def _publish_trajectory(self, angles_rad: List[float], duration_secs: float = 2.0):
        if not self._client or not self._client.is_connected:
            return (1, "Not connected to rosbridge")

        joint_names = [f"joint{i+1}" for i in range(6)]
        msg = roslibpy.Message({
            "joint_names": joint_names,
            "points": [{
                "positions": angles_rad,
                "time_from_start": {"secs": int(duration_secs), "nsecs": 0},
            }],
        })
        self._traj_pub.publish(msg)
        return (0, f"Trajectory published ({duration_secs}s)")

    def set_position(self, x, y, z, roll=180, pitch=0, yaw=0, speed=100) -> Tuple[int, str]:
        r = (1, "set_position requires IK/MoveIt which is not available via rosbridge. Use set_servo_angle instead.")
        self._log("set_position", {"x": x, "y": y, "z": z}, r)
        return r

    def set_servo_angle(self, angles: List[float], speed=50) -> Tuple[int, str]:
        if not self._state.is_connected:
            r = (1, "Not connected")
            self._log("set_servo_angle", {"angles": angles}, r)
            return r

        if len(angles) != 6:
            r = (1, f"Expected 6 joint angles, got {len(angles)}")
            self._log("set_servo_angle", {"angles": angles}, r)
            return r

        angles_rad = [self._deg2rad(a) for a in angles]
        max_delta = max(abs(a - b) for a, b in zip(angles_rad, self._current_joints_rad))
        duration = max(1.0, max_delta / (max(speed, 1) * math.pi / 180))

        code, msg = self._publish_trajectory(angles_rad, duration)
        if code != 0:
            self._log("set_servo_angle", {"angles": angles}, (code, msg))
            return (code, msg)

        self._state.joints = list(angles)
        r = (0, f"Joints command sent: {angles} ({duration:.1f}s)")
        self._log("set_servo_angle", {"angles": angles, "speed": speed}, r, duration * 1000)
        return r

    def set_gripper_position(self, pos: float, speed=1500) -> Tuple[int, str]:
        r = (1, "Gripper control not yet implemented for Gazebo bridge")
        self._log("set_gripper", {"pos": pos}, r)
        return r

    def go_home(self) -> Tuple[int, str]:
        if not self._state.is_connected:
            r = (1, "Not connected")
            self._log("go_home", {}, r)
            return r

        code, msg = self._publish_trajectory(HOME_JOINTS_RAD, duration_secs=3.0)
        if code != 0:
            self._log("go_home", {}, (code, msg))
            return (code, msg)

        self._state.joints = [0.0] * 6
        r = (0, "Homing command sent (3s trajectory)")
        self._log("go_home", {}, r, 3000)
        return r

    def emergency_stop(self) -> Tuple[int, str]:
        # Publish current position to freeze
        if self._state.is_connected:
            self._publish_trajectory(self._current_joints_rad, duration_secs=0.1)
        self._state.state = 4
        r = (0, "Emergency stop — held current position")
        self._log("emergency_stop", {}, r, 0)
        return r

    def motion_enable(self, enable=True) -> Tuple[int, str]:
        self._state.is_enabled = enable
        r = (0, f"Motion {'enabled' if enable else 'disabled'} (Gazebo — always enabled)")
        self._log("motion_enable", {"enable": enable}, r)
        return r

    def set_mode(self, mode: int) -> Tuple[int, str]:
        self._state.mode = mode
        r = (0, f"Mode set to {mode} (Gazebo — mode has no effect)")
        self._log("set_mode", {"mode": mode}, r)
        return r

    def set_state(self, state: int) -> Tuple[int, str]:
        self._state.state = state
        r = (0, f"State set to {state}")
        self._log("set_state", {"state": state}, r)
        return r

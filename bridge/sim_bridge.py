import math
from datetime import datetime
from typing import List, Tuple

from .robot_interface import RobotBridge, RobotState, ActionLog

# xArm 6 real specifications
HOME_POSITION = [206.0, 0.0, 120.5, 180.0, 0.0, 0.0]
HOME_JOINTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Workspace limits (mm)
WORKSPACE = {
    "x": (-700, 700),
    "y": (-700, 700),
    "z": (-150, 600),
}

# Joint limits (degrees) - xArm 6
JOINT_LIMITS = [
    (-360, 360),  # J1
    (-118, 120),  # J2
    (-225, 11),   # J3
    (-360, 360),  # J4
    (-97, 180),   # J5
    (-360, 360),  # J6
]

# Link lengths for FK visualization (mm) - from xArm 6 URDF
LINK_LENGTHS = [267.0, 289.6, 77.5, 342.5, 97.0, 76.0]


class SimXArmBridge(RobotBridge):
    def __init__(self):
        self._state = RobotState()
        self._action_log: List[ActionLog] = []
        self._ready = False

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

    def _check_ready(self):
        if not self._state.is_connected:
            return (1, "Not connected. Call connect() first.")
        if not self._ready:
            return (1, "Robot not ready. Call motion_enable(), set_mode(0), set_state(0) first.")
        return None

    def _validate_position(self, x, y, z):
        if not (WORKSPACE["x"][0] <= x <= WORKSPACE["x"][1]):
            return (1, f"X={x} out of workspace [{WORKSPACE['x'][0]}, {WORKSPACE['x'][1]}]")
        if not (WORKSPACE["y"][0] <= y <= WORKSPACE["y"][1]):
            return (1, f"Y={y} out of workspace [{WORKSPACE['y'][0]}, {WORKSPACE['y'][1]}]")
        if not (WORKSPACE["z"][0] <= z <= WORKSPACE["z"][1]):
            return (1, f"Z={z} out of workspace [{WORKSPACE['z'][0]}, {WORKSPACE['z'][1]}]")
        return None

    def _validate_joints(self, angles: List[float]):
        if len(angles) != 6:
            return (1, f"Expected 6 joint angles, got {len(angles)}")
        for i, (angle, (lo, hi)) in enumerate(zip(angles, JOINT_LIMITS)):
            if not (lo <= angle <= hi):
                return (1, f"Joint {i+1} angle {angle} out of limits [{lo}, {hi}]")
        return None

    def _calc_move_duration(self, target_pos, speed=100) -> float:
        current = self._state.position[:3]
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(current, target_pos[:3])))
        return max(50, (dist / max(speed, 1)) * 1000)

    def connect(self) -> Tuple[int, str]:
        self._state.is_connected = True
        self._state.is_simulation = True
        r = (0, "Connected (simulation mode)")
        self._log("connect", {}, r)
        return r

    def disconnect(self) -> Tuple[int, str]:
        self._state.is_connected = False
        self._state.is_enabled = False
        self._ready = False
        r = (0, "Disconnected")
        self._log("disconnect", {}, r)
        return r

    def get_state(self) -> RobotState:
        return self._state

    def get_action_log(self, limit=50) -> List[ActionLog]:
        return self._action_log[-limit:]

    def motion_enable(self, enable=True) -> Tuple[int, str]:
        if not self._state.is_connected:
            r = (1, "Not connected")
            self._log("motion_enable", {"enable": enable}, r)
            return r
        self._state.is_enabled = enable
        r = (0, f"Motion {'enabled' if enable else 'disabled'}")
        self._log("motion_enable", {"enable": enable}, r)
        return r

    def set_mode(self, mode: int) -> Tuple[int, str]:
        if not self._state.is_connected:
            r = (1, "Not connected")
            self._log("set_mode", {"mode": mode}, r)
            return r
        self._state.mode = mode
        r = (0, f"Mode set to {mode}")
        self._log("set_mode", {"mode": mode}, r)
        return r

    def set_state(self, state: int) -> Tuple[int, str]:
        if not self._state.is_connected:
            r = (1, "Not connected")
            self._log("set_state", {"state": state}, r)
            return r
        self._state.state = state
        if self._state.is_enabled and self._state.mode == 0 and state == 0:
            self._ready = True
        r = (0, f"State set to {state}")
        self._log("set_state", {"state": state}, r)
        return r

    def set_position(self, x, y, z, roll=180, pitch=0, yaw=0, speed=100) -> Tuple[int, str]:
        err = self._check_ready()
        if err:
            self._log("set_position", {"x": x, "y": y, "z": z}, err)
            return err

        err = self._validate_position(x, y, z)
        if err:
            self._log("set_position", {"x": x, "y": y, "z": z}, err)
            return err

        target = [x, y, z, roll, pitch, yaw]
        duration = self._calc_move_duration(target, speed)
        self._state.position = target
        r = (0, f"Moved to [{x}, {y}, {z}] in {duration:.0f}ms")
        self._log("set_position", {"x": x, "y": y, "z": z, "roll": roll, "pitch": pitch, "yaw": yaw, "speed": speed}, r, duration)
        return r

    def set_servo_angle(self, angles: List[float], speed=50) -> Tuple[int, str]:
        err = self._check_ready()
        if err:
            self._log("set_servo_angle", {"angles": angles}, err)
            return err

        err = self._validate_joints(angles)
        if err:
            self._log("set_servo_angle", {"angles": angles}, err)
            return err

        max_delta = max(abs(a - b) for a, b in zip(angles, self._state.joints))
        duration = max(50, (max_delta / max(speed, 1)) * 1000)
        self._state.joints = list(angles)
        r = (0, f"Joints set to {angles}")
        self._log("set_servo_angle", {"angles": angles, "speed": speed}, r, duration)
        return r

    def set_gripper_position(self, pos: float, speed=1500) -> Tuple[int, str]:
        err = self._check_ready()
        if err:
            self._log("set_gripper", {"pos": pos}, err)
            return err
        if not (0 <= pos <= 850):
            r = (1, f"Gripper position {pos} out of range [0, 850]")
            self._log("set_gripper", {"pos": pos}, r)
            return r

        duration = abs(self._state.gripper_pos - pos) / max(speed, 1) * 1000
        self._state.gripper_pos = pos
        r = (0, f"Gripper {'opened' if pos > 400 else 'closed'} to {pos}")
        self._log("set_gripper", {"pos": pos, "speed": speed}, r, duration)
        return r

    def go_home(self) -> Tuple[int, str]:
        err = self._check_ready()
        if err:
            self._log("go_home", {}, err)
            return err

        duration = self._calc_move_duration(HOME_POSITION, speed=200)
        self._state.position = list(HOME_POSITION)
        self._state.joints = list(HOME_JOINTS)
        r = (0, "Moved to home position")
        self._log("go_home", {}, r, duration)
        return r

    def emergency_stop(self) -> Tuple[int, str]:
        self._state.state = 4
        self._ready = False
        r = (0, "Emergency stop executed")
        self._log("emergency_stop", {}, r, 0)
        return r

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple
from datetime import datetime


@dataclass
class RobotState:
    position: List[float] = field(default_factory=lambda: [206.0, 0.0, 120.5, 180.0, 0.0, 0.0])
    joints: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    gripper_pos: float = 850.0  # 0=closed, 850=open
    mode: int = 0
    state: int = 0
    error_code: int = 0
    is_connected: bool = False
    is_enabled: bool = False
    is_simulation: bool = True


@dataclass
class ActionLog:
    timestamp: str
    command: str
    params: dict
    result: Tuple[int, str]
    duration_ms: float


class RobotBridge(ABC):
    @abstractmethod
    def connect(self) -> Tuple[int, str]: ...

    @abstractmethod
    def disconnect(self) -> Tuple[int, str]: ...

    @abstractmethod
    def get_state(self) -> RobotState: ...

    @abstractmethod
    def get_action_log(self, limit: int = 50) -> List[ActionLog]: ...

    @abstractmethod
    def set_position(self, x, y, z, roll=180, pitch=0, yaw=0, speed=100) -> Tuple[int, str]: ...

    @abstractmethod
    def set_servo_angle(self, angles: List[float], speed=50) -> Tuple[int, str]: ...

    @abstractmethod
    def set_gripper_position(self, pos: float, speed=1500) -> Tuple[int, str]: ...

    @abstractmethod
    def go_home(self) -> Tuple[int, str]: ...

    @abstractmethod
    def emergency_stop(self) -> Tuple[int, str]: ...

    @abstractmethod
    def motion_enable(self, enable=True) -> Tuple[int, str]: ...

    @abstractmethod
    def set_mode(self, mode: int) -> Tuple[int, str]: ...

    @abstractmethod
    def set_state(self, state: int) -> Tuple[int, str]: ...

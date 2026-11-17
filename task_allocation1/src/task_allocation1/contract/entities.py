"""实体：Task / UAV / Nest。纯数据，可在内存直接构造，不碰任何 I/O。

字段口径对齐参考项目 ``common/types.py``，但只保留本内核消费的子集：
时间窗、异构性能、服务时间；不提供时序组字段。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .distance import Position
from .enums import TaskStatus, UAVStatus


@dataclass
class Task:
    id: int
    x: float
    y: float
    z: float = 0.0
    reward: float = 1.0
    earliest: float | None = None
    latest: float | None = None
    arrival_time: float = 0.0
    status: TaskStatus = TaskStatus.PENDING
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def position(self) -> Position:
        return (self.x, self.y, self.z)


@dataclass
class UAV:
    id: int
    x: float
    y: float
    z: float = 0.0
    remaining_range: float = 1e9
    speed: float = 10.0
    max_time: float | None = None
    max_range: float | None = None
    nest_id: int | None = None
    status: UAVStatus = UAVStatus.IDLE
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def position(self) -> Position:
        return (self.x, self.y, self.z)

    @property
    def is_available(self) -> bool:
        return self.status == UAVStatus.IDLE

    def full_charge_range(self, default: float) -> float:
        """满电航程唯一口径：per-UAV max_range 优先，缺省回退全局值。"""
        return float(self.max_range) if self.max_range is not None else float(default)


@dataclass
class Nest:
    id: int
    x: float
    y: float
    z: float = 0.0
    capacity: int = 10
    available: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def position(self) -> Position:
        return (self.x, self.y, self.z)

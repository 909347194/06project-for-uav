"""实体：Task / UAV / Nest。纯数据，可在内存直接构造，不碰任何 I/O。

字段口径对齐参考项目 ``common/types.py``，保留完整语义（时间窗、异构性能、
服务时间等），供上层滚动服务复用；本库单周期内核消费其子集。
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
    sequence_group: int | None = None
    sequence_order: int = 0

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

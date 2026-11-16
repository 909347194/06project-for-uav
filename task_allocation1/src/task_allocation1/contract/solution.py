"""解结构：UAVTour / AllocationSolution。

对应论文决策变量：``y_j`` ↔ executed_task_ids，``x_uj / π_u`` ↔ tour.task_ids，
``z_ub`` ↔ tour.end_nest_id。**本库固定 z_ub = 起飞机巢（uav.nest_id）**，
因此终点不是自由决策变量，也不做容量竞争分配。

同巢不变量：每条非空航次满足 ``end_nest_id == start_nest_id``（起飞机巢）。
:meth:`AllocationSolution.home_return_ok` 供调用方与测试校验。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .entities import UAV


@dataclass
class UAVTour:
    uav_id: int
    task_ids: list[int] = field(default_factory=list)
    end_nest_id: int | None = None
    start_x: float = 0.0
    start_y: float = 0.0
    start_z: float = 0.0
    start_nest_id: int | None = None
    est_cost: float = 0.0
    est_reward: float = 0.0
    feasible: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_idle(self) -> bool:
        return not self.task_ids

    @property
    def start_position(self) -> tuple[float, float, float]:
        return (self.start_x, self.start_y, self.start_z)

    @classmethod
    def from_uav(
        cls,
        uav: UAV,
        task_ids: list[int] | None = None,
        end_nest_id: int | None = None,
    ) -> UAVTour:
        """用 UAV 当前状态快照起点；终点缺省 = 起飞机巢（同巢口径）。"""
        return cls(
            uav_id=uav.id,
            task_ids=list(task_ids or []),
            end_nest_id=(uav.nest_id if end_nest_id is None else end_nest_id),
            start_x=uav.x,
            start_y=uav.y,
            start_z=uav.z,
            start_nest_id=uav.nest_id,
        )

    def is_home_return(self) -> bool:
        """空航次恒真；非空航次要求终点巢 == 起飞巢。"""
        if self.is_idle:
            return True
        return self.end_nest_id is not None and self.end_nest_id == self.start_nest_id


@dataclass
class AllocationSolution:
    tours: list[UAVTour] = field(default_factory=list)
    candidate_task_ids: list[int] = field(default_factory=list)
    executed_task_ids: list[int] = field(default_factory=list)
    f1_reward: float = 0.0
    f2_cost: float = 0.0
    feasible: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    def assignment_pairs(self) -> list[tuple[int, int]]:
        return [
            (tour.uav_id, tid)
            for tour in self.tours
            for tid in tour.task_ids
        ]

    def task_to_uav(self) -> dict[int, int]:
        return {tid: tour.uav_id for tour in self.tours for tid in tour.task_ids}

    def tour_of(self, uav_id: int) -> UAVTour | None:
        return next((t for t in self.tours if t.uav_id == uav_id), None)

    def validate_unique_tasks(self) -> bool:
        """一任务一机：任一 task_id 出现两次即失败。"""
        seen: set[int] = set()
        for tour in self.tours:
            for tid in tour.task_ids:
                if tid in seen:
                    return False
                seen.add(tid)
        return True

    def home_return_ok(self) -> bool:
        """同巢起降硬不变量：所有航次终点 == 起飞巢。"""
        return all(t.is_home_return() for t in self.tours)

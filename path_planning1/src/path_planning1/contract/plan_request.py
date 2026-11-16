"""**预留契约**：单条航段的规划请求（一个起点→一个终点）。

早期为 ``PlanAlgorithm.plan(request, world)`` 定的形状，当前无实现者、不被消费。
实际在线入口是 ``PlanningContext.plan(start_xy, goal_xy, start_z_m, goal_z_m=...)``，
请求参数由 ``core/params_config.py::request_from_config`` 从 JSON 读出；配置结构见
``default_config()``。后端不要按本类型拼请求。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import defaults
from .enums import Objective
from .planner_params import PlannerParams
from .vehicle_limits import EnergyModelParams, VehicleLimits
from .world_spec import RegionOfInterest


@dataclass(frozen=True)
class PlanRequest:
    world_id: str
    start_enu: tuple[float, float, float]
    goal_enu: tuple[float, float, float]

    objective: Objective = Objective.TIME
    vehicle: VehicleLimits = field(default_factory=VehicleLimits)
    energy: EnergyModelParams = field(default_factory=EnergyModelParams)
    params: PlannerParams = field(default_factory=PlannerParams)
    roi: RegionOfInterest | None = None

    seed: int = defaults.DEFAULT_SEED
    time_budget_s: float | None = None
    route_segment_id: str | None = None
    request_id: str | None = None

"""规划结果：轨迹、指标、诊断、结果包。

分层：``Trajectory`` 是算法产物；``Metrics``/``Diagnostics`` 是给人看与给服务层用的副产物。

**分组**：``Trajectory / Metrics / Diagnostics`` 是当前实现真正消费并复用的类型；
``PlanResult`` 是**预留契约**（``PlanAlgorithm.plan`` 的返回形状），无实现者。实际交付的
结果包是 ``core/route_segment_outcome.py::RouteSegmentPlan``：它复用了上面三个类型，但多出
``path_enu / stats / timings / params_hash``，且 ``ok`` 判据看 ``path_enu`` 而非 ``trajectory``。
后端按 ``RouteSegmentPlan`` 接，别按 ``PlanResult`` 反序列化。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .enums import FailureCode, Status
from .world_spec import RegionOfInterest


@dataclass
class Trajectory:
    """固定时间步的时空轨迹（本包优化的 Δt 均匀）。"""

    t_s: np.ndarray               # (N,) 从出发起算
    position_enu: np.ndarray      # (N, 3) m
    velocity_enu: np.ndarray      # (N, 3) 地速 m/s
    acceleration_enu: np.ndarray  # (N, 3) m/s²
    dt_uniform_s: float


@dataclass
class Metrics:
    energy_j: float | None = None
    energy_per_km_wh: float | None = None
    length_m: float | None = None
    duration_s: float | None = None
    min_clearance_m: float | None = None
    max_airspeed_mps: float | None = None
    max_acceleration_mps2: float | None = None
    mean_altitude_m: float | None = None


@dataclass
class Diagnostics:
    roi_used: RegionOfInterest | None = None
    candidates: int = 0                 # 实际尝试的 RRT 初值数
    feasible_candidates: int = 0        # 其中真几何净空达标的候选数
    rrt_iterations: int | None = None   # 选中候选的 RRT 迭代数
    rrt_nodes: int | None = None        # 选中候选的树节点数
    solver_status: str | None = None
    solver_iterations: int | None = None
    params_hash: str | None = None
    elapsed_s: float | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class PlanResult:
    """结果包。``ok`` 只看 ``status`` 与是否真的有轨迹，不看 message。"""

    algorithm_version: str
    status: Status
    failure_code: FailureCode = FailureCode.NONE
    message: str = ""

    trajectory: Trajectory | None = None
    metrics: Metrics = field(default_factory=Metrics)
    diagnostics: Diagnostics = field(default_factory=Diagnostics)

    request_id: str | None = None
    route_segment_id: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is Status.OK and self.trajectory is not None

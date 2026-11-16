"""请求：一次单周期分配的完整输入。纯数据，不含 I/O、不含求解逻辑。

字段：
- tasks / uavs / nests：内存实体列表；只有 ``uav.is_available`` 的机参与。
- distance_fn：注入的有向距离口径（米）；None → 按 ``cost_config`` 解析（默认三维欧氏）。
- cost_config：距离代价口径（``euclid3d`` / ``vertical_section`` 及其飞行带、净空、爬升率）。
- ground_z / obstacle_top_z：``vertical_section`` 的采样器（地面高程 / 建筑屋顶包络顶）；
  数据仍由调用方给，本库不读文件。
- problem_config / solver_config：评价口径与求解预算。
- fleet_uavs：全部无人机（含 BUSY/FAULT），仅用于推导机巢占用；缺省 = uavs。
- current_time：本周期规划时刻（时间窗游标起点）；给出则覆盖 problem_config。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import ProblemConfig, SolverConfig
from .cost_config import CostConfig, SurfaceSampler
from .distance import DistanceFn
from .entities import UAV, Nest, Task


@dataclass
class AllocateRequest:
    tasks: list[Task] = field(default_factory=list)
    uavs: list[UAV] = field(default_factory=list)
    nests: list[Nest] = field(default_factory=list)
    distance_fn: DistanceFn | None = None
    cost_config: CostConfig = field(default_factory=CostConfig)
    ground_z: SurfaceSampler | None = None
    obstacle_top_z: SurfaceSampler | None = None
    problem_config: ProblemConfig = field(default_factory=ProblemConfig)
    solver_config: SolverConfig = field(default_factory=SolverConfig)
    fleet_uavs: list[UAV] | None = None
    current_time: float | None = None

"""task_allocation1 —— 多机巢多无人机协同任务分配纯计算内核。

**这是本包唯一对外边界**：调用方只需 ``import task_allocation1 as ta1``。
顶层 ``__all__`` 是正式 API 清单（``tests/test_ta1_public_api.py`` 守卫）。

    一次分配：result = ta1.allocate(ta1.AllocateRequest(tasks=..., uavs=..., nests=...))
    读结果：  result.solution.tours / result.solution.executed_task_ids / result.metrics_dict()
    同巢校验：result.home_return_ok （每条非空航次 end_nest_id == 起飞巢）

口径：**同巢起降**——每架机从自己的机巢起飞、任务做完返回同一个机巢；终点不是自由
决策变量，也不做异巢降落/最小费用流。论文的三段主线均保留：
    DMDE 进化搜索（代价空间差分 + 温度退火反映射）
    → 解码修复链（合并/航程修复/改派/覆盖补全/时间窗重排）
    → 交付后局部优化（跨机 relocate/swap + 组内 2-opt，四道门）。

纯计算库：不联网、不读文件、不落库、无全局随机状态；距离口径由 ``distance_fn``
或 ``cost_config`` 决定（默认三维欧氏；``vertical_section`` 走垂直切面估计，地面/建筑
包络由调用方注入采样器）。随机性由 ``SolverConfig.seed`` 注入。服务壳（HTTP/缓存/
进程池）、跨周期滚动编排均不在本包。
两库分工：本库是分配层（谁飞、什么顺序、回哪个巢，恒为起飞巢），
``path_planning1`` 是落地层（分配定稿后逐航段出真航迹与时长）；两库各自带契约、不共享代码。

分层：``contract/`` 纯数据与协议 → ``core/`` 算法 → 顶层门面只做 re-export。
单文件上限 ``contract/`` ≤120 行、``core/`` ≤300 行（``tests/test_ta1_layout.py`` 守卫）。
"""

from __future__ import annotations

from . import contract
from .contract import (
    UAV,
    AllocateRequest,
    AllocateResult,
    AllocationSolution,
    ConstraintConfig,
    CostConfig,
    DistanceFn,
    EndNestPolicy,
    ModelType,
    Nest,
    ProblemConfig,
    SolverConfig,
    SurfaceSampler,
    Task,
    TaskStatus,
    UAVStatus,
    UAVTour,
    defaults,
    euclidean3d,
)
from .core.solver import allocate
from .core.vertical_section import VerticalSectionCost, build_distance_fn

ALGORITHM_ID = "ta1-dmde-home-nest"
__version__ = "0.2.0"

__all__ = [
    "ALGORITHM_ID",
    "UAV",
    "AllocateRequest",
    "AllocateResult",
    "AllocationSolution",
    "ConstraintConfig",
    "CostConfig",
    "DistanceFn",
    "EndNestPolicy",
    "ModelType",
    "Nest",
    "ProblemConfig",
    "SolverConfig",
    "SurfaceSampler",
    "Task",
    "TaskStatus",
    "UAVStatus",
    "UAVTour",
    "VerticalSectionCost",
    "__version__",
    "allocate",
    "build_distance_fn",
    "contract",
    "defaults",
    "euclidean3d",
]

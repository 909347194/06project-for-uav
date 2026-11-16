"""契约层：纯数据、枚举、配置、请求/结果类型。

本子包**不 import core**（``tests/test_ta1_layout.py`` 守卫），也不碰任何 I/O：
不读文件、不联网、不落库。所有需要算距离的地方只声明 ``distance_fn`` 协议，
具体实现由调用方注入。
"""

from .config import ConstraintConfig, ProblemConfig, SolverConfig
from .cost_config import CostConfig, SurfaceSampler
from .defaults import (
    DEFAULT_NEST_OFFSET,
    DEFAULT_RANGE_EXCESS_WEIGHT_MODE,
    HOME_END_NEST_MODE,
)
from .distance import DistanceFn, euclidean3d
from .entities import UAV, Nest, Task
from .enums import EndNestPolicy, ModelType, TaskStatus, UAVStatus
from .request import AllocateRequest
from .result import AllocateResult
from .solution import AllocationSolution, UAVTour

__all__ = [
    "DEFAULT_NEST_OFFSET",
    "DEFAULT_RANGE_EXCESS_WEIGHT_MODE",
    "HOME_END_NEST_MODE",
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
    "euclidean3d",
]

"""path_planning1 对外契约（子包形式，每个文件只讲一件事）。

    enums.py              目标 / 状态 / 失败码
    local_frame.py        LocalFrame、GridSpec（经纬度↔ENU、索引↔ENU）
    geometry_types.py     Building、NoFlyZone（扩展槽）
    world_spec.py         WorldSnapshot、RegionOfInterest（两层 ROI 语义）+ 默认飞行带
    vehicle_limits.py     VehicleLimits（含净空）、EnergyModelParams
    planner_params.py     PlannerParams（只放算法旋钮）
    plan_request.py       PlanRequest（一个航段一次请求）
    plan_result.py        Trajectory / Metrics / Diagnostics / PlanResult
    planner_protocol.py   PlanAlgorithm
    defaults.py           所有默认数值的唯一清单

约定：本子包**不 import 任何 core/api 模块**（纯数据与协议），由
``tests/test_pp1_layout.py`` 强制。

**分组（交付时看清楚）**：``Trajectory / Metrics / Diagnostics / RegionOfInterest /
VehicleLimits / PlannerParams / LocalFrame / GridSpec / Building`` 与各枚举是当前实现
真正消费的类型；``PlanAlgorithm / PlanRequest / PlanResult / WorldSnapshot`` 是**内部遗留
形状**——早期定的"两包同构"契约，无实现者、不被任何 ``core`` 模块消费。它们仍可从本
子包导入（一组契约测试依赖它们），但**已从 ``__all__``
移除**：``from path_planning1.contract import *`` 不会带出它们，顶层门面也不导出。
后端只认 ``import path_planning1 as pp1`` 的顶层 ``__all__``，不要按这几个名字写接入代码。
"""

from __future__ import annotations

from . import defaults
from .enums import FailureCode, Objective, Status
from .geometry_types import Building, NoFlyZone
from .local_frame import M_PER_DEG_LAT, M_PER_DEG_LON_EQ, GridSpec, LocalFrame

# 下面四个 `as 原名` 是**显式再导出**：保留子包导入路径给已有夹具/测试，
# 但不进 __all__，因此不属于交付面（见模块 docstring）。
from .plan_request import PlanRequest as PlanRequest
from .plan_result import Diagnostics, Metrics, Trajectory
from .plan_result import PlanResult as PlanResult
from .planner_params import PlannerParams
from .planner_protocol import PlanAlgorithm as PlanAlgorithm
from .vehicle_limits import EnergyModelParams, VehicleLimits
from .world_spec import DEFAULT_ALT_MAX_M, DEFAULT_ALT_MIN_M, RegionOfInterest
from .world_spec import WorldSnapshot as WorldSnapshot

__all__ = [
    "DEFAULT_ALT_MAX_M",
    "DEFAULT_ALT_MIN_M",
    "DEFAULT_CLEARANCE_M",
    "DEFAULT_GRID_SPACING_M",
    "DEFAULT_ROI_MARGIN_M",
    "DEFAULT_ROI_WIDTH_RATIO",
    "DEFAULT_SEED",
    "M_PER_DEG_LAT",
    "M_PER_DEG_LON_EQ",
    "Building",
    "Diagnostics",
    "EnergyModelParams",
    "FailureCode",
    "GridSpec",
    "LocalFrame",
    "Metrics",
    "NoFlyZone",
    "Objective",
    "PlannerParams",
    "RegionOfInterest",
    "Status",
    "Trajectory",
    "VehicleLimits",
    "defaults",
]

# 上面四个内部遗留形状仍在本模块可导入（import 语句保留），只是不进 __all__。
# test_pp1_public_api.py 守这条线：顶层门面与 contract.__all__ 都不许出现它们。

# 常用默认值提升到包级，外部只 import 一次；数值仍只在 ``defaults.py`` 定义
DEFAULT_CLEARANCE_M = defaults.DEFAULT_CLEARANCE_M
DEFAULT_GRID_SPACING_M = defaults.DEFAULT_GRID_SPACING_M
DEFAULT_ROI_MARGIN_M = defaults.DEFAULT_ROI_MARGIN_M
DEFAULT_ROI_WIDTH_RATIO = defaults.DEFAULT_ROI_WIDTH_RATIO
DEFAULT_SEED = defaults.DEFAULT_SEED

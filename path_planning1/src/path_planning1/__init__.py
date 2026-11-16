"""path_planning1 —— 静态障碍（建筑 + 禁飞区）约束、无风的「RRT 初始化 + 连续轨迹优化（最短飞行时间）」封装。

**这是本包唯一对外边界**：后端只需 ``import path_planning1 as pp1``，从顶层取启动装配、
在线入口、结果类型与配置解析函数；不需要（也不应该）从 ``core`` 深路径导入。

    启动一次：pp1.create_context(cfg) → PlanningContext（装载世界 + 取得全域场 + 覆盖自检）
              可选 field_artifact=<目录>：读构建期只读产物，毫秒级启动、多进程共享内存
              构建期：ctx.save_field(<目录>) 把刚算好的场落成产物
    每条请求：ctx.plan(start_xy, goal_xy, start_z_m) → RouteSegmentPlan（高度一律绝对 z）
    读结果：  res.status / res.failure_code / res.metrics_dict() / res.trajectory
    部署事故：pp1.FieldContractError（产物过期 / 契约漂移 / 窗口越界），别映射成 400

当前能力：单航段（ROI 走廊 → 窗口 → SDF → RRT → 净空真几何终检 → 时间域 NLP，含 DEM
真高带 10–120 m、净空 5 m）；建筑来自数据目录 ``building/*.shp``（DBF 为 GBK）或
``building/*.geojson``（UTF-8），共用同一套 Polygon/高度/面积口径；静态禁飞区来自
``nofly.geojson``（绝对高程、可悬浮），与建筑共用同一套障碍通道。结果形状是 ``RouteSegmentPlan``（``trajectory`` 为主交付，
``path_enu`` 为几何兜底）；能耗目标**未实现**（需要风场），见 ``README.md``「已知限制」。

两库分工：``task_allocation1`` 是分配层（谁飞、什么顺序、回哪个巢，恒为起飞巢），
本库是落地层（分配定稿后逐航段出真航迹与时长）；两库各自带契约、不共享代码。

分层：``contract/`` 纯数据与枚举 → ``core/`` 算法与启动装配 → 顶层门面只做 re-export。
HTTP / 缓存 / 进程池 / 超时 / 序列化归上层服务，本包不做，也不依赖 Web 框架。
单文件上限 ``contract/`` ≤120 行、``core/`` ≤300 行（``tests/test_pp1_layout.py`` 强制）。

**内部遗留，勿依赖**：``contract`` 子包里的 ``PlanAlgorithm`` / ``PlanRequest`` /
``PlanResult`` / ``WorldSnapshot`` 是早期设计的形状，当前实现不消费（无实现者）。
它们仍可从子包导入（现有测试引用），但已从 ``contract.__all__`` 与本门面
移除；后端不要按它们写接入代码。顶层 ``__all__`` 是唯一交付面清单，由
``tests/test_pp1_public_api.py`` 守卫。
"""

from __future__ import annotations

from . import contract
from .contract import (
    Diagnostics,
    FailureCode,
    LocalFrame,
    Metrics,
    Objective,
    RegionOfInterest,
    Status,
    Trajectory,
    VehicleLimits,
    defaults,
)
from .core.params_config import (
    default_config,
    field_mode_from_config,
    max_od_length_from_config,
    params_from_config,
    params_to_config,
    request_from_config,
    world_kwargs_from_config,
)
from .core.planning_context import PlanningContext, create_context
from .core.route_segment_outcome import RouteSegmentPlan, RouteSegmentStats
from .core.route_segment_params import RouteSegmentParams
from .core.world_field import FieldContractError

ALGORITHM_ID = "pp1-rrt-nlp-time"
__version__ = "0.4.0"

__all__ = [
    "ALGORITHM_ID",
    "Diagnostics",
    "FailureCode",
    "FieldContractError",
    "LocalFrame",
    "Metrics",
    "Objective",
    "PlanningContext",
    "RegionOfInterest",
    "RouteSegmentParams",
    "RouteSegmentPlan",
    "RouteSegmentStats",
    "Status",
    "Trajectory",
    "VehicleLimits",
    "__version__",
    "contract",
    "create_context",
    "default_config",
    "defaults",
    "field_mode_from_config",
    "max_od_length_from_config",
    "params_from_config",
    "params_to_config",
    "request_from_config",
    "world_kwargs_from_config",
]

"""**预留契约**（两包同构的 warmup / plan / describe）——当前**没有实现者**。

早期为统一两包接口定的形状。本包现在真正交付的是 ``path_planning1.create_context()``
+ ``PlanningContext.plan()``，返回 ``core/route_segment_outcome.py::RouteSegmentPlan``，
并不满足本协议：``warmup(world, request)`` 与 ``create_context(cfg)`` 不同构，``PlanResult``
也没有 ``path_enu / stats / timings / params_hash``，连 ``ok`` 判据都不一样。

保留这个文件只为不破坏已有引用；**后端不要按本协议写接入代码**。将来若真要统一两包，
必须先补一个适配实现（并决定 stats/path_enu 怎么映射），再启用本协议。
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .plan_request import PlanRequest
from .plan_result import PlanResult
from .world_spec import WorldSnapshot


@runtime_checkable
class PlanAlgorithm(Protocol):
    """``warmup`` 做一次性重活（ROI 窗口栅格化、SDF、CasADi 插值器），按 world + ROI 缓存。

    ``plan`` 是纯函数语义：同样输入给同样输出，不得修改 ``world``。
    """

    version: str

    def warmup(self, world: WorldSnapshot, request: PlanRequest) -> None: ...

    def plan(self, request: PlanRequest, world: WorldSnapshot) -> PlanResult: ...

    def describe(self) -> dict[str, Any]:
        """算法元信息：版本、参数、params_hash。服务 /v1/versions 用。"""

"""顶层门面守卫：后端工程师只需要 ``import path_planning1 as pp1``。

这一组测试锁的是**交付面**：正式 API 的名字、指向的实现、以及"哪些东西不许出现在门面上"。
以前顶层 docstring 宣称入口是 ``PlanAlgorithm.warmup/plan/describe``，但那个协议没有实现者，
后端按它接会接空；这里防止这类自相矛盾再回来。
"""

from __future__ import annotations

from path_planning1.contract import FailureCode, Status, defaults
from path_planning1.core import params_config as cfgmod
from path_planning1.core import planning_context as pc
from path_planning1.core import route_segment_outcome as outcome

import path_planning1 as pp1

_REQUIRED = {
    "ALGORITHM_ID", "Diagnostics", "FailureCode", "FieldContractError",
    "LocalFrame", "Metrics", "Objective",
    "PlanningContext", "RegionOfInterest", "RouteSegmentParams", "RouteSegmentPlan",
    "RouteSegmentStats", "Status", "Trajectory", "VehicleLimits", "__version__",
    "contract", "create_context", "default_config", "defaults",
    "field_mode_from_config", "max_od_length_from_config", "params_from_config",
    "params_to_config", "request_from_config", "world_kwargs_from_config",
}


def test_public_api_is_flat_and_complete() -> None:
    assert set(pp1.__all__) == _REQUIRED
    for name in pp1.__all__:
        assert getattr(pp1, name) is not None, name


def test_public_api_points_at_the_real_implementation() -> None:
    """门面只做 re-export，不是第二套实现。"""
    assert pp1.create_context is pc.create_context
    assert pp1.PlanningContext is pc.PlanningContext
    assert pp1.RouteSegmentPlan is outcome.RouteSegmentPlan
    assert pp1.RouteSegmentStats is outcome.RouteSegmentStats
    assert pp1.default_config is cfgmod.default_config
    assert pp1.params_from_config is cfgmod.params_from_config
    assert pp1.FailureCode is FailureCode and pp1.Status is Status


def test_reserved_contracts_stay_off_the_public_surface() -> None:
    """内部遗留形状仍可从 contract 子包导入（兼容夹具/测试），但两层交付面都不带它们。"""
    reserved = ("PlanAlgorithm", "PlanRequest", "PlanResult", "WorldSnapshot")
    for name in reserved:
        assert not hasattr(pp1, name), name
        assert hasattr(pp1.contract, name), name          # 兼容：路径还在
        assert name not in pp1.contract.__all__, name    # 但 import * 不再带出来
    assert pp1.FieldContractError is not None
    assert issubclass(pp1.FieldContractError, RuntimeError)
    assert not issubclass(pp1.FieldContractError, ValueError)  # 别被误映射成 4xx


def test_backend_gets_everything_from_one_import() -> None:
    cfg = pp1.default_config()
    assert cfg["world"]["field_mode"] == defaults.WORLD_FIELD_MODE_STARTUP
    assert cfg["world"]["max_od_length_m"] == defaults.WORLD_FIELD_MAX_OD_LENGTH_M
    assert pp1.params_from_config({}) == pp1.RouteSegmentParams()
    assert callable(pp1.create_context) and hasattr(pp1.PlanningContext, "plan")

    req = pp1.request_from_config({"request": {
        "start_xy": [1.0, 2.0], "goal_xy": [3.0, 4.0], "start_z_m": 25.0,
    }})
    assert req["start_xy"] == (1.0, 2.0) and req["goal_z_m"] is None


def test_metrics_dict_carries_the_field_flavour_hash() -> None:
    """后端常只序列化这份扁平字典，口径指纹丢了就分不清 startup / on_demand。"""
    out = outcome.rejected_plan(FailureCode.OUTSIDE_WORLD, "域外", "deadbeef1234")
    flat = out.metrics_dict()
    assert flat["params_hash"] == "deadbeef1234"
    assert flat["status"] == "infeasible"
    assert flat["failure_code"] == "outside_world"

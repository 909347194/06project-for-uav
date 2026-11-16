"""JSON/字典入口：缺省即默认、拼错报错、量值有界、可往返。"""

from __future__ import annotations

import json

import pytest
from path_planning1.contract import defaults
from path_planning1.core.params_config import (
    default_config,
    max_od_length_from_config,
    params_from_config,
    params_to_config,
    request_from_config,
    world_kwargs_from_config,
)
from path_planning1.core.route_segment_params import RouteSegmentParams


def test_empty_config_equals_defaults() -> None:
    base = RouteSegmentParams()
    assert params_from_config(None) == base
    assert params_from_config({}) == base
    assert params_from_config({"params": {}}) == base
    # JSON 往返：默认值必须能原样摊平再读回来
    assert params_from_config({"params": params_to_config(base)}) == base


def test_partial_config_keeps_other_defaults() -> None:
    p = params_from_config({"params": {"vehicle": {"v_max_mps": 15}}})
    assert p.vehicle.v_max_mps == 15.0
    assert p.vehicle.a_max_mps2 == defaults.DEFAULT_A_MAX_MPS2
    assert p.roi.width_ratio == defaults.DEFAULT_ROI_WIDTH_RATIO
    assert p.planner.n_resample == defaults.NLP_N_RESAMPLE


def test_json_string_numbers_are_accepted() -> None:
    p = params_from_config({"params": {"spacing_m": [4, "4", 2.0], "seed": "7"}})
    assert p.spacing_m == (4.0, 4.0, 2.0)
    assert p.seed == 7


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="未知键"):
        params_from_config({"params": {"vehcle": {}}})
    with pytest.raises(ValueError, match="未知顶层键"):
        params_from_config({"parms": {}})
    with pytest.raises(ValueError, match="未知键"):
        params_from_config({"params": {"vehicle": {"v_max": 10}}})


def test_zero_width_corridor_is_rejected() -> None:
    with pytest.raises(ValueError, match="零宽走廊"):
        params_from_config({"params": {"roi": {"width_ratio": None, "half_width_m": None}}})


@pytest.mark.parametrize("params", [
    {"roi": {"width_ratio": 0}},                 # 0 宽 = 零宽走廊
    {"spacing_m": [4, 4]},                       # 必须是 3 个数
    {"spacing_m": None},                         # 显式 null 也必须是 ValueError
    {"spacing_m": [4, 0, 2]},                    # 格距必须 > 0
    {"agl_min_m": 120, "agl_max_m": 10},         # 上下限反了
    {"vehicle": {"v_max_mps": 0}},
    {"planner": {"n_resample": 2}},              # 至少 3 个节点
    {"planner": {"dt_min_s": 10, "dt_max_s": 1}},
    {"planner": {"rrt_min_step_m": 100}},        # 大于 rrt_step_m=60
    {"planner": {"rrt_initializations": 0}},     # 至少 1 个初值
    {"planner": {"rrt_initializations": 21}},    # 上限挡住误配的线性放大
])
def test_invalid_values_are_rejected(params: dict) -> None:
    with pytest.raises(ValueError):
        params_from_config({"params": params})


def test_explicit_null_sections_are_rejected() -> None:
    """整段省略 = 用默认；显式写 null 是笔误，不能静默当默认、也不能抛 TypeError。"""
    for section in ("roi", "vehicle", "planner"):
        with pytest.raises(ValueError, match="不能为 null"):
            params_from_config({"params": {section: None}})
    with pytest.raises(ValueError, match="spacing_m"):
        params_from_config({"params": {"spacing_m": None}})


def test_round_trip_preserves_hash() -> None:
    from path_planning1.core.params_fingerprint import params_hash

    base = RouteSegmentParams()
    again = params_from_config(json.loads(json.dumps({"params": params_to_config(base)})))
    assert params_hash(planner=again.planner, vehicle=again.vehicle, roi=again.roi,
                       spacing_m=again.spacing_m, agl_range_m=(again.agl_min_m, again.agl_max_m),
                       seed=again.seed) == params_hash(
        planner=base.planner, vehicle=base.vehicle, roi=base.roi, spacing_m=base.spacing_m,
        agl_range_m=(base.agl_min_m, base.agl_max_m), seed=base.seed)


@pytest.mark.parametrize("params", [
    {"vehicle": {"v_max_mps": "inf"}},
    {"vehicle": {"a_max_mps2": float("nan")}},
    {"roi": {"margin_m": "inf"}},
    {"spacing_m": [4, float("nan"), 2]},
    {"agl_max_m": float("inf")},
])
def test_non_finite_params_are_rejected(params: dict) -> None:
    """NaN/Inf 与 lo/hi 比较恒为假，以前会一路流进求解器 / required_pad_m。"""
    with pytest.raises(ValueError, match="有限数"):
        params_from_config({"params": params})


def test_non_finite_max_od_is_rejected() -> None:
    for value in ("inf", float("nan"), float("inf")):
        with pytest.raises(ValueError, match="有限数"):
            max_od_length_from_config({"world": {"max_od_length_m": value}})


def test_world_and_request_sections() -> None:
    cfg = {"world": {"root": "data/worlds/guangzhou", "with_dem": False},
           "request": {"start_xy": [1636.8, 4198.2], "goal_xy": [2040.6, 1393.6],
                       "start_z_m": 25}}
    assert world_kwargs_from_config(cfg) == {"root": "data/worlds/guangzhou", "with_dem": False}
    req = request_from_config(cfg)
    assert req["start_xy"] == (1636.8, 4198.2) and req["goal_z_m"] is None
    with pytest.raises(ValueError, match="world.root"):
        world_kwargs_from_config({"request": cfg["request"]})
    with pytest.raises(ValueError, match="start_xy"):
        request_from_config({"request": {"start_xy": None, "goal_xy": [1, 2], "start_z_m": 25}})
    assert default_config()["params"] == params_to_config()

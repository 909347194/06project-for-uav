"""配置清单的守卫：任何默认值在别处写第二份，这些断言就会失败。

做法很土但有效——把"契约 dataclass 的默认值"和 ``contract/defaults.py`` 的常量逐项比对。
改造前 ``RouteSegmentParams`` / ``RRTConfig`` 各存一份 60/20/3000/0.1/5.0，改一处不生效；
``clearance_m`` 更是定义两次却长期没人发现，因为旧测试只看了 ``PlannerParams``。
"""

from __future__ import annotations

import dataclasses

import pytest
from path_planning1.contract import (
    EnergyModelParams,
    PlannerParams,
    PlanRequest,
    RegionOfInterest,
    VehicleLimits,
    WorldSnapshot,
    defaults,
)
from path_planning1.core.route_segment_planner import RouteSegmentParams
from path_planning1.core.rrt_path_search import RRTConfig

from path_planning1 import contract


def _default(obj, name: str):
    """取 dataclass 实例的字段默认值（含 default_factory）。"""
    f = next(f for f in dataclasses.fields(obj) if f.name == name)
    if f.default is not dataclasses.MISSING:
        return f.default
    return f.default_factory()


def test_planner_params_defaults_match_checklist() -> None:
    p = PlannerParams()
    assert p.rrt_step_m == defaults.RRT_STEP_M
    assert p.rrt_min_step_m == defaults.RRT_MIN_STEP_M
    assert p.rrt_sdf_step_scale == defaults.RRT_SDF_STEP_SCALE
    assert p.rrt_max_iters == defaults.RRT_MAX_ITERS
    assert p.rrt_goal_sample_rate == defaults.RRT_GOAL_SAMPLE_RATE
    assert p.rrt_shortcut_iters == defaults.RRT_SHORTCUT_ITERS
    assert p.clearance_sample_step_m == defaults.CLEARANCE_SAMPLE_STEP_M
    assert p.rrt_initializations == defaults.RRT_INITIALIZATIONS
    assert p.n_resample == defaults.NLP_N_RESAMPLE
    assert p.initial_time_step_s == defaults.NLP_INITIAL_TIME_STEP_S
    assert p.segment_samples == defaults.NLP_SEGMENT_SAMPLES
    assert p.dt_min_s == defaults.NLP_DT_MIN_S
    assert p.dt_max_s == defaults.NLP_DT_MAX_S
    assert p.ipopt_max_iter == defaults.NLP_IPOPT_MAX_ITER


def test_rrt_initializations_rejects_out_of_range_direct_construction() -> None:
    """JSON 解析有范围闸；直接构造 ``PlannerParams`` 也不能绕过。"""
    for n in (0, 21):
        with pytest.raises(ValueError, match="rrt_initializations"):
            PlannerParams(rrt_initializations=n)


def test_vehicle_and_energy_defaults_match_checklist() -> None:
    v, e = VehicleLimits(), EnergyModelParams()
    assert v.clearance_m == defaults.DEFAULT_CLEARANCE_M
    assert (v.v_max_mps, v.a_max_mps2) == (defaults.DEFAULT_V_MAX_MPS, defaults.DEFAULT_A_MAX_MPS2)
    assert (v.thrust_max_n, v.mass_kg) == (defaults.DEFAULT_THRUST_MAX_N, defaults.DEFAULT_MASS_KG)
    assert e.rho_kg_m3 == defaults.ENERGY_RHO_KG_M3
    assert e.gravity_mps2 == defaults.GRAVITY_MPS2


def test_route_segment_params_defaults_are_all_references() -> None:
    seg = RouteSegmentParams()
    assert seg.spacing_m == defaults.DEFAULT_GRID_SPACING_M
    assert seg.agl_min_m == defaults.DEFAULT_ALT_MIN_M
    assert seg.agl_max_m == defaults.DEFAULT_ALT_MAX_M
    assert seg.seed == defaults.DEFAULT_SEED
    assert seg.roi.width_ratio == defaults.DEFAULT_ROI_WIDTH_RATIO
    assert seg.roi.margin_m == defaults.DEFAULT_ROI_MARGIN_M
    assert seg.planner == PlannerParams() and seg.vehicle == VehicleLimits()


def test_world_and_request_defaults_are_all_references() -> None:
    w = WorldSnapshot(
        world_id="w",
        frame=contract.LocalFrame(lon0=0.0, lat0=0.0),
        buildings=[],
        grid=contract.GridSpec(origin_enu=(0.0, 0.0, 0.0), spacing_m=(4.0, 4.0, 2.0), shape=(2, 2, 2)),
    )
    assert (w.alt_min_m, w.alt_max_m) == (defaults.DEFAULT_ALT_MIN_M, defaults.DEFAULT_ALT_MAX_M)
    req = PlanRequest(world_id="w", start_enu=(0.0, 0.0, 0.0), goal_enu=(1.0, 0.0, 0.0))
    assert req.seed == defaults.DEFAULT_SEED


def test_rrt_config_has_no_independent_defaults() -> None:
    """**全部字段**必须是必填：只有 ``from_planner`` 能填，杜绝第三份默认值。"""
    expected = {
        "step_m", "min_step_m", "sdf_step_scale", "max_iters", "goal_sample_rate",
        "clearance_m", "seed", "sample_step_m", "shortcut_iters",
    }
    assert {f.name for f in dataclasses.fields(RRTConfig)} == expected
    for f in dataclasses.fields(RRTConfig):
        assert f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING, f.name
    cfg = RRTConfig.from_planner(PlannerParams(), clearance_m=defaults.DEFAULT_CLEARANCE_M)
    assert (cfg.sample_step_m, cfg.shortcut_iters) == (
        defaults.CLEARANCE_SAMPLE_STEP_M, defaults.RRT_SHORTCUT_ITERS
    )
    # 实现细节旋钮穿到 RRTConfig：在 PlannerParams 上改，RRTConfig 跟着改，中间没有第二份默认值
    varied = RRTConfig.from_planner(
        dataclasses.replace(PlannerParams(), rrt_shortcut_iters=7, clearance_sample_step_m=0.5),
        clearance_m=defaults.DEFAULT_CLEARANCE_M,
    )
    assert (varied.sample_step_m, varied.shortcut_iters) == (0.5, 7)


def test_contract_reexports_point_at_the_same_numbers() -> None:
    """包级常量必须是清单的引用，不是复制的字面量。"""
    assert contract.DEFAULT_ALT_MIN_M == defaults.DEFAULT_ALT_MIN_M
    assert contract.DEFAULT_ALT_MAX_M == defaults.DEFAULT_ALT_MAX_M
    assert contract.DEFAULT_CLEARANCE_M == defaults.DEFAULT_CLEARANCE_M
    assert contract.DEFAULT_GRID_SPACING_M == defaults.DEFAULT_GRID_SPACING_M
    assert contract.DEFAULT_SEED == defaults.DEFAULT_SEED


def test_checklist_snapshot_is_complete_and_serializable() -> None:
    snap = defaults.snapshot()
    for key in ("DEFAULT_ALT_MIN_M", "DEFAULT_CLEARANCE_M", "RRT_STEP_M", "DEFAULT_SEED"):
        assert key in snap
    assert all(k.isupper() for k in snap)          # 只有常量，没有函数/类
    assert RegionOfInterest().margin_m == 0.0      # 走廊只剩"多大"的参数（无幽灵开关）

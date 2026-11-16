"""建场契约守卫：外扩派生、覆盖自检、场身份指纹、准入结果码、无运行期回退。

这一组对应"边界 OD 不内缩、不丢弃、不现场重算；把覆盖范围做成建场契约"。
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from path_planning1.contract import (
    FailureCode,
    GridSpec,
    RegionOfInterest,
    Status,
    defaults,
)
from path_planning1.core import planning_context as pc
from path_planning1.core.corridor_window import corridor_window, required_pad_m
from path_planning1.core.params_config import (
    default_config,
    max_od_length_from_config,
    world_kwargs_from_config,
)
from path_planning1.core.params_fingerprint import digest_for
from path_planning1.core.route_segment_outcome import rejected_plan
from path_planning1.core.route_segment_params import RouteSegmentParams
from path_planning1.core.world_field import (
    FieldContractError,
    WindowOutsideField,
    WorldField,
    check_contract_coverage,
    field_signature,
)


def _pad(roi=None, spacing=(4.0, 4.0, 2.0), length=3000.0, extra=30.0, minimum=100.0):
    return required_pad_m(
        roi or RegionOfInterest(width_ratio=0.2, margin_m=50.0),
        spacing, length, extra_m=extra, min_pad_m=minimum,
    )


# —— 1. 外扩派生：按最终 lateral、对齐格距、单调 —————————————————————————————

@pytest.mark.parametrize("spacing", [(4.0, 4.0, 2.0), (3.0, 3.0, 2.0),
                                     (5.0, 2.5, 2.0), (2.5, 4.0, 1.0),
                                     (6.0, 4.0, 2.0), (5.0, 3.0, 2.0)])
def test_pad_is_a_multiple_of_both_horizontal_steps(spacing) -> None:
    """两条水平轴都要对齐：只对齐 min(sx, sy) 会让另一轴格网原点平移。"""
    pad = _pad(spacing=spacing)
    for step in spacing[:2]:
        assert abs(pad / step - round(pad / step)) < 1e-9
    assert pad >= defaults.WORLD_FIELD_PAD_MIN_M


def test_changing_contract_keeps_grid_phase_on_both_axes() -> None:
    """同一世界、同一 spacing，只调 L_max 也不能让取样格点整体平移。"""
    spacing = (6.0, 4.0, 2.0)
    pads = [_pad(spacing=spacing, length=L) for L in (1000.0, 3000.0, 6000.0)]
    for step in spacing[:2]:
        assert all(abs(p / step - round(p / step)) < 1e-9 for p in pads)
    assert pads[0] < pads[1] < pads[2]


def test_pad_uses_final_lateral_not_raw_ratio_length() -> None:
    """显式 half_width_m 会绕过 r·L；按 0.5·r·L 算会严重低估外扩。"""
    narrow = _pad(RegionOfInterest(width_ratio=0.2, margin_m=50.0), length=3000.0)
    explicit = _pad(RegionOfInterest(half_width_m=800.0, margin_m=50.0), length=3000.0)
    assert narrow == 392.0
    assert explicit > narrow + 400.0
    # margin 也算进 lateral，不是"额外再加一次"的独立量
    wide_margin = _pad(RegionOfInterest(width_ratio=0.2, margin_m=200.0))
    assert wide_margin > narrow


def test_pad_grows_with_max_od_length() -> None:
    assert _pad(length=1000.0) < _pad(length=3000.0) < _pad(length=6000.0)


# —— 2. 场身份与指纹 —————————————————————————————————————————————————————

def _field(bounds=(0.0, 0.0, 4000.0, 4000.0), pad=392.0,
           spacing=(4.0, 4.0, 2.0), nz=4) -> WorldField:
    sx, sy, _sz = spacing
    x0, y0, x1, y1 = bounds
    gx0, gy0 = x0 - pad, y0 - pad
    nx = math.ceil((x1 + pad - gx0) / sx) + 1
    ny = math.ceil((y1 + pad - gy0) / sy) + 1
    grid = GridSpec(origin_enu=(gx0, gy0, 10.0), spacing_m=spacing,
                    shape=(nx, ny, nz), yaw_rad=0.0)
    return WorldField(
        grid=grid,
        sdf=np.zeros((nz, nx, ny), dtype=np.float32),
        occupancy=np.zeros((nx, ny, nz), dtype=bool),
    )


def test_field_signature_distinguishes_field_versions() -> None:
    assert field_signature(None) == {}          # on_demand 无场签名
    a, b = _field(), _field(pad=100.0)
    sa, sb = field_signature(a), field_signature(b)
    assert sa != sb
    assert sa["world_field"]["shape"] == list(a.grid.shape)
    assert sb["world_field"]["origin_enu"][0] == -100.0


def test_digest_separates_startup_from_on_demand() -> None:
    p = RouteSegmentParams()
    on_demand = digest_for(p, field_signature(None))
    startup = digest_for(p, field_signature(_field()))
    assert on_demand == "fb7b2c24ca34"       # 默认 5 初值后的 on_demand 基线
    assert startup != on_demand
    assert digest_for(p, field_signature(_field(pad=100.0))) != startup


def test_sample_bias_is_part_of_the_field_flavour(monkeypatch) -> None:
    """改补偿量会换一套 SDF 基准却不改格网，光靠 origin/shape 抓不到。"""
    import path_planning1.core.world_field as wf

    p = RouteSegmentParams()
    base = field_signature(_field())
    assert base["world_field"]["sample_bias_m"] == defaults.WORLD_FIELD_SAMPLE_BIAS_M
    monkeypatch.setattr(wf, "WORLD_FIELD_SAMPLE_BIAS_M", 2.5)
    changed = field_signature(_field())
    assert changed != base
    assert digest_for(p, changed) != digest_for(p, base)
    # 补偿量只在场路径生效：on_demand 不带场额外键
    assert field_signature(None) == {}
    assert digest_for(p, field_signature(None)) == "fb7b2c24ca34"


# —— 3. 启动期覆盖自检 ————————————————————————————————————————————————————

def test_self_check_passes_for_derived_pad_and_fails_for_old_pad() -> None:
    roi = RegionOfInterest(width_ratio=0.2, margin_m=50.0)
    spacing = (4.0, 4.0, 2.0)
    bounds = (0.0, 0.0, 4000.0, 4000.0)
    check_contract_coverage(_field(bounds, 392.0), bounds, roi, spacing, 3000.0)
    with pytest.raises(WindowOutsideField):
        check_contract_coverage(_field(bounds, 100.0), bounds, roi, spacing, 3000.0)


def test_self_check_rejects_impossible_contract() -> None:
    roi = RegionOfInterest(width_ratio=0.2, margin_m=50.0)
    with pytest.raises(FieldContractError, match="没构造出任何合法"):
        check_contract_coverage(_field(), (0.0, 0.0, 0.0, 0.0), roi,
                                (4.0, 4.0, 2.0), 3000.0)


# —— 4. 配置层 ———————————————————————————————————————————————————————————

def test_max_od_length_config_is_separate_from_load_city() -> None:
    assert max_od_length_from_config(None) == defaults.WORLD_FIELD_MAX_OD_LENGTH_M
    assert default_config()["world"]["max_od_length_m"] == 3000.0
    cfg = {"world": {"root": "x", "max_od_length_m": 1500.0, "field_mode": "startup"}}
    assert max_od_length_from_config(cfg) == 1500.0
    assert world_kwargs_from_config(cfg) == {"root": "x"}
    with pytest.raises(ValueError, match="不能为 null"):
        max_od_length_from_config({"world": {"root": "x", "max_od_length_m": None}})
    with pytest.raises(ValueError, match="必须 ≥"):
        max_od_length_from_config({"world": {"root": "x", "max_od_length_m": 0}})
    with pytest.raises(ValueError, match="未知键"):
        max_od_length_from_config({"world": {"root": "x", "max_od_lenght_m": 10}})


# —— 5. 装配层准入：结果码而非异常 —————————————————————————————————————————

class _DS:
    buildings = ()
    frame = "frame"
    dem = None
    bounds_m = (0.0, 0.0, 1000.0, 1000.0)
    n_buildings = 0


def _ctx(field, *, max_od=500.0, mode=None) -> pc.PlanningContext:
    return pc.PlanningContext(
        dataset=_DS(),
        params=RouteSegmentParams(),
        field=field,
        field_mode=mode or (defaults.WORLD_FIELD_MODE_STARTUP if field is not None
                            else defaults.WORLD_FIELD_MODE_ON_DEMAND),
        bounds_m=_DS.bounds_m,
        max_od_length_m=max_od,
        field_pad_m=392.0 if field is not None else 0.0,
        digest="abc123",
    )


def test_outside_world_is_a_result_not_an_exception() -> None:
    for start, goal in [((-1.0, 0.0), (10.0, 10.0)),
                        ((10.0, 10.0), (10.0, 1001.0))]:
        out = _ctx(object()).plan(start, goal, 25.0)
        assert out.status is Status.INFEASIBLE
        assert out.failure_code is FailureCode.OUTSIDE_WORLD
        assert not out.ok and out.path_enu is None
        assert out.params_hash == "abc123" and out.diagnostics.params_hash == "abc123"
        assert out.timings["total_s"] == 0.0


def test_od_too_long_is_rejected_before_the_core() -> None:
    field = object()
    ctx = _ctx(field, max_od=500.0)
    out = ctx.plan((0.0, 0.0), (900.0, 900.0), 25.0)
    assert out.failure_code is FailureCode.OD_TOO_LONG
    assert "max_od_length_m" in out.message


def test_on_demand_does_not_apply_field_contract(monkeypatch) -> None:
    sentinel = object()
    calls = []
    monkeypatch.setattr(pc, "plan_route_segment", lambda *a, **k: (calls.append(k) or sentinel))
    ctx = _ctx(None, max_od=500.0)
    assert ctx.plan((0.0, 0.0), (900.0, 900.0), 25.0) is sentinel
    assert calls[0]["world_field"] is None
    # 但端点出界仍然拒（城市外没有数据，与有没有场无关）
    assert ctx.plan((-1.0, 0.0), (10.0, 10.0), 25.0).failure_code is FailureCode.OUTSIDE_WORLD


def test_valid_request_always_passes_the_field(monkeypatch) -> None:
    sentinel, field = object(), object()
    calls = []
    monkeypatch.setattr(pc, "plan_route_segment", lambda *a, **k: (calls.append(k) or sentinel))
    assert _ctx(field).plan((10.0, 10.0), (20.0, 20.0), 25.0, goal_z_m=60.0) is sentinel
    assert calls[0]["world_field"] is field
    assert calls[0]["goal_z_m"] == 60.0


def test_rejected_plan_shape_matches_core_failures() -> None:
    out = rejected_plan(FailureCode.OUTSIDE_WORLD, "msg", "abc")
    assert out.status is Status.INFEASIBLE
    assert out.failure_code is FailureCode.OUTSIDE_WORLD
    assert out.message == "msg" and out.path_enu is None
    assert out.metrics_dict()["status"] == "infeasible"


# —— 6. 核心规划器：startup 下没有运行期回退 —————————————————————————————————

def test_no_runtime_fallback_when_window_is_outside_field(monkeypatch) -> None:
    """场不够就让异常冒泡成配置错误；绝不悄悄重建 occupancy/SDF 换一套口径。"""
    import path_planning1.core.route_segment_planner as planner

    p = RouteSegmentParams()
    window = corridor_window((100.0, 100.0, 0.0), (300.0, 300.0, 0.0),
                             p.roi, p.spacing_m, p.agl_min_m, p.agl_max_m)
    monkeypatch.setattr(planner, "build_segment_window",
                        lambda *a, **k: (window, None, p.agl_min_m, p.agl_max_m))
    monkeypatch.setattr(planner, "filter_by_bbox", lambda *a, **k: [])
    rebuilt = []
    monkeypatch.setattr(planner, "build_occupancy",
                        lambda *a, **k: rebuilt.append("occupancy"))
    monkeypatch.setattr(planner, "build_sdf", lambda *a, **k: rebuilt.append("sdf"))

    class _BadField:
        grid = window.grid

        def window_sdf(self, _grid):
            raise WindowOutsideField("测试：建场契约被破坏")

    with pytest.raises(WindowOutsideField, match="建场契约"):
        planner.plan_route_segment(
            [], "frame", None, (100.0, 100.0), (300.0, 300.0), 25.0, p,
            world_field=_BadField(),
        )
    assert rebuilt == []          # 没有偷偷现算


def test_on_demand_still_computes_window_sdf(monkeypatch) -> None:
    """删的是异常兜底，不是 on_demand 的现算能力。"""
    import path_planning1.core.route_segment_planner as planner

    p = RouteSegmentParams()
    window = corridor_window((100.0, 100.0, 0.0), (300.0, 300.0, 0.0),
                             p.roi, p.spacing_m, p.agl_min_m, p.agl_max_m)
    monkeypatch.setattr(planner, "build_segment_window",
                        lambda *a, **k: (window, None, p.agl_min_m, p.agl_max_m))
    monkeypatch.setattr(planner, "filter_by_bbox", lambda *a, **k: [])
    rebuilt = []
    monkeypatch.setattr(planner, "build_occupancy",
                        lambda *a, **k: rebuilt.append("occupancy") or np.zeros(window.grid.shape))
    monkeypatch.setattr(planner, "build_sdf",
                        lambda *a, **k: rebuilt.append("sdf") or np.zeros(window.grid.shape))

    # 全 0 假 SDF 会在"起终点净空"这一步正常返回失败结果；这里只验证走到了现算分支。
    out = planner.plan_route_segment(
        [], "frame", None, (100.0, 100.0), (300.0, 300.0), 25.0, p,
        world_field=None,
    )
    assert rebuilt[:2] == ["occupancy", "sdf"]
    assert out.status is Status.INFEASIBLE


def test_explicit_half_width_does_not_break_pad_derivation() -> None:
    roi = RegionOfInterest(half_width_m=120.0, margin_m=50.0)
    p = replace(RouteSegmentParams(), roi=roi)
    assert pc.world_field_kwargs(_DS(), p)["pad_m"] == _pad(roi)

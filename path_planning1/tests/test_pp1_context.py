"""启动上下文守卫：默认启动建场一次、在线只取，on_demand 才回退现算。"""

from __future__ import annotations

import math
from dataclasses import replace

import path_planning1.core.planning_context as pc
import pytest
from path_planning1.contract import defaults
from path_planning1.core.corridor_window import required_pad_m
from path_planning1.core.params_config import (
    default_config,
    field_mode_from_config,
    world_kwargs_from_config,
)
from path_planning1.core.route_segment_params import RouteSegmentParams


class _FakeDataset:
    buildings = ()
    frame = "frame"
    dem = None
    # 贴近真实城市尺度：对角线 ≈8485 m > 默认 3000 m 契约，封顶逻辑不会误伤其余用例
    bounds_m = (0.0, 0.0, 6000.0, 6000.0)
    n_buildings = 0


@pytest.fixture
def startup(monkeypatch):
    """替换真的 load_city / build_world_field / plan_route_segment，只记录调用。"""
    ds, field, plan = _FakeDataset(), object(), object()
    calls = {"load": [], "build": [], "plan": []}

    def fake_load(**kwargs):
        calls["load"].append(kwargs)
        return ds

    def fake_build(**kwargs):
        calls["build"].append(kwargs)
        return field

    def fake_plan(*args, **kwargs):
        calls["plan"].append((args, kwargs))
        return plan

    monkeypatch.setattr(pc, "load_city", fake_load)
    monkeypatch.setattr(pc, "build_world_field", fake_build)
    monkeypatch.setattr(pc, "plan_route_segment", fake_plan)
    # 假 field 不是 WorldField：启动自检与场身份签名在这里只需记录/跳过，
    # 它们各自的真实行为由 test_pp1_field_contract.py 覆盖。
    monkeypatch.setattr(pc, "check_contract_coverage", lambda *a, **k: None)
    monkeypatch.setattr(pc, "field_signature", lambda _f: {})
    return ds, field, plan, calls


def test_field_mode_defaults_to_startup() -> None:
    assert defaults.DEFAULT_WORLD_FIELD_MODE == defaults.WORLD_FIELD_MODE_STARTUP
    assert field_mode_from_config(None) == defaults.WORLD_FIELD_MODE_STARTUP
    assert field_mode_from_config({}) == defaults.WORLD_FIELD_MODE_STARTUP
    assert field_mode_from_config({"world": {}}) == defaults.WORLD_FIELD_MODE_STARTUP
    assert default_config()["world"]["field_mode"] == defaults.WORLD_FIELD_MODE_STARTUP


def test_field_mode_rejects_typos_and_null() -> None:
    with pytest.raises(ValueError, match="不能为 null"):
        field_mode_from_config({"world": {"field_mode": None}})
    with pytest.raises(ValueError, match="field_mode"):
        field_mode_from_config({"world": {"field_mode": "sometimes"}})
    with pytest.raises(ValueError, match="未知键"):
        field_mode_from_config({"world": {"field_mod": "startup"}})


def test_field_mode_is_not_a_load_city_kwarg() -> None:
    cfg = {"world": {"root": "x", "with_dem": False, "field_mode": "on_demand"}}
    assert world_kwargs_from_config(cfg) == {"root": "x", "with_dem": False}


def test_startup_builds_once_and_plan_always_passes_field(startup) -> None:
    ds, field, plan, calls = startup
    ctx = pc.create_context({"world": {"root": "x"}, "params": {}})

    assert ctx.dataset is ds and ctx.field is field and ctx.field_ready
    assert ctx.field_mode == defaults.WORLD_FIELD_MODE_STARTUP
    assert len(calls["load"]) == 1 and len(calls["build"]) == 1

    out = ctx.plan((1.0, 2.0), (3.0, 4.0), 25.0, goal_z_m=60.0)
    assert out is plan
    args, kwargs = calls["plan"][0]
    assert args[3:] == ((1.0, 2.0), (3.0, 4.0), 25.0, ctx.params)
    assert kwargs["world_field"] is field
    assert kwargs["goal_z_m"] == 60.0
    assert kwargs["buildings_total"] == ds.n_buildings

    # 第二条请求：不再建场，仍固定带同一份场
    ctx.plan((1.0, 2.0), (3.0, 4.0), 25.0)
    assert len(calls["build"]) == 1 and len(calls["plan"]) == 2
    assert calls["plan"][1][1]["world_field"] is field


def test_on_demand_skips_build_and_passes_none(startup) -> None:
    _ds, _field, _plan, calls = startup
    ctx = pc.create_context({"world": {"root": "x", "field_mode": "on_demand"}})

    assert ctx.field is None and not ctx.field_ready
    assert ctx.field_mode == defaults.WORLD_FIELD_MODE_ON_DEMAND
    assert len(calls["build"]) == 0
    ctx.plan((0.0, 0.0), (1.0, 1.0), 25.0)
    assert calls["plan"][0][1]["world_field"] is None


def test_cli_mode_overrides_config(startup) -> None:
    _ds, _field, _plan, calls = startup
    ctx = pc.create_context(
        {"world": {"root": "x", "field_mode": "startup"}},
        field_mode=defaults.WORLD_FIELD_MODE_ON_DEMAND,
    )
    assert ctx.field is None and len(calls["build"]) == 0


def test_create_context_argument_rules(startup) -> None:
    with pytest.raises(ValueError, match="不要再给"):
        pc.create_context({"world": {"root": "x"}}, world_kwargs={"root": "y"})
    with pytest.raises(ValueError, match="world_kwargs"):
        pc.create_context()
    with pytest.raises(ValueError, match="field_mode"):
        pc.create_context(world_kwargs={"root": "x"}, field_mode="sometimes")


def test_world_field_kwargs_derives_aligned_pad(startup) -> None:
    ds = _FakeDataset()
    params = RouteSegmentParams()
    step = min(params.spacing_m[:2])
    expected = required_pad_m(
        params.roi, params.spacing_m, defaults.WORLD_FIELD_MAX_OD_LENGTH_M,
        extra_m=defaults.WORLD_FIELD_PAD_EXTRA_M,
        min_pad_m=defaults.WORLD_FIELD_PAD_MIN_M,
    )
    kw = pc.world_field_kwargs(ds, params)
    assert kw["pad_m"] == expected
    assert kw["pad_m"] % step == 0          # 格距整数倍：格网原点不平移
    assert kw["bounds_m"] == ds.bounds_m and kw["spacing_m"] == params.spacing_m

    wide = replace(params, roi=replace(params.roi, margin_m=200.0))
    wide_pad = pc.world_field_kwargs(ds, wide)["pad_m"]
    assert wide_pad > expected and wide_pad % step == 0
    assert pc.world_field_kwargs(ds, params, 6000.0)["pad_m"] > expected


def test_max_od_length_is_configurable_and_validated(startup) -> None:
    cfg = {"world": {"root": "x", "max_od_length_m": 1200.0}}
    ctx = pc.create_context(cfg)
    assert ctx.max_od_length_m == 1200.0
    assert ctx.field_pad_m < pc.world_field_kwargs(ctx.dataset, ctx.params)["pad_m"]
    with pytest.raises(ValueError, match="max_od_length_m"):
        pc.create_context({"world": {"root": "x", "max_od_length_m": None}})
    with pytest.raises(ValueError, match="正的有限数"):
        pc.create_context({"world": {"root": "x"}}, max_od_length_m=-1.0)


def test_startup_self_check_runs_before_returning(startup, monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(pc, "check_contract_coverage",
                        lambda *a, **k: seen.append((a[1], a[4])))
    ctx = pc.create_context({"world": {"root": "x", "max_od_length_m": 2000.0}})
    assert seen == [(ctx.dataset.bounds_m, 2000.0)]


def test_max_od_is_capped_by_city_diagonal(startup) -> None:
    """配天文数字不会让 pad / 内存无界放大；返回的是实际生效的契约值。"""
    ds = _FakeDataset()
    diag = math.hypot(ds.bounds_m[2] - ds.bounds_m[0], ds.bounds_m[3] - ds.bounds_m[1])
    ctx = pc.create_context({"world": {"root": "x", "max_od_length_m": 100000.0}})
    assert ctx.max_od_length_m == diag
    assert ctx.field_pad_m == pc.world_field_kwargs(ds, ctx.params, diag)["pad_m"]
    # 端点都在 bounds 内，对角线长度本身就是合法 OD，不该被自己的契约拒掉
    out = ctx.plan((0.0, 0.0), (6000.0, 6000.0), 25.0)
    assert out is not None


def test_degenerate_bounds_fail_at_startup(startup, monkeypatch) -> None:
    flat = _FakeDataset()
    flat.bounds_m = (0.0, 0.0, 0.0, 0.0)
    monkeypatch.setattr(pc, "load_city", lambda **_kw: flat)
    with pytest.raises(ValueError, match="退化"):
        pc.create_context({"world": {"root": "x"}})


def test_context_exposes_dataset_digest_and_field_source(startup) -> None:
    ctx = pc.create_context({"world": {"root": "x"}})
    assert len(ctx.dataset_digest) == 12
    assert ctx.field_source == "built"
    assert pc.create_context({"world": {"root": "x"}}).dataset_digest == ctx.dataset_digest
    ctx_demand = pc.create_context({"world": {"root": "x", "field_mode": "on_demand"}})
    assert ctx_demand.field_source == "none"


def test_startup_can_load_artifact_instead_of_building(startup, monkeypatch) -> None:
    _ds, field, _plan, calls = startup
    seen: dict = {}

    def fake_load(path, **kwargs):
        seen.update(kwargs)
        seen["path"] = path
        return field, {"schema": 1}

    monkeypatch.setattr(pc, "load_field_artifact", fake_load)
    ctx = pc.create_context({"world": {"root": "x"}}, field_artifact="artifacts/gz")

    assert ctx.field is field and ctx.field_source == "artifact"
    assert len(calls["build"]) == 0                    # 启动不再付 120 s 建场
    assert seen["path"] == "artifacts/gz"
    assert seen["dataset_digest"] == ctx.dataset_digest
    assert seen["contract"]["max_od_length_m"] == ctx.max_od_length_m
    assert seen["contract"]["pad_m"] == ctx.field_pad_m
    assert seen["contract"]["bounds_m"] == ctx.bounds_m


def test_artifact_is_rejected_in_on_demand(startup) -> None:
    with pytest.raises(ValueError, match="field_artifact"):
        pc.create_context(
            {"world": {"root": "x", "field_mode": "on_demand"}}, field_artifact="a"
        )

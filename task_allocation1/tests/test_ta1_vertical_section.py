"""垂直切面距离尺子：可配置项逐条验证（不依赖任何 GIS 依赖）。"""

from __future__ import annotations

import math

import pytest

import task_allocation1 as ta1


def _flat(xs, ys):
    return [0.0] * len(xs)


def _wall(top=60.0, lo=200.0, hi=260.0):
    def sampler(xs, ys):
        return [top if lo <= x <= hi else 0.0 for x in xs]

    return sampler


def _cfg(**kw):
    base = {"model": "vertical_section", "agl_min_m": 10.0, "agl_max_m": 120.0,
            "clearance_m": 5.0, "climb_rate": 0.15, "num_samples": 200}
    base.update(kw)
    return ta1.CostConfig(**base)


def test_flat_ground_is_exactly_straight_line() -> None:
    """平地、无建筑：切面航程必须逐位等于三维直线（退化口径）。"""
    fn = ta1.build_distance_fn(_cfg(), ground_z=_flat)
    a, b = (0.0, 0.0, 40.0), (1000.0, 0.0, 40.0)
    assert fn(a, b) == pytest.approx(ta1.euclidean3d(a, b), abs=1e-9)
    a, b = (0.0, 0.0, 40.0), (1000.0, 300.0, 25.0)
    assert fn(a, b) == pytest.approx(ta1.euclidean3d(a, b), abs=1e-9)


def test_building_forced_climb_is_longer_but_not_absurd() -> None:
    fn = ta1.build_distance_fn(_cfg(), ground_z=_flat, obstacle_top_z=_wall())
    a, b = (0.0, 0.0, 20.0), (1000.0, 0.0, 20.0)
    d = fn(a, b)
    assert d > 1000.0
    assert d < 1050.0


def test_clearance_and_band_raise_distance_monotonically() -> None:
    a, b = (0.0, 0.0, 20.0), (1000.0, 0.0, 20.0)
    d_small = ta1.build_distance_fn(_cfg(), ground_z=_flat, obstacle_top_z=_wall())(a, b)
    d_clear = ta1.build_distance_fn(_cfg(clearance_m=30.0), ground_z=_flat,
                                    obstacle_top_z=_wall())(a, b)
    d_band = ta1.build_distance_fn(_cfg(agl_min_m=80.0), ground_z=_flat,
                                   obstacle_top_z=_wall())(a, b)
    assert d_small < d_clear
    assert d_small < d_band


def test_gentler_climb_rate_spreads_the_ramp_and_shortens_the_estimate() -> None:
    """坡够长时，同样的升高摊得更开，只多一点点长度；坡太陡则更费米数。

    （坡太缓而航段太短时会来不及回落，反而偏长——这是代理的已知边界，不在本条断言内。）
    """
    a, b = (0.0, 0.0, 20.0), (4000.0, 0.0, 20.0)
    wall = _wall(top=60.0, lo=1900.0, hi=2100.0)
    steep = ta1.build_distance_fn(_cfg(climb_rate=0.5), ground_z=_flat,
                                  obstacle_top_z=wall)(a, b)
    gentle = ta1.build_distance_fn(_cfg(climb_rate=0.05), ground_z=_flat,
                                   obstacle_top_z=wall)(a, b)
    assert gentle < steep


def test_over_band_mode_controls_infeasibility() -> None:
    """带外必须爬（楼 200 m > 带顶 120 m）：allow 给有限米数，infeasible 给 inf。"""
    a, b = (0.0, 0.0, 20.0), (1000.0, 0.0, 20.0)
    wall = _wall(top=200.0)
    allow = ta1.build_distance_fn(_cfg(), ground_z=_flat, obstacle_top_z=wall)(a, b)
    hard = ta1.build_distance_fn(_cfg(over_band_mode="infeasible"), ground_z=_flat,
                                 obstacle_top_z=wall)(a, b)
    assert math.isfinite(allow)
    assert hard == math.inf


def test_cache_is_ordered_and_hits_on_repeat() -> None:
    calls = {"n": 0}

    def counting(xs, ys):
        calls["n"] += 1
        return [0.0] * len(xs)

    fn = ta1.build_distance_fn(_cfg(), ground_z=counting)
    a, b = (0.0, 0.0, 40.0), (100.0, 0.0, 40.0)
    fn(a, b)
    assert calls["n"] == 1
    fn(a, b)
    assert calls["n"] == 1                      # 同向重复：命中
    fn(b, a)
    assert calls["n"] == 2                      # 反向：单独一个键（方向相关）
    assert len(fn.cache) == 2


def test_bad_sampler_output_is_rejected() -> None:
    fn = ta1.build_distance_fn(_cfg(), ground_z=lambda xs, ys: [0.0])
    with pytest.raises(ValueError, match="逐点一一对应"):
        fn((0.0, 0.0, 20.0), (1000.0, 0.0, 20.0))


def test_nan_samples_are_interpolated_not_crashed() -> None:
    """采样器给 NaN（DEM 空洞）：填平后继续算，而不是把整条腿变 NaN。"""
    def holey(xs, ys):
        return [float("nan") if 400 <= x <= 600 else 0.0 for x in xs]

    fn = ta1.build_distance_fn(_cfg(), ground_z=holey)
    d = fn((0.0, 0.0, 20.0), (1000.0, 0.0, 20.0))
    assert math.isfinite(d)
    assert d == pytest.approx(1000.0, abs=1e-9)   # 空洞按插值=0 地面，低于飞行带 → 不影响


def test_config_validation_errors() -> None:
    with pytest.raises(ValueError, match="不认识"):
        _cfg(model="astar").validate()
    with pytest.raises(ValueError, match="agl_max_m"):
        _cfg(agl_min_m=120.0, agl_max_m=10.0).validate()
    with pytest.raises(ValueError, match="climb_rate"):
        _cfg(climb_rate=0.0).validate()
    with pytest.raises(ValueError, match="num_samples"):
        _cfg(num_samples=1).validate()
    with pytest.raises(ValueError, match="over_band_mode"):
        _cfg(over_band_mode="clip").validate()


def test_vertical_section_without_samplers_is_an_error() -> None:
    with pytest.raises(ValueError, match="至少需要一个采样器"):
        ta1.build_distance_fn(_cfg())


def test_euclid_model_returns_plain_euclidean() -> None:
    fn = ta1.build_distance_fn(ta1.CostConfig())
    assert fn((0.0, 0.0, 0.0), (3.0, 4.0, 0.0)) == 5.0
    assert fn is ta1.euclidean3d


def test_allocate_uses_cost_config_through_request() -> None:
    """请求级配置真正生效：同一批实体，换尺子 path_m 必须变大、航次结构不变。"""
    def build(cost_config, **extra):
        return ta1.AllocateRequest(
            tasks=[ta1.Task(id=1, x=1000.0, y=0.0, reward=1.0),
                   ta1.Task(id=2, x=1200.0, y=0.0, reward=1.0)],
            uavs=[ta1.UAV(id=0, x=0.0, y=0.0, z=20.0, speed=15.0,
                          remaining_range=9000.0, max_time=2400.0, nest_id=0)],
            nests=[ta1.Nest(id=0, x=0.0, y=0.0, z=20.0, capacity=2)],
            cost_config=cost_config,
            solver_config=ta1.SolverConfig(pop_size=8, max_generations=4, seed=1),
            **extra,
        )

    plain = ta1.allocate(build(ta1.CostConfig()))
    walled = ta1.allocate(build(_cfg(), ground_z=_flat, obstacle_top_z=_wall()))
    assert plain.solution.executed_task_ids == walled.solution.executed_task_ids
    assert walled.metrics_dict()["path_m"] > plain.metrics_dict()["path_m"]

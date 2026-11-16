"""跨层绑定的回归测试：参数是否只有一处来源、结果是否只有一种形状。

这些断言存在的理由：改造前 ``RouteSegmentParams`` / ``RRTConfig`` 各自抄了一份 RRT 默认值，
``clearance_m`` 在 ``VehicleLimits`` 和 ``RouteSegmentParams`` 两处各定义一次，而契约测试只断言了
"``PlannerParams`` 没有 clearance"，于是"净空只在一处"这条不变量早就破了却没人发现。
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
from path_planning1.contract import (
    DEFAULT_ALT_MAX_M,
    DEFAULT_ALT_MIN_M,
    Building,
    FailureCode,
    GridSpec,
    LocalFrame,
    PlannerParams,
    RegionOfInterest,
    Status,
    VehicleLimits,
    WorldSnapshot,
)
from path_planning1.core.city_dataset import load_city
from path_planning1.core.corridor_window import z_envelope_from_ground
from path_planning1.core.params_fingerprint import params_hash
from path_planning1.core.route_segment_planner import RouteSegmentParams, plan_route_segment
from path_planning1.core.rrt_path_search import RRTConfig


def _frame() -> LocalFrame:
    return LocalFrame(lon0=113.0, lat0=23.0, lat_ref=23.05)


# ---------- 参数单一来源 ----------


def test_clearance_lives_only_in_vehicle_limits() -> None:
    names = {f.name for f in dataclasses.fields(RouteSegmentParams)}
    assert "clearance_m" not in names            # 只有 property，不是字段
    assert RouteSegmentParams().clearance_m == VehicleLimits().clearance_m
    tightened = dataclasses.replace(RouteSegmentParams(), vehicle=VehicleLimits(clearance_m=9.0))
    assert tightened.clearance_m == 9.0           # 改车辆就生效，改不到"第二份默认值"
    # RRT 阈值 = 净空 + SDF 竖直量化容差（= sz），两者都不是第二份净空默认值
    assert tightened.rrt_config().clearance_m == 9.0 + tightened.sdf_tolerance_m


def test_route_segment_params_holds_no_algorithm_knobs() -> None:
    """算法旋钮只在 PlannerParams：RouteSegmentParams 里不该再出现同名默认值。"""
    knobs = {
        "step_m", "min_step_m", "max_iters", "goal_sample_rate", "sdf_step_scale",
        "rrt_step_m", "rrt_max_iters", "rrt_goal_sample_rate",
    }
    assert knobs.isdisjoint({f.name for f in dataclasses.fields(RouteSegmentParams)})


def test_rrt_config_is_derived_not_copied() -> None:
    planner = PlannerParams()
    cfg = RRTConfig.from_planner(planner, clearance_m=5.0, seed=7)
    assert cfg.step_m == planner.rrt_step_m
    assert cfg.min_step_m == planner.rrt_min_step_m
    assert cfg.max_iters == planner.rrt_max_iters
    assert cfg.goal_sample_rate == planner.rrt_goal_sample_rate
    assert cfg.seed == 7


def test_flight_band_defaults_have_one_source() -> None:
    assert RouteSegmentParams().agl_min_m == DEFAULT_ALT_MIN_M
    assert RouteSegmentParams().agl_max_m == DEFAULT_ALT_MAX_M
    world = WorldSnapshot(
        world_id="w", frame=_frame(), buildings=[],
        grid=GridSpec(origin_enu=(0.0, 0.0, 0.0), spacing_m=(4.0, 4.0, 2.0), shape=(10, 10, 20)),
    )
    assert (world.alt_min_m, world.alt_max_m) == (DEFAULT_ALT_MIN_M, DEFAULT_ALT_MAX_M)


def test_z_envelope_single_formula() -> None:
    assert z_envelope_from_ground(12.5, 10.0, 120.0) == (22.5, 132.5)


def test_params_hash_tracks_every_knob() -> None:
    kwargs: dict[str, object] = {
        "planner": PlannerParams(),
        "vehicle": VehicleLimits(),
        "roi": RegionOfInterest(),
        "spacing_m": (4.0, 4.0, 2.0),
        "agl_range_m": (10.0, 120.0),
        "seed": 0,
    }
    base = params_hash(**kwargs)  # type: ignore[arg-type]
    variants: list[dict[str, object]] = [
        {"planner": PlannerParams(rrt_step_m=50.0)},
        {"vehicle": VehicleLimits(clearance_m=7.5)},
        {"roi": RegionOfInterest(width_ratio=0.1)},
        {"spacing_m": (8.0, 8.0, 2.0)},
        {"agl_range_m": (10.0, 90.0)},
        {"seed": 1},
        {"clearance_search_radius_m": 20.0},   # 只改"报多远"，但也是另一个结果
    ]
    for variant in variants:
        assert params_hash(**{**kwargs, **variant}) != base  # type: ignore[arg-type]
    assert params_hash(**kwargs) == base          # 同参同值


def test_params_hash_covers_implementation_constants() -> None:
    """RRT 实现细节旋钮也改变输出，必须改变 hash —— 否则改口径会静默命中旧缓存。

    这些常数以前在 ``params_fingerprint.py`` 里按模块常量硬读，改 ``PlannerParams`` 不生效、
    改 defaults 又要重导入；现在它们是 ``PlannerParams`` 的字段，走同一条可配 + 可进指纹的路。
    """
    kwargs: dict[str, object] = {
        "planner": PlannerParams(),
        "vehicle": VehicleLimits(),
        "roi": RegionOfInterest(),
        "spacing_m": (4.0, 4.0, 2.0),
        "agl_range_m": (10.0, 120.0),
        "seed": 0,
    }
    base = params_hash(**kwargs)  # type: ignore[arg-type]
    for field_name, value in (
        ("rrt_shortcut_iters", 500),
        ("clearance_sample_step_m", 0.5),
    ):
        varied = {**kwargs, "planner": dataclasses.replace(PlannerParams(), **{field_name: value})}
        assert params_hash(**varied) != base, field_name  # type: ignore[arg-type]


# ---------- 结果唯一形状 ----------


def _run(alt: float = 25.0, buildings: list[Building] | None = None,
         spacing: tuple[float, float, float] = (20.0, 20.0, 20.0)):
    """小场景（空场地 + 粗栅格），只为验证形状与失败码，不依赖 data 目录。"""
    params = RouteSegmentParams(
        roi=RegionOfInterest(width_ratio=0.4, margin_m=20.0),
        spacing_m=spacing,
        agl_max_m=120.0,
    )
    return plan_route_segment(
        buildings or [], _frame(), None,
        (0.0, 0.0), (400.0, 0.0), alt, params,
    )


def test_plan_route_segment_shape_and_types() -> None:
    res = _run()
    assert res.status is Status.OK and res.failure_code is FailureCode.NONE
    assert res.metrics.length_m is not None and res.metrics.min_clearance_m is not None
    # 时间域层接上后 duration_s 有值（最短飞行时间），但能耗仍然为空——没有风场，不编造
    assert res.metrics.duration_s is not None and res.metrics.duration_s > 0
    assert res.metrics.energy_j is None
    assert isinstance(res.stats.points, int) and isinstance(res.stats.buildings_in_window, int)
    assert isinstance(res.stats.window_cells, int)
    assert res.params_hash and len(res.params_hash) == 12
    flat = res.metrics_dict()
    assert json.dumps(flat, ensure_ascii=False)  # 可序列化（summary/CLI 都要用）
    assert flat["status"] == "ok" and flat["points"] == res.stats.points


def test_trajectory_is_time_optimal_and_within_limits() -> None:
    """轨迹层要满足的事：端点固定、速度/加速度不超限、时长 ≥ 直线下界、两端悬停。

    "时长 ≥ 直线下界"（400 m / v_max）是**独立于求解器**的物理下界：任何可行轨迹都不可能比
    "以最高速走直线"更快。它挡住的是"求解器返回了一个漂亮但没意义的时间剖面"。
    """
    res = _run()
    traj = res.trajectory
    assert traj is not None, res.diagnostics.warnings
    assert traj.position_enu.shape == traj.velocity_enu.shape == traj.acceleration_enu.shape
    assert np.allclose(traj.position_enu[0], res.path_enu[0], atol=1e-6)
    assert np.allclose(traj.position_enu[-1], res.path_enu[-1], atol=1e-6)
    assert np.linalg.norm(traj.velocity_enu[0]) < 1e-6   # 起点悬停
    assert np.linalg.norm(traj.velocity_enu[-1]) < 1e-6  # 终点悬停
    v = np.linalg.norm(traj.velocity_enu, axis=1)
    a = np.linalg.norm(traj.acceleration_enu, axis=1)
    assert v.max() <= 20.0 + 1e-6
    assert a.max() <= 3.0 + 1e-3
    straight = float(np.linalg.norm(res.path_enu[-1] - res.path_enu[0]))
    assert traj.t_s[-1] >= straight / 20.0
    assert abs(traj.t_s[-1] - (len(traj.t_s) - 1) * traj.dt_uniform_s) < 1e-9
    assert res.metrics.max_airspeed_mps == pytest.approx(float(v.max()), abs=1e-6)


def test_int_and_float_configs_share_params_hash() -> None:
    """配置从 JSON 读进来常是整数（``[4,4,2]`` / ``[10,120]``），默认值是浮点。

    指纹必须把它们归一：否则同一份配置算出两个 hash，缓存永远不命中、日志里两个键各说各话。
    这条断言存在的理由就是本轮真踩过——同一 OD 同一高度，CLI 与 JSON 配置给出不同 hash。
    """
    floats = params_hash(
        planner=PlannerParams(), vehicle=VehicleLimits(), roi=RegionOfInterest(width_ratio=0.2, margin_m=50.0),
        spacing_m=(4.0, 4.0, 2.0), agl_range_m=(10.0, 120.0), seed=0,
    )
    ints = params_hash(
        planner=PlannerParams(), vehicle=VehicleLimits(), roi=RegionOfInterest(width_ratio=0.2, margin_m=50.0),
        spacing_m=(4, 4, 2), agl_range_m=(10, 120), seed=0,
    )
    assert floats == ints


def test_bad_numbers_raise_immediately() -> None:
    """坏数值（NaN 坐标、零/负间距）是调用方 bug → ValueError，0 秒、不建窗口。

    为什么要断言"不是 failure_code"：``OUT_OF_FLIGHT_BAND`` 会被后端映射成"换个高度重试"，
    而 NaN 坐标换高度也没用——它是 JSON 里给了 ``null`` 或单位算错。
    """
    params = RouteSegmentParams(spacing_m=(20.0, 20.0, 20.0))
    with pytest.raises(ValueError, match="start_xy"):
        plan_route_segment([], _frame(), None, (float("nan"), 0.0), (400.0, 0.0), 25.0, params)
    with pytest.raises(ValueError, match="spacing_m"):
        plan_route_segment(
            [], _frame(), None, (0.0, 0.0), (400.0, 0.0), 25.0,
            RouteSegmentParams(spacing_m=(0.0, 20.0, 20.0)),
        )
    # 水平重合原来由 corridor_window 抛英文 ValueError，现在统一走入口自检
    with pytest.raises(ValueError, match="水平重合"):
        plan_route_segment([], _frame(), None, (0.0, 0.0), (0.0, 0.0), 25.0, params)


def test_endpoints_carry_their_own_agl() -> None:
    """两端各给真高（参考实现的口径就是 3D 端点，两个点各定 z）。

    平坦场景（``dem=None``，地面=0）下看绝对 z：起点=起点真高、终点=终点真高，
    而不是整段共用一个高度。省略终点真高时退回"两端同高"（机巢起降的常见情形）。
    """
    params = RouteSegmentParams(
        roi=RegionOfInterest(width_ratio=0.4, margin_m=20.0),
        spacing_m=(20.0, 20.0, 20.0),
        agl_max_m=120.0,
    )
    res = plan_route_segment(
        [], _frame(), None, (0.0, 0.0), (400.0, 0.0), 20.0, params, goal_z_m=60.0
    )
    assert res.status is Status.OK, res.message
    assert res.path_enu is not None
    assert res.path_enu[0][2] == pytest.approx(20.0, abs=1e-6)
    assert res.path_enu[-1][2] == pytest.approx(60.0, abs=1e-6)

    same = plan_route_segment([], _frame(), None, (0.0, 0.0), (400.0, 0.0), 25.0, params=params)
    assert same.path_enu is not None
    assert same.path_enu[0][2] == pytest.approx(25.0, abs=1e-6)
    assert same.path_enu[-1][2] == pytest.approx(25.0, abs=1e-6)


def test_plan_route_segment_reports_out_of_band() -> None:
    res = _run(alt=200.0)
    assert res.status is Status.INFEASIBLE
    assert res.failure_code is FailureCode.OUT_OF_FLIGHT_BAND
    assert res.path_enu is None
    # 纯入参越界不该建窗口/算 SDF：没有 grid_s 计时，也没花掉时间
    assert "grid_s" not in res.timings
    assert res.timings["total_s"] < 0.05
    assert res.stats.window_cells == 0        # 空统计，不是"算完才发现越界"


def test_plan_route_segment_reports_start_blocked() -> None:
    tall = Building(
        footprint=((-40.0, -40.0), (40.0, -40.0), (40.0, 40.0), (-40.0, 40.0)),
        height_m=150.0,
    )
    res = _run(buildings=[tall])
    assert res.status is Status.INFEASIBLE
    assert res.failure_code is FailureCode.START_OR_GOAL_BLOCKED


def test_shortcut_budget_zero_keeps_a_valid_path() -> None:
    """``shortcut_iters=0`` 只能"不做捷径"，不能"跳过校验直接连终点"。

    老实现预算为 0 时把路径退化成 ``[start, goal]`` 直线：穿楼而过却报 ``ok``，
    实测净空 -11.70 m（OD 0-1，alt 25）。现在要么返回原始 RRT 路径（每条边都校验过），
    要么由 ``PATH_CLEARANCE_VIOLATED`` 明确失败——不允许"ok + 负净空"。
    """
    wall = Building(
        footprint=((180.0, -10.0), (220.0, -10.0), (220.0, 10.0), (180.0, 10.0)),
        height_m=40.0,
    )
    params = RouteSegmentParams(
        roi=RegionOfInterest(width_ratio=0.6, margin_m=20.0),
        spacing_m=(20.0, 20.0, 20.0),
        agl_max_m=120.0,
        planner=dataclasses.replace(PlannerParams(), rrt_shortcut_iters=0, rrt_max_iters=4000),
    )
    res = plan_route_segment([wall], _frame(), None, (0.0, 0.0), (400.0, 0.0), 25.0, params)
    assert res.status is Status.OK, res.message
    assert res.metrics.min_clearance_m is not None
    assert res.metrics.min_clearance_m >= params.clearance_m
    assert res.metrics.length_m > 400.0        # 绕过墙，不是穿墙的直线
    assert res.stats.points >= 2


# ---------- 数据加载入口 ----------


def _fake_city(tmp_path, *, with_dem: bool):
    region = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [[
                [113.0, 23.0], [113.02, 23.0], [113.02, 23.02], [113.0, 23.02], [113.0, 23.0],
            ]]},
        }],
    }
    (tmp_path / "region.geojson").write_text(json.dumps(region), encoding="utf-8")
    building = {"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {"Height": 20.0},
        "geometry": {"type": "Polygon", "coordinates": [[
            [113.005, 23.005], [113.010, 23.005], [113.010, 23.010],
            [113.005, 23.010], [113.005, 23.005],
        ]]},
    }]}
    (tmp_path / "building").mkdir()
    (tmp_path / "building" / "b.geojson").write_text(json.dumps(building), encoding="utf-8")
    return tmp_path


def test_load_city_dem_is_required_unless_opted_out(tmp_path) -> None:
    root = _fake_city(tmp_path, with_dem=True)
    with pytest.raises(FileNotFoundError):
        load_city(root, with_dem=True)               # 缺 dem.tif 就明确报，不静默按平地
    ds = load_city(root, with_dem=False)
    assert ds.terrain_applied is False and ds.n_buildings == 1


def test_geometry_clearance_gate_never_passes_a_bad_path(monkeypatch) -> None:
    """终检闸：真几何净空不达标必须报 PATH_CLEARANCE_VIOLATED，绝不静默给 ok。

    这道闸是全库最后一道防线（README 说它抓出过真实缺陷），但它一直只在注释里出现，
    没有任何回归断言。这里直接把真几何读数打到 0，锁死"不达标就不交付"。
    """
    from path_planning1.core.clearance_check import BuildingPrisms

    monkeypatch.setattr(
        BuildingPrisms, "worst",
        lambda self, path, step: (0.0, np.asarray(path[0], dtype=float)),
    )
    res = _run()
    assert res.failure_code is FailureCode.PATH_CLEARANCE_VIOLATED
    assert res.status is Status.INFEASIBLE and res.path_enu is None
    assert res.trajectory is None and "真几何净空" in res.message


def test_total_s_is_wall_clock_not_sum_of_segments() -> None:
    """分段之间还有未计时的活儿，分段和会少报；后端把 total_s 当延迟预算用。"""
    res = _run()
    parts = sum(v for k, v in res.timings.items() if k != "total_s")
    assert res.timings["total_s"] >= parts - 1e-9
    assert res.diagnostics.elapsed_s == pytest.approx(res.timings["total_s"])

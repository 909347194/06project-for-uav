"""两条"以前只是纸面"的约束现在真的生效了：推力上限、走廊半宽。

背景（三条已知缺口里的两条）：

* NLP 以前只有 ``‖v‖≤v_max``、``‖a‖≤a_max``，``mass_kg``/``thrust_max_n`` 没有消费者；
  文档里"5 kg、3 m/s² ⇒ 15 N ≪ 100 N"还漏了重力（真实是 51–64 N）。
* 走廊 ``half_width`` 只用来定窗口大小（``lateral = half + margin``），交付轨迹可以飘到
  ``lateral`` 之外；``stats.max_lateral_m`` 量的还是 RRT 折线。

这两条的判据都落在"结果能被读出"上：``max_thrust_n``/``thrust_ratio`` 与
``max_lateral_m``/``inside_corridor``，不是靠文档嘴说。
"""

from __future__ import annotations

import numpy as np
import pytest
from path_planning1.contract import (
    Building,
    FailureCode,
    LocalFrame,
    RegionOfInterest,
    Status,
    VehicleLimits,
)
from path_planning1.core.route_segment_planner import RouteSegmentParams, plan_route_segment


def _frame() -> LocalFrame:
    return LocalFrame(lon0=113.0, lat0=23.0, lat_ref=23.05)


def _plan(buildings: list[Building], *, veh: VehicleLimits | None = None, ratio: float = 0.6,
          spacing: tuple[float, float, float] = (10.0, 10.0, 10.0),
          goal_z_m: float | None = None):
    params = RouteSegmentParams(
        roi=RegionOfInterest(width_ratio=ratio, margin_m=20.0),
        spacing_m=spacing,
        agl_max_m=120.0,
        vehicle=veh or VehicleLimits(),
    )
    return plan_route_segment(
        buildings, _frame(), None, (0.0, 0.0), (400.0, 0.0), 25.0, params,
        goal_z_m=goal_z_m,
    )


# ---------- 推力 ----------


def test_vehicle_that_cannot_hover_is_a_config_error() -> None:
    """``T_max < m·g`` 连悬停都不成立：0 秒 ValueError，而不是算完报 ok。"""
    with pytest.raises(ValueError, match="不自洽"):
        _plan([], veh=VehicleLimits(thrust_max_n=40.0, mass_kg=5.0))
    with pytest.raises(ValueError, match="thrust_max_n"):
        _plan([], veh=VehicleLimits(thrust_max_n=-1.0))


def test_thrust_is_reported_and_matches_force_balance() -> None:
    """基线机型（5 kg/100 N）离上限还远：报出的推力必须 ≤ 上限，且与力平衡一致。"""
    res = _plan([], goal_z_m=60.0)
    traj = res.trajectory
    assert traj is not None, res.message
    g = np.array([0.0, 0.0, 9.81])
    want = 5.0 * float(np.max(np.linalg.norm(traj.acceleration_enu + g, axis=1)))
    assert res.stats.max_thrust_n == pytest.approx(want, rel=1e-9)
    assert res.stats.thrust_ratio == pytest.approx(want / 100.0, rel=1e-9)
    assert res.stats.thrust_ratio < 0.8          # 实测 64% 量级，不是"15 N"那种错数


def test_thrust_constraint_bites_when_vehicle_is_heavy() -> None:
    """重载机型（9.5 kg/100 N ⇒ 单位推力 10.5 m/s²）：爬升段必须被推力咬住。

    ``‖a+g‖ ≤ T_max/m`` 是真正的不等式；只算 ``m‖a‖`` 的实现会给出 3 m/s² 爬升、
    推力 118 N 的假解，这条断言就是拦它的。
    """
    heavy = VehicleLimits(mass_kg=9.5, thrust_max_n=100.0)
    res = _plan([], veh=heavy, goal_z_m=60.0)
    assert res.trajectory is not None, res.message
    assert res.stats.max_thrust_n <= 100.0 + 1e-6
    assert res.stats.thrust_ratio <= 1.0 + 1e-9
    # 竖直加速度被压到 √((T/m)²−a_h²)−g 附近：3 m/s² 满爬升不再可行
    assert float(np.max(res.trajectory.acceleration_enu[:, 2])) < 3.0


# ---------- 走廊 ----------


def test_trajectory_stays_inside_half_width_even_if_path_pokes_out() -> None:
    """200 m 高的墙把"爬过去"这条路堵死，只能横向绕：折线绕到 58 m，轨迹被拉回 50 m 内。

    这条同时证明两件事：``half_width`` 真的进了解（轨迹 < 折线），且
    ``max_lateral_m`` / ``path_max_lateral_m`` 是两个不同口径——以前只有一个数，报的是折线。
    """
    wall = Building(
        footprint=((180.0, -30.0), (220.0, -30.0), (220.0, 30.0), (180.0, 30.0)),
        height_m=200.0,
    )
    res = _plan([wall], ratio=0.25, spacing=(6.0, 6.0, 6.0))
    assert res.trajectory is not None, res.diagnostics.warnings
    half = res.stats.corridor_width_m / 2.0
    assert half == pytest.approx(50.0, abs=1e-6)
    assert res.stats.inside_corridor is True
    assert res.stats.max_lateral_m <= half + 1e-6
    assert res.stats.path_max_lateral_m > res.stats.max_lateral_m + 1.0   # 折线确实更靠外
    assert res.stats.max_lateral_m > 1.0                                  # 确实绕了，不是直穿
    lateral = np.abs(res.trajectory.position_enu[:, 1])
    assert res.stats.max_lateral_m == pytest.approx(float(lateral.max()), abs=1e-6)


def test_corridor_too_narrow_for_clearance_reports_roi_no_path() -> None:
    """半宽 40 m、墙占 ±30 m、SDF 净空要 5+6 m：物理上就没有可交付轨迹。

    以前这会被当成"NLP 没收敛"降级：``status=ok`` 但没有 ``trajectory``，后端只能拿到一条
    出了半宽的折线。现在它是一条明确的业务失败——走廊是 route 的语义边界。
    """
    wall = Building(
        footprint=((180.0, -30.0), (220.0, -30.0), (220.0, 30.0), (180.0, 30.0)),
        height_m=200.0,
    )
    res = _plan([wall], ratio=0.2, spacing=(6.0, 6.0, 6.0))
    assert res.failure_code is FailureCode.ROI_NO_PATH
    assert res.status is Status.INFEASIBLE and res.path_enu is None
    assert "走廊半宽 40.0 m" in res.message and "margin_m 只垫栅格" in res.message


def test_max_lateral_is_trajectory_not_warm_start_path() -> None:
    """``max_lateral_m`` 必须是交付物的口径；折线的另报 ``path_max_lateral_m``。

    两者相等只说明这条航段没触发差异，所以这里断言的是"字段存在且都可读"，
    再加"轨迹值不超过半宽"这条硬事实——折线没有这条保证（它是内部初值）。
    """
    res = _plan([], ratio=0.2)
    assert res.trajectory is not None
    assert res.stats.path_max_lateral_m >= 0.0
    assert res.stats.max_lateral_m <= res.stats.corridor_width_m / 2.0 + 1e-6
    flat = res.metrics_dict()
    for key in ("inside_corridor", "path_max_lateral_m", "max_thrust_n", "thrust_ratio"):
        assert key in flat


def test_agl_and_z_stats_describe_the_delivered_trajectory() -> None:
    """真高/高度统计必须量轨迹（要飞的线），不能拿 RRT 折线的 z 顶替。

    40 m 墙的场景里折线为绕墙爬到 ~52 m、轨迹选择横向绕只到 ~26 m；旧实现报的是
    折线值，后端拿 agl_max_m 校验飞行带就会得到错误结论。
    """
    wall = Building(
        footprint=((180.0, -20.0), (220.0, -20.0), (220.0, 20.0), (180.0, 20.0)),
        height_m=40.0,
    )
    res = _plan([wall], ratio=0.5)
    assert res.trajectory is not None and res.path_enu is not None, res.message
    z = res.trajectory.position_enu[:, 2]
    assert res.stats.z_min_m == pytest.approx(float(z.min()), abs=1e-6)
    assert res.stats.z_max_m == pytest.approx(float(z.max()), abs=1e-6)
    # 该场景没有 DEM，地面 = 0，真高就是绝对 z
    assert res.stats.agl_min_m == pytest.approx(float(z.min()), abs=1e-6)
    assert res.stats.agl_max_m == pytest.approx(float(z.max()), abs=1e-6)
    assert res.path_enu[:, 2].max() > res.stats.z_max_m + 1.0    # 折线确实更高


def test_node_ladder_never_increases_node_count() -> None:
    """n_resample 允许小到 3；退让阶梯不能反而把 3 个节点"退"成 12 个。"""
    from path_planning1.core.trajectory_optimization import _ladder

    assert _ladder(60) == (40, 30)
    for n0 in (3, 4, 8, 10, 12, 20, 60):
        assert all(3 <= n < n0 for n in _ladder(n0)), n0

"""净空两套实现对账：``BuildingPrisms``（shapely 路线） vs 本目录的 ``reference_clearance``。

这些断言回答一个具体问题："交付给后端的净空数字，是只有一份代码背书，还是两套算法都认？"
第二套实现换的是**算 dxy 的路径**（逐边线段距离 + 自写射线判内），公式仍是棱柱真距离
``sqrt(dxy² + dz²)``——公式一致是必须的，独立性来自计算过程不同。
"""

from __future__ import annotations

import numpy as np
import pytest
from path_planning1.contract import Building, LocalFrame, RegionOfInterest
from path_planning1.core.clearance_check import BuildingPrisms
from path_planning1.core.route_segment_planner import RouteSegmentParams, plan_route_segment
from reference_clearance import reference_min_clearance, worst_point


def _box(x0: float, y0: float, x1: float, y1: float, height: float = 40.0,
         base: float = 0.0) -> Building:
    return Building(
        footprint=((x0, y0), (x1, y0), (x1, y1), (x0, y1)),
        height_m=height,
        base_z_m=base,
    )


def test_analytic_cases_above_beside_and_diagonal() -> None:
    """三种手算得出的位置：正上方、贴侧面、斜上方（后者必须合成而不是取小）。"""
    box = _box(-10.0, -10.0, 10.0, 10.0, height=40.0)
    prisms = BuildingPrisms([box])
    # 屋顶正上方 6 m：水平距 0（轮廓内），竖向间隔 6
    p_above = np.array([[[0.0, 0.0, 46.0]]])
    assert prisms.min_clearance(p_above[0]) == pytest.approx(6.0, abs=1e-9)
    # 西侧 3 m、z 落在 [0,40] 内：只有水平距离
    p_side = np.array([[13.0, 0.0, 20.0]])
    assert prisms.min_clearance(p_side) == pytest.approx(3.0, abs=1e-9)
    # 斜上方：水平 3、竖向 6 ⇒ sqrt(3²+6²)
    p_diag = np.array([[13.0, 0.0, 46.0]])
    assert prisms.min_clearance(p_diag) == pytest.approx(np.hypot(3.0, 6.0), abs=1e-9)
    # 参考实现给出同样的三个数（独立路径、同一口径）
    for pts, want in ((p_above[0], 6.0), (p_side, 3.0), (p_diag, np.hypot(3.0, 6.0))):
        assert reference_min_clearance(pts, [box]) == pytest.approx(want, abs=1e-9)


def test_reference_agrees_on_concave_footprint() -> None:
    """凹多边形（L 形）是两套实现最容易分家的地方：轮廓内有"缺口"，min 只能走边。"""
    l_shape = Building(
        footprint=((0.0, 0.0), (8.0, 0.0), (8.0, 3.0), (3.0, 3.0), (3.0, 8.0), (0.0, 8.0)),
        height_m=30.0,
    )
    notch = np.array([[6.0, 6.0, 15.0], [6.0, 6.0, 35.0], [1.5, 1.5, 15.0]])
    prisms = BuildingPrisms([l_shape])
    for i in range(len(notch)):
        one = notch[i : i + 1]
        assert prisms.min_clearance(one, 1.0) == pytest.approx(
            reference_min_clearance(one, [l_shape]), abs=1e-9
        )


def test_reference_agrees_on_random_boxes_and_wiggly_path() -> None:
    """随机方盒 + 折线批量对账：只允许浮点误差，不允许 0.1 m 级分歧。"""
    rng = np.random.default_rng(7)
    boxes: list[Building] = []
    for _ in range(24):
        cx, cy = rng.uniform(-150, 150, 2)
        w, h = rng.uniform(4, 20, 2)
        boxes.append(_box(cx - w, cy - h, cx + w, cy + h, height=float(rng.uniform(5, 60))))
    t = np.linspace(0.0, 1.0, 220)
    path = np.stack([300 * t, 60 * np.sin(6 * np.pi * t), 40 + 20 * t], axis=1)
    prisms = BuildingPrisms(boxes, 30.0)
    prod = prisms.min_clearance(path, 1.0)
    ref = reference_min_clearance(path, boxes, sample_step_m=1.0, search_radius_m=30.0)
    assert prod == pytest.approx(ref, abs=1e-9)
    # 最差点也要对得上：报"哪里贴太近"同样得两套实现认
    got_pt = prisms.worst(path, 1.0)[1]
    ref_val, ref_pt = worst_point(path, boxes, sample_step_m=1.0, search_radius_m=30.0)
    assert np.allclose(got_pt, ref_pt, atol=1e-9)
    assert ref_val == pytest.approx(prod, abs=1e-9)


def test_trajectory_clearance_matches_reference_end_to_end() -> None:
    """端到端：交付的 ``metrics.min_clearance_m``（轨迹、shapely 量）必须等于参考实现量的值。

    这条才对应"交付数字有没有第二套背书"——前几条只证明几何算子等价，这里证明
    "算出来的那条轨迹、那个数"也对得上。
    """
    wall = _box(180.0, -60.0, 220.0, 60.0, height=45.0)
    tower = _box(80.0, 40.0, 110.0, 70.0, height=60.0)
    params = RouteSegmentParams(
        roi=RegionOfInterest(width_ratio=0.6, margin_m=20.0),
        spacing_m=(10.0, 10.0, 10.0),
        agl_max_m=120.0,
    )
    res = plan_route_segment(
        [wall, tower], LocalFrame(lon0=113.0, lat0=23.0, lat_ref=23.05),
        None, (0.0, 0.0), (400.0, 0.0), 25.0, params,
    )
    assert res.status is not None and res.trajectory is not None, res.message
    prisms = BuildingPrisms([wall, tower], 30.0)
    produced = prisms.min_clearance(res.trajectory.position_enu, 1.0)
    reference = reference_min_clearance(res.trajectory.position_enu, [wall, tower])
    assert produced == pytest.approx(reference, abs=1e-9)
    assert res.metrics.min_clearance_m == pytest.approx(reference, abs=1e-9)

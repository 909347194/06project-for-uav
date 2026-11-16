"""绝对 z 入口：两种写法等价，真高由地面派生。"""

from __future__ import annotations

import numpy as np
import pytest
from path_planning1.contract import LocalFrame, Status
from path_planning1.core.dem_terrain import Dem
from path_planning1.core.route_segment_planner import plan_route_segment
from path_planning1.core.route_segment_xyz import plan_route_segment_xyz


def _flat_dem(z_m: float = 10.0) -> Dem:
    return Dem(
        z_m=np.full((20, 20), z_m, dtype=np.float32),
        lon_ul=113.3, lat_ul=23.12, dlon=1e-4, dlat=1e-4,
    )


def _frame() -> LocalFrame:
    return LocalFrame(lon0=113.3, lat0=23.12, lat_ref=23.12)


def test_xyz_entry_equals_xy_plus_z() -> None:
    """``(x, y, z)`` 三元组写法与 ``xy + start_z`` 写法必须同解。"""
    frame = _frame()
    xyz = plan_route_segment_xyz([], frame, None, (0.0, 0.0, 35.0), (300.0, 0.0, 35.0))
    xy = plan_route_segment([], frame, None, (0.0, 0.0), (300.0, 0.0), 35.0, goal_z_m=35.0)
    assert xyz.status is xy.status
    assert xyz.metrics.length_m == xy.metrics.length_m
    assert xyz.metrics.duration_s == xy.metrics.duration_s
    assert np.allclose(xyz.path_enu, xy.path_enu)


def test_agl_is_derived_from_ground() -> None:
    """地面 10 m、绝对 z=35 → 真高 25：z 与 AGL 不再是同一个数，库自己换算。"""
    dem, frame = _flat_dem(10.0), _frame()
    res = plan_route_segment([], frame, dem, (0.0, 0.0), (300.0, 0.0), 35.0, goal_z_m=35.0)
    assert res.status is Status.OK, res.failure_code
    assert res.stats.ground_start_m == pytest.approx(10.0, abs=0.5)
    assert res.stats.agl_min_m == pytest.approx(25.0, abs=0.5)
    assert res.trajectory.position_enu[:, 2].min() >= 35.0 - 1e-6


def test_z_below_ground_fails_band() -> None:
    """z 低于地面 → 真高为负 → out_of_flight_band（不静默抬到地面）。"""
    dem, frame = _flat_dem(10.0), _frame()
    res = plan_route_segment_xyz([], frame, dem, (0.0, 0.0, 12.0), (300.0, 0.0, 35.0))
    assert res.status is Status.INFEASIBLE
    assert res.failure_code.value == "out_of_flight_band"
    assert "真高" in (res.message or "")
    assert res.path_enu is None


def test_xy_entry_without_dem_uses_z_as_agl() -> None:
    """dem=None（平地 z=0）：绝对 z 就是真高，25 m 直接落在带内。"""
    res = plan_route_segment([], _frame(), None, (0.0, 0.0), (300.0, 0.0), 25.0)
    assert res.status is Status.OK
    assert res.stats.ground_start_m == 0.0

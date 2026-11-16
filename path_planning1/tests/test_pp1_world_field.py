"""全域场：取样形状/节点一致、越界报错、平地小世界有场与现算结果一致。"""

from __future__ import annotations

import numpy as np
import pytest
from path_planning1.contract import Building, LocalFrame, Status
from path_planning1.core.route_segment_params import RouteSegmentParams
from path_planning1.core.route_segment_planner import plan_route_segment
from path_planning1.core.route_segment_window import build_segment_window
from path_planning1.core.world_field import (
    WindowOutsideField,
    build_world_field,
)


def _frame() -> LocalFrame:
    return LocalFrame(lon0=113.3, lat0=23.12, lat_ref=23.12)


def _building(x: float, y: float, size: float = 20.0, height: float = 30.0) -> Building:
    return Building(
        footprint=((x, y), (x + size, y), (x + size, y + size), (x, y + size)),
        height_m=height,
    )


def _field(buildings: list[Building], bounds=(-150.0, -150.0, 450.0, 450.0)):
    return build_world_field(
        buildings, _frame(), None, spacing_m=(4.0, 4.0, 2.0),
        agl_min_m=10.0, agl_max_m=120.0, bounds_m=bounds,
    )


def test_world_field_marks_buildings_and_stays_positive_above() -> None:
    field = _field([_building(100.0, 100.0)])
    inside = field.at(np.array([[110.0, 110.0, 20.0]]))[0]
    above = field.at(np.array([[110.0, 110.0, 130.0]]))[0]
    assert inside < 0.0
    assert above > 0.0


def test_window_sdf_matches_point_sampling_at_nodes() -> None:
    field = _field([_building(150.0, -20.0, height=60.0)])
    window, _, _, _ = build_segment_window(
        None, _frame(), (0.0, 0.0), (300.0, 0.0), RouteSegmentParams()
    )
    sdf = field.window_sdf(window.grid)
    assert sdf.shape == window.grid.shape
    shp = np.array(window.grid.shape)
    idx = np.floor(shp * np.array([[0.2, 0.5, 0.2], [0.5, 0.8, 0.5], [0.8, 0.3, 0.1]]))
    pts = window.grid.index_to_enu(idx)
    for k, point in enumerate(pts):
        node = sdf[int(idx[k, 0]), int(idx[k, 1]), int(idx[k, 2])]
        assert abs(node - field.at(point[None, :])[0]) < 1e-4


def test_window_outside_field_raises() -> None:
    field = _field([])
    window, _, _, _ = build_segment_window(
        None, _frame(), (0.0, 0.0), (2000.0, 0.0), RouteSegmentParams()
    )
    with pytest.raises(WindowOutsideField):
        field.window_sdf(window.grid)


def test_planner_uses_field_and_matches_on_the_fly_on_flat_world() -> None:
    params = RouteSegmentParams()
    field = _field([])
    on_field = plan_route_segment(
        [], _frame(), None, (0.0, 0.0), (300.0, 0.0), 25.0, params, world_field=field
    )
    on_fly = plan_route_segment([], _frame(), None, (0.0, 0.0), (300.0, 0.0), 25.0, params)
    assert on_field.status is Status.OK
    assert on_fly.status is Status.OK
    assert on_field.failure_code is on_fly.failure_code
    assert abs(on_field.metrics.length_m - on_fly.metrics.length_m) < 5.0
    assert on_field.stats.window_shape == on_fly.stats.window_shape

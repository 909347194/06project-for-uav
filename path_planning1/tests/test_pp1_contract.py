"""path_planning1 契约自检：坐标换算、栅格索引、ROI 解析、互操作夹具。

这些测试的意义：契约是唯一边界，错在这里不会报错、只会让结果整体偏几米。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from path_planning1.contract import (
    GridSpec,
    LocalFrame,
    NoFlyZone,
    Objective,
    PlannerParams,
    PlanRequest,
    RegionOfInterest,
    VehicleLimits,
    WorldSnapshot,
)

import path_planning1

FIXTURE = Path(__file__).resolve().parents[2] / "data" / "contracts" / "plan_request_example.json"


def test_package_identity() -> None:
    assert path_planning1.ALGORITHM_ID == "pp1-rrt-nlp-time"
    assert path_planning1.__version__ == "0.4.0"


def test_local_frame_roundtrip() -> None:
    frame = LocalFrame(lon0=113.29109288376981, lat0=23.073499711374893, lat_ref=23.095175858945964)
    (x, y) = frame.to_local(113.30, 23.10)
    lon, lat = frame.to_lonlat(x, y)
    assert lon == pytest.approx(113.30, abs=1e-9)
    assert lat == pytest.approx(23.10, abs=1e-9)
    # 4.8 km × 4.8 km 的广州窗口：x 方向 1 米应约等于 1 米，不是几度
    assert frame.to_local(frame.lon0 + 1e-3, frame.lat0)[0] > 90.0


def test_grid_spec_roundtrip() -> None:
    grid = GridSpec(origin_enu=(0.0, 0.0, 30.0), spacing_m=(4.0, 4.0, 2.0), shape=(10, 12, 8))
    idx = np.array([[0, 0, 0], [3, 5, 2]], dtype=int)
    enu = grid.index_to_enu(idx)
    assert np.allclose(enu, [[0.0, 0.0, 30.0], [12.0, 20.0, 34.0]])
    assert np.allclose(grid.enu_to_index(enu), idx)


def test_roi_half_width_priority() -> None:
    roi = RegionOfInterest(width_ratio=0.2)
    assert roi.half_width_for(2000.0) == pytest.approx(200.0)
    roi_explicit = RegionOfInterest(width_ratio=0.2, half_width_m=75.0)
    assert roi_explicit.half_width_for(2000.0) == pytest.approx(75.0)


def test_roi_has_no_ghost_switches() -> None:
    """走廊就是计算域，所以只剩"多大"的参数：不许再留无人消费的开关。"""
    roi = RegionOfInterest(width_ratio=0.2, margin_m=50.0)
    assert not hasattr(roi, "clip_grid")
    assert not hasattr(roi, "hard_constraint")
    assert roi.half_width_for(2000.0) == pytest.approx(200.0)


def test_rotated_grid_roundtrip() -> None:
    """斜栅格（走廊轴向）：索引↔ENU 必须严格互逆，且轴 0 真的指向 yaw 方向。"""
    grid = GridSpec(
        origin_enu=(10.0, 20.0, 30.0), spacing_m=(4.0, 4.0, 2.0), shape=(5, 3, 4),
        yaw_rad=np.pi / 2,            # 轴 0 指向 +y，轴 1 指向 -x
    )
    a0, a1 = grid.axes_xy
    assert np.allclose(a0, [0.0, 1.0]) and np.allclose(a1, [-1.0, 0.0])
    enu = grid.index_to_enu(np.array([[2.0, 1.0, 0.0]]))
    assert np.allclose(enu, [[10.0 - 4.0, 20.0 + 8.0, 30.0]])
    assert np.allclose(grid.enu_to_index(enu), [[2.0, 1.0, 0.0]])


def test_planner_params_rrt_are_metric() -> None:
    """RRT 参数已从"格"(15/5) 换算成"米"(60/20)，缩放步长要落在 [min, max] 内。"""
    p = PlannerParams()
    assert p.rrt_step_m == pytest.approx(60.0)
    assert p.rrt_min_step_m == pytest.approx(20.0)
    assert p.rrt_sdf_step_scale == pytest.approx(0.4)
    step = min(max(p.rrt_sdf_step_scale * 30.0, p.rrt_min_step_m), p.rrt_step_m)
    assert step == pytest.approx(20.0)


def test_world_defaults_match_agreed_band_and_clearance() -> None:
    """飞行带 10–120 m（用户定）、净空 5 m（为使街道峡谷可通行）；目标是最短飞行时间。"""
    world = WorldSnapshot(
        world_id="gz", frame=LocalFrame(lon0=113.291, lat0=23.073), buildings=[],
        grid=GridSpec(origin_enu=(0.0, 0.0, 10.0), spacing_m=(4.0, 4.0, 2.0),
                      shape=(10, 10, 56)),
    )
    assert (world.alt_min_m, world.alt_max_m) == (10.0, 120.0)
    assert VehicleLimits().clearance_m == pytest.approx(5.0)
    assert PlanRequest(world_id="gz", start_enu=(0.0, 0.0, 60.0),
                       goal_enu=(1.0, 0.0, 60.0)).objective is Objective.TIME
    # 净空只有一个来源（VehicleLimits），PlannerParams 不许重复定义
    assert not hasattr(PlannerParams(), "clearance_m")


def test_nofly_slot_exists_but_is_empty() -> None:
    """WorldSnapshot.nofly 仍是预留形状；实际引擎消费 CityDataset.nofly 的同一类型。"""
    world = WorldSnapshot(
        world_id="gz", frame=LocalFrame(lon0=113.291, lat0=23.073), buildings=[],
        grid=GridSpec(origin_enu=(0.0, 0.0, 10.0), spacing_m=(4.0, 4.0, 2.0),
                      shape=(10, 10, 56)),
    )
    assert world.nofly == []
    zone = NoFlyZone(footprint=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0)), z_max_m=120.0)
    assert zone.z_min_m == 0.0 and zone.z_max_m == pytest.approx(120.0)


def test_interop_fixture_is_consumable() -> None:
    """同一份航段夹具：本包必须能无歧义地读成 PlanRequest。"""
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    req = PlanRequest(
        world_id=raw["world_id"],
        start_enu=tuple(raw["start_enu"]),
        goal_enu=tuple(raw["goal_enu"]),
        objective=Objective.ENERGY,
        vehicle=VehicleLimits(
            v_max_mps=raw["vehicle"]["v_max_mps"],
            mass_kg=raw["vehicle"]["mass_kg"],
            clearance_m=raw["vehicle"]["clearance_m"],
        ),
        roi=RegionOfInterest(
            width_ratio=raw["roi"]["width_ratio"], margin_m=raw["roi"]["margin_m"]
        ),
        seed=raw["seed"],
        route_segment_id=raw["segment_id"],
    )
    assert req.start_enu == (100.0, 120.0, 60.0)
    assert req.goal_enu == (2200.0, 1800.0, 60.0)
    assert req.roi.half_width_for(2_691.0) > 0.0   # 水平距离约 2.69 km

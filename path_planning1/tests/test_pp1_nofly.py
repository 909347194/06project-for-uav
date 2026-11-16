"""静态禁飞区：geojson 入口、悬浮棱柱语义、两条障碍通道与版本隔离。

禁飞区最容易出的不是"没生效"，而是**静默算错**：投影米坐标被当度、绝对高程被当 AGL、
悬浮层被填成实心柱、数据变了产物没重建。这里把这些都钉在启动期或单元级。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from path_planning1.contract import (
    FailureCode,
    GridSpec,
    LocalFrame,
    NoFlyZone,
    RegionOfInterest,
    Status,
)
from path_planning1.core.city_dataset import load_city
from path_planning1.core.city_loader import load_nofly
from path_planning1.core.clearance_check import BuildingPrisms
from path_planning1.core.dataset_identity import dataset_digest
from path_planning1.core.dem_terrain import Dem
from path_planning1.core.occupancy_sdf import build_occupancy
from path_planning1.core.params_config import world_kwargs_from_config
from path_planning1.core.planning_context import create_context
from path_planning1.core.route_segment_params import RouteSegmentParams
from path_planning1.core.route_segment_planner import plan_route_segment

_RING = [[113.0, 23.0], [113.02, 23.0], [113.02, 23.02], [113.0, 23.02], [113.0, 23.0]]


def _world(root: Path, features: list[dict] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "building").mkdir(exist_ok=True)
    # 建筑是必需项：给桩世界放一栋角落里的小楼，免得被"空世界"保护挡住。
    stub = {"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {"Height": 20.0},
        "geometry": {"type": "Polygon", "coordinates": [[
            [113.018, 23.018], [113.020, 23.018], [113.020, 23.020],
            [113.018, 23.020], [113.018, 23.018],
        ]]},
    }]}
    (root / "building" / "stub.geojson").write_text(json.dumps(stub), encoding="utf-8")
    (root / "region.geojson").write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {}, "geometry": {
            "type": "Polygon", "coordinates": [_RING]}}],
    }), encoding="utf-8")
    if features is not None:
        (root / "nofly.geojson").write_text(json.dumps({
            "type": "FeatureCollection", "features": features,
        }), encoding="utf-8")
    return root


def _feature(ring: list[list[float]], **props) -> dict:
    return {"type": "Feature", "properties": props,
            "geometry": {"type": "Polygon", "coordinates": [ring]}}


def _flat_dem(z_m: float = 3650.0) -> Dem:
    """覆盖 ``_RING`` 的平 DEM：1e-4 度/格，201×201。"""
    return Dem(
        z_m=np.full((201, 201), z_m, dtype=np.float32),
        lon_ul=113.0, lat_ul=23.02, dlon=1e-4, dlat=1e-4,
    )


def test_nofly_agl_mode_converts_to_absolute_on_flat_dem(tmp_path: Path) -> None:
    path = tmp_path / "nofly.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [
        _feature(_RING, agl_min_m=0.0, agl_max_m=120.0, reason="低空"),
    ]}), encoding="utf-8")
    zones = load_nofly(
        path, LocalFrame(113.0, 23.0), dem=_flat_dem(3650.0), altitude_mode="agl"
    )
    assert (zones[0].z_min_m, zones[0].z_max_m, zones[0].reason) == (
        3650.0, 3770.0, "低空"
    )


def test_nofly_agl_mode_covers_dem_cell_corners(tmp_path: Path) -> None:
    """双线性 DEM 的单元内极值可能不在顶点/格心；四角包络必须覆盖 0–100 m。"""
    path = tmp_path / "nofly.geojson"
    ring = [[113.009, 23.009], [113.011, 23.009], [113.011, 23.011],
            [113.009, 23.011], [113.009, 23.009]]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [
        _feature(ring, agl_min_m=0.0, agl_max_m=120.0),
    ]}), encoding="utf-8")
    dem = Dem(
        z_m=np.array([[0.0, 100.0], [100.0, 0.0]], dtype=np.float32),
        lon_ul=113.0, lat_ul=23.02, dlon=0.02, dlat=0.02,
    )
    zones = load_nofly(
        path, LocalFrame(113.0, 23.0), dem=dem, altitude_mode="agl"
    )
    assert (zones[0].z_min_m, zones[0].z_max_m) == (0.0, 220.0)


def test_nofly_agl_mode_is_strict_about_dem_and_fields(tmp_path: Path) -> None:
    path = tmp_path / "nofly.geojson"

    def write(**props) -> None:
        path.write_text(json.dumps({"type": "FeatureCollection", "features": [
            _feature(_RING, **props),
        ]}), encoding="utf-8")

    write(agl_min_m=10.0, agl_max_m=120.0)
    with pytest.raises(ValueError, match="需要 DEM"):
        load_nofly(path, LocalFrame(113.0, 23.0), altitude_mode="agl")
    write(agl_min_m=10.0)
    with pytest.raises(ValueError, match="必须同时提供"):
        load_nofly(path, LocalFrame(113.0, 23.0), dem=_flat_dem(), altitude_mode="agl")
    write(z_min_m=20.0, z_max_m=120.0)
    with pytest.raises(ValueError, match="不能同时提供 z_min_m"):
        load_nofly(path, LocalFrame(113.0, 23.0), dem=_flat_dem(), altitude_mode="agl")
    write(agl_min_m=0.0, agl_max_m=120.0)
    with pytest.raises(ValueError, match="不能同时提供 agl_min_m"):
        load_nofly(path, LocalFrame(113.0, 23.0), dem=_flat_dem(), altitude_mode="absolute")


def test_nofly_agl_mode_rejects_polygon_outside_dem(tmp_path: Path) -> None:
    path = tmp_path / "nofly.geojson"
    ring = [[113.03, 23.00], [113.05, 23.00], [113.05, 23.02],
            [113.03, 23.02], [113.03, 23.00]]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [
        _feature(ring, agl_min_m=0.0, agl_max_m=120.0),
    ]}), encoding="utf-8")
    with pytest.raises(ValueError, match="超出 DEM"):
        load_nofly(path, LocalFrame(113.0, 23.0), dem=_flat_dem(), altitude_mode="agl")


def test_load_city_agl_mode_uses_dem(monkeypatch, tmp_path: Path) -> None:
    root = _world(tmp_path / "city", [_feature(_RING, agl_min_m=0.0, agl_max_m=120.0)])
    (root / "dem.tif").write_bytes(b"stub")
    monkeypatch.setattr(
        "path_planning1.core.city_dataset.load_dem",
        lambda path: _flat_dem(3650.0),
    )
    ds = load_city(root, nofly_altitude_mode="agl")
    assert (ds.nofly[0].z_min_m, ds.nofly[0].z_max_m) == (3650.0, 3770.0)
    with pytest.raises(ValueError, match="需要 DEM"):
        load_city(root, with_dem=False, nofly_altitude_mode="agl")


def test_load_city_agl_mode_requires_dem_even_without_nofly_file(tmp_path: Path) -> None:
    """AGL 是数据口径，不是“有文件才校验”的；没禁飞文件也不能配 agl + 平地。"""
    root = _world(tmp_path / "city")
    with pytest.raises(ValueError, match="需要 DEM"):
        load_city(root, with_dem=False, nofly_altitude_mode="agl")


def test_world_kwargs_accepts_and_validates_nofly_altitude_mode(tmp_path: Path) -> None:
    root = _world(tmp_path / "city")
    kw = world_kwargs_from_config({"world": {
        "root": str(root), "nofly_altitude_mode": "agl",
    }})
    assert kw["nofly_altitude_mode"] == "agl"
    with pytest.raises(ValueError, match="nofly_altitude_mode"):
        world_kwargs_from_config({"world": {
            "root": str(root), "nofly_altitude_mode": "wrong",
        }})


def test_nofly_altitude_mode_changes_digest(tmp_path: Path) -> None:
    root = _world(tmp_path / "city", [_feature(_RING, agl_min_m=0.0, agl_max_m=120.0)])
    absolute = dataset_digest({"root": root})
    agl = dataset_digest({"root": root, "nofly_altitude_mode": "agl"})
    assert absolute != agl


def test_nofly_altitude_mode_not_in_digest_without_nofly_file(tmp_path: Path) -> None:
    """没有禁飞文件时 mode 不参与 fingerprint；否则会把老数据的 digest 无端改掉。"""
    root = _world(tmp_path / "city")
    assert dataset_digest({"root": root}) == dataset_digest(
        {"root": root, "nofly_altitude_mode": "agl"}
    )


def test_geojson_loads_to_enu_with_absolute_z_and_reason(tmp_path: Path) -> None:
    ring = [[113.005, 23.005], [113.010, 23.005], [113.010, 23.010],
            [113.005, 23.010], [113.005, 23.005]]
    root = _world(tmp_path / "city", [_feature(ring, z_min_m=30.0, z_max_m=60.0, reason="机场")])
    ds = load_city(root, with_dem=False)
    assert ds.n_nofly == 1
    zone = ds.nofly[0]
    assert (zone.z_min_m, zone.z_max_m, zone.reason) == (30.0, 60.0, "机场")
    assert all(x > 0 and y > 0 for x, y in zone.footprint)
    assert len(zone.footprint) == 4

    bare = load_city(_world(tmp_path / "bare", [_feature(ring)]), with_dem=False)
    assert (bare.nofly[0].z_min_m, bare.nofly[0].z_max_m) == (0.0, 1000.0)


def test_bad_nofly_data_fails_at_load(tmp_path: Path) -> None:
    ring = [[113.005, 23.005], [113.010, 23.005], [113.010, 23.010], [113.005, 23.005]]
    projected = [[380000.0, 2550000.0], [381000.0, 2550000.0],
                 [381000.0, 2551000.0], [380000.0, 2550000.0]]
    with pytest.raises(ValueError, match="不是经纬度"):
        load_city(_world(tmp_path / "projected", [_feature(projected)]), with_dem=False)
    with pytest.raises(ValueError, match="z_min_m"):
        load_city(_world(tmp_path / "zbad", [_feature(ring, z_min_m=60, z_max_m=30)]),
                  with_dem=False)
    holed = {"type": "Feature", "properties": {}, "geometry": {
        "type": "Polygon", "coordinates": [ring, [
            [113.006, 23.006], [113.007, 23.006], [113.007, 23.007], [113.006, 23.006]]]}}
    with pytest.raises(ValueError, match="内环"):
        load_city(_world(tmp_path / "holed", [holed]), with_dem=False)


def test_explicit_nofly_path_must_exist_but_default_may_be_absent(tmp_path: Path) -> None:
    root = _world(tmp_path / "city")
    assert load_city(root, with_dem=False).nofly == ()
    with pytest.raises(FileNotFoundError, match="nofly_path"):
        load_city(root, with_dem=False, nofly_path="missing.geojson")
    kw = world_kwargs_from_config({"world": {"root": str(root), "nofly_path": "nofly.geojson"}})
    assert kw["nofly_path"] == "nofly.geojson"
    with pytest.raises(ValueError, match="nofly_path"):
        world_kwargs_from_config({"world": {"root": str(root), "nofly_path": 7}})


def test_nofly_zone_is_a_floating_slab_not_a_column() -> None:
    grid = GridSpec(origin_enu=(0.0, 0.0, 0.0), spacing_m=(2.0, 2.0, 2.0), shape=(20, 20, 20))
    zone = NoFlyZone(footprint=((4.0, 0.0), (8.0, 0.0), (8.0, 10.0), (4.0, 10.0)),
                     z_min_m=8.0, z_max_m=12.0)
    occ = build_occupancy([], grid, nofly=[zone])
    assert not occ[3, 5, 1]   # z=2：板下方自由
    assert occ[3, 5, 5]       # z=10：板内占据
    assert not occ[3, 5, 8]   # z=16：板上方自由
    assert not occ[0, 5, 5]   # 板外自由


def _params() -> RouteSegmentParams:
    return RouteSegmentParams(
        roi=RegionOfInterest(width_ratio=0.4, margin_m=20.0),
        spacing_m=(20.0, 20.0, 20.0),
    )


def test_endpoint_inside_nofly_returns_its_own_failure_code() -> None:
    zone = NoFlyZone(footprint=((-50.0, -50.0), (50.0, -50.0), (50.0, 50.0), (-50.0, 50.0)),
                     z_min_m=0.0, z_max_m=120.0)
    res = plan_route_segment([], LocalFrame(113.0, 23.0, 23.05), None,
                             (0.0, 0.0), (400.0, 0.0), 25.0, _params(), nofly=[zone])
    assert res.status is Status.INFEASIBLE
    assert res.failure_code is FailureCode.ENDPOINT_IN_NOFLY


def test_planner_routes_around_a_floating_nofly_slab() -> None:
    zone = NoFlyZone(footprint=((180.0, -80.0), (220.0, -80.0), (220.0, 80.0), (180.0, 80.0)),
                     z_min_m=20.0, z_max_m=40.0, reason="临时空域")
    res = plan_route_segment([], LocalFrame(113.0, 23.0, 23.05), None,
                             (0.0, 0.0), (400.0, 0.0), 25.0, _params(), nofly=[zone])
    assert res.status is Status.OK and res.path_enu is not None
    delivered = res.trajectory.position_enu if res.trajectory is not None else res.path_enu
    prism = BuildingPrisms([zone])
    assert not any(prism.contains(point) for point in delivered)
    assert float(delivered[:, 2].max()) > 45.0   # 从悬浮层上方绕，而不是钻过去


def test_context_carries_nofly_from_data_directory(tmp_path: Path) -> None:
    ring = [[113.005, 23.005], [113.010, 23.005], [113.010, 23.010], [113.005, 23.005]]
    root = _world(tmp_path / "city", [_feature(ring, z_min_m=0.0, z_max_m=120.0)])
    ctx = create_context({"world": {"root": str(root), "with_dem": False,
                                    "field_mode": "on_demand"}})
    assert ctx.dataset.n_nofly == 1
    sx, sy = ctx.dataset.frame.to_local(113.007, 23.007)   # 禁飞多边形中心
    res = ctx.plan((sx, sy), (sx + 400.0, sy), 25.0)
    assert res.failure_code is FailureCode.ENDPOINT_IN_NOFLY


def test_nofly_file_changes_dataset_digest_and_reverts(tmp_path: Path) -> None:
    root = _world(tmp_path / "city")
    base = dataset_digest({"root": root})
    ring = [[113.005, 23.005], [113.010, 23.005], [113.010, 23.010], [113.005, 23.005]]
    (root / "nofly.geojson").write_text(json.dumps({
        "type": "FeatureCollection", "features": [_feature(ring, z_max_m=60.0)]}), encoding="utf-8")
    with_zone = dataset_digest({"root": root})
    assert with_zone != base
    (root / "nofly.geojson").write_text(json.dumps({
        "type": "FeatureCollection", "features": [_feature(ring, z_max_m=80.0)]}), encoding="utf-8")
    assert dataset_digest({"root": root}) != with_zone
    (root / "nofly.geojson").unlink()
    assert dataset_digest({"root": root}) == base

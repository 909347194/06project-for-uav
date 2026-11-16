"""建筑数据两种入口：shapefile（GBK）与 GeoJSON（UTF-8）共用同一套口径。

这里钉住三件最容易静默出错的事：高度必须是有限正数、几何口径一致（MultiPolygon
取最大、洞填实）、目录里一栋可用建筑都没有时直接报错而不是建出空世界。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from path_planning1.contract import LocalFrame
from path_planning1.core.city_dataset import load_city
from path_planning1.core.city_loader import load_buildings
from path_planning1.core.dataset_identity import dataset_digest

_FRAME = LocalFrame(113.0, 23.0, 23.01)
_RING = [
    [113.005, 23.005], [113.010, 23.005], [113.010, 23.010],
    [113.005, 23.010], [113.005, 23.005],
]


def _feature(ring: list[list[float]], **props: object) -> dict:
    return {
        "type": "Feature",
        "properties": props,
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }


def _write_geojson(directory: Path, name: str, features: list[dict]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}),
        encoding="utf-8",
    )
    return path


def _write_shp(directory: Path, name: str, height: float = 30.0) -> Path:
    import shapefile

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    writer = shapefile.Writer(str(path), shapeType=shapefile.POLYGON)
    writer.field("Height", "N", size=10, decimal=2)
    writer.poly([_RING])
    writer.record(height)
    writer.close()
    return path


def test_geojson_buildings_load_with_same_enu_and_height(tmp_path: Path) -> None:
    _write_geojson(tmp_path, "b.geojson", [_feature(_RING, Height=18.5)])
    buildings = load_buildings(tmp_path, _FRAME)
    assert len(buildings) == 1
    b = buildings[0]
    assert b.height_m == 18.5
    assert b.base_z_m == 0.0
    assert b.source == "b"
    assert b.footprint[0] == pytest.approx(_FRAME.to_local(*_RING[0]))


def test_shp_and_geojson_can_coexist(tmp_path: Path) -> None:
    _write_shp(tmp_path, "a.shp", height=30.0)
    _write_geojson(tmp_path, "b.geojson", [_feature(_RING, Height=12.0)])
    buildings = load_buildings(tmp_path, _FRAME)
    assert len(buildings) == 2
    assert {b.source for b in buildings} == {"a", "b"}
    assert sorted(b.height_m for b in buildings) == [12.0, 30.0]


def test_custom_height_field_applies_to_geojson(tmp_path: Path) -> None:
    _write_geojson(tmp_path, "b.geojson", [_feature(_RING, H=42.0)])
    assert load_buildings(tmp_path, _FRAME, height_field="H")[0].height_m == 42.0
    with pytest.raises(ValueError, match="Height"):
        load_buildings(tmp_path, _FRAME)


@pytest.mark.parametrize("bad", [None, 0, -3, "12m", float("nan"), float("inf")])
def test_bad_geojson_height_fails_loudly(tmp_path: Path, bad: object) -> None:
    props = {} if bad is None else {"Height": bad}
    _write_geojson(tmp_path, "b.geojson", [_feature(_RING, **props)])
    with pytest.raises(ValueError, match="高度字段"):
        load_buildings(tmp_path, _FRAME)


def test_bad_shp_height_still_fails_loudly(tmp_path: Path) -> None:
    _write_shp(tmp_path, "a.shp", height=0.0)
    with pytest.raises(ValueError, match="高度字段"):
        load_buildings(tmp_path, _FRAME)


def test_geojson_projected_coords_are_rejected(tmp_path: Path) -> None:
    projected = [
        [380000.0, 2550000.0], [381000.0, 2550000.0],
        [381000.0, 2551000.0], [380000.0, 2550000.0],
    ]
    _write_geojson(tmp_path, "b.geojson", [_feature(projected, Height=20.0)])
    with pytest.raises(ValueError, match="不是经纬度"):
        load_buildings(tmp_path, _FRAME)


def test_multi_polygon_takes_largest_and_hole_is_filled(tmp_path: Path) -> None:
    small = [[113.000, 23.000], [113.001, 23.000], [113.001, 23.001], [113.000, 23.000]]
    hole = [[113.006, 23.006], [113.007, 23.006], [113.007, 23.007], [113.006, 23.006]]
    multi = {
        "type": "Feature",
        "properties": {"Height": 20.0},
        "geometry": {"type": "MultiPolygon", "coordinates": [
            [small], [_RING, hole],
        ]},
    }
    path = _write_geojson(tmp_path, "b.geojson", [multi])
    buildings = load_buildings(path.parent, _FRAME)
    assert len(buildings) == 1
    assert len(buildings[0].footprint) == len(_RING) - 1   # 洞被忽略，外环填实


@pytest.mark.parametrize("features", [[], [{"type": "Feature", "properties": {"Height": 20.0},
                                            "geometry": {"type": "Point", "coordinates": [113.01, 23.01]}}]])
def test_no_usable_building_fails_instead_of_empty_world(tmp_path: Path, features: list[dict]) -> None:
    _write_geojson(tmp_path, "b.geojson", features)
    with pytest.raises(ValueError, match="没有可用建筑"):
        load_buildings(tmp_path, _FRAME)


def test_tiny_fragments_do_not_count_as_buildings(tmp_path: Path) -> None:
    tiny = [[113.0, 23.0], [113.00000001, 23.0], [113.00000001, 23.00000001], [113.0, 23.0]]
    _write_geojson(tmp_path, "b.geojson", [_feature(tiny, Height=20.0)])
    with pytest.raises(ValueError, match="没有可用建筑"):
        load_buildings(tmp_path, _FRAME, min_area_m2=1.0)


def test_load_city_reads_geojson_buildings_and_digest_tracks_them(tmp_path: Path) -> None:
    region = {"type": "FeatureCollection", "features": [_feature(_RING)]}
    (tmp_path / "region.geojson").write_text(json.dumps(region), encoding="utf-8")
    path = _write_geojson(tmp_path / "building", "b.geojson", [_feature(_RING, Height=18.0)])
    ds = load_city(tmp_path, with_dem=False)
    assert ds.n_buildings == 1 and ds.buildings[0].height_m == 18.0

    before = dataset_digest({"root": tmp_path})
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [
        _feature(_RING, Height=19.0)]}), encoding="utf-8")
    assert dataset_digest({"root": tmp_path}) != before

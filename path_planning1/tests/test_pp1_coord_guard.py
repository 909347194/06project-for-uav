"""坐标格式自检：投影米坐标必须在数据加载期被挡住，不静默算错建筑位置。

本库只吃 WGS84 经纬度（EPSG:4326）。喂进 UTM / CGCS2000 等投影米坐标时，
``to_local`` 会把米当度、建筑落到地球别处，规划器仍返回看似正常的轨迹——
这是最难查的一类错误。自检让它在启动期就明确失败。
转换本身由上层在数据入库前用 pyproj 完成，不进本库依赖。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from path_planning1.contract import LocalFrame
from path_planning1.core.city_dataset import load_city
from path_planning1.core.city_loader import frame_from_geojson, load_buildings

_BAD_MSG = "不是经纬度"


def _write_geojson(root: Path, ring: list[list[float]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    data = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        }],
    }
    (root / "region.geojson").write_text(json.dumps(data), encoding="utf-8")
    (root / "building").mkdir(exist_ok=True)
    built = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"Height": 20.0},
            "geometry": {"type": "Polygon", "coordinates": [[
                [113.005, 23.005], [113.010, 23.005], [113.010, 23.010],
                [113.005, 23.010], [113.005, 23.005],
            ]]},
        }],
    }
    (root / "building" / "b.geojson").write_text(json.dumps(built), encoding="utf-8")


def _write_shp_with_point(path: Path, x: float, y: float) -> None:
    """写一个最小 shapefile（Point / 经纬度位置填投影米），供 load_buildings 读取。"""
    import shapefile

    writer = shapefile.Writer(str(path), shapeType=shapefile.POINT)
    writer.field("Height", "N", size=10, decimal=2)
    writer.point(x, y)
    writer.record(20.0)
    writer.close()


def test_geojson_projected_coords_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "projected"
    ring = [
        [380000.0, 2550000.0], [381000.0, 2550000.0],
        [381000.0, 2551000.0], [380000.0, 2551000.0], [380000.0, 2550000.0],
    ]
    _write_geojson(root, ring)
    with pytest.raises(ValueError, match=_BAD_MSG):
        frame_from_geojson(root / "region.geojson")
    with pytest.raises(ValueError, match=_BAD_MSG):
        load_city(root, with_dem=False)


def test_shp_projected_coords_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "projected_shp"
    _write_geojson(root, [[113.0, 23.0], [113.02, 23.0], [113.02, 23.02],
                          [113.0, 23.02], [113.0, 23.0]])
    shp = root / "building" / "bad.shp"
    _write_shp_with_point(shp, 380000.0, 2550000.0)
    with pytest.raises(ValueError, match=_BAD_MSG):
        load_buildings(root / "building", LocalFrame(113.0, 23.0, 23.01))


def test_lonlat_out_of_range_variants(tmp_path: Path) -> None:
    """经度 113 没问题，但纬度放到 2550000 这类米数必须挡住。"""
    root = tmp_path / "bad_lat"
    _write_geojson(root, [[113.0, 23.0], [113.02, 23.0], [113.02, 2550000.0],
                          [113.0, 2550000.0], [113.0, 23.0]])
    with pytest.raises(ValueError, match=_BAD_MSG):
        frame_from_geojson(root / "region.geojson")


def test_valid_geojson_still_loads(tmp_path: Path) -> None:
    root = tmp_path / "valid"
    _write_geojson(root, [[113.0, 23.0], [113.02, 23.0], [113.02, 23.02],
                          [113.0, 23.02], [113.0, 23.0]])
    frame = frame_from_geojson(root / "region.geojson")
    assert frame.lon0 == pytest.approx(113.0)
    assert frame.lat0 == pytest.approx(23.0)
    ds = load_city(root, with_dem=False)
    assert ds.n_buildings == 1

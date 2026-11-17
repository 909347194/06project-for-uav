"""ta1 自带的地形适配：世界目录 → 垂直切面采样器（不 import pp1）。"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .buildings import BuildingLayer
from .dem import DEM, load_geotiff_dem
from .frame import LocalFrame

_HEIGHT_FIELDS = ("Height", "Height_1", "height", "height_m")


@dataclass(frozen=True)
class TerrainData:
    """一个世界的 frame、DEM（可无）与建筑层；采样器返回绝对 z。"""

    frame: LocalFrame
    dem: DEM | None
    buildings: BuildingLayer

    def ground_z(self, xs, ys) -> np.ndarray:
        """各点地面绝对高程；无 DEM 时为 0。"""
        if self.dem is None:
            return np.zeros_like(np.asarray(xs, dtype=float))
        lon, lat = self.frame.to_lonlat(xs, ys)
        return self.dem.at_lonlat(lon, lat)

    def obstacle_top_z(self, xs, ys) -> np.ndarray:
        """各点障碍包络顶 = 地面 + 该点建筑相对高度（无楼处 = 地面）。"""
        ground = np.asarray(self.ground_z(xs, ys), dtype=float)
        return ground + self.buildings.heights_at(xs, ys)

    def samplers(self):
        return self.ground_z, self.obstacle_top_z


def _iter_rings(geometry: dict):
    """GeoJSON Polygon / MultiPolygon → 外环坐标序列（忽略内环）。"""
    if not geometry:
        return
    kind = geometry.get("type")
    coords = geometry.get("coordinates") or []
    if kind == "Polygon":
        for ring in coords[:1]:
            yield ring
    elif kind == "MultiPolygon":
        for poly in coords:
            for ring in poly[:1]:
                yield ring
    elif kind == "GeometryCollection":
        for child in geometry.get("geometries", []):
            yield from _iter_rings(child)


def _region_bounds(path: Path) -> tuple[float, float, float, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    lons: list[float] = []
    lats: list[float] = []
    for feature in data.get("features", []):
        for ring in _iter_rings(feature.get("geometry") or {}):
            for point in ring:
                lons.append(float(point[0]))
                lats.append(float(point[1]))
    if not lons:
        raise ValueError(f"{path} 没有可用的区域坐标")
    if not (-180 <= min(lons) <= 180 and -90 <= min(lats) <= 90):
        raise ValueError(f"{path} 不是 WGS84 经纬度")
    return min(lons), min(lats), max(lons), max(lats)


def frame_from_region(path: str | Path) -> LocalFrame:
    """与 pp1 同口径：锚点取 bbox 左下角，``lat_ref`` 取 bbox 中心纬度。"""
    left, bottom, _right, top = _region_bounds(Path(path))
    return LocalFrame(lon0=left, lat0=bottom, lat_ref=0.5 * (bottom + top))


def _height(props: dict, field: str) -> float | None:
    for key in (field, *_HEIGHT_FIELDS):
        if key in props:
            try:
                value = float(props[key])
            except (TypeError, ValueError):
                return None
            return value if np.isfinite(value) and value > 0 else None
    return None


def _add_geojson(path: Path, frame: LocalFrame, field: str, out: list, heights: list) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    for feature in data.get("features", []):
        props = feature.get("properties") or {}
        h = _height(props, field)
        if h is None:
            continue
        for ring in _iter_rings(feature.get("geometry") or {}):
            if len(ring) < 3:
                continue
            lon = np.asarray([p[0] for p in ring], dtype=float)
            lat = np.asarray([p[1] for p in ring], dtype=float)
            x, y = frame.to_local(lon, lat)
            out.append(np.column_stack([x, y]))
            heights.append(h)


def _add_shapefile(path: Path, frame: LocalFrame, field: str, out: list, heights: list) -> None:
    try:
        import shapefile
    except ImportError as exc:  # pragma: no cover - 环境问题
        raise ModuleNotFoundError(
            "读 Shapefile 需要 pyshp：pip install 'task_allocation1[terrain]'"
        ) from exc
    reader = shapefile.Reader(str(path), encoding="gbk", encodingErrors="replace")
    try:
        names = [f[0] for f in reader.fields[1:]]
        name = next((key for key in (field, *_HEIGHT_FIELDS) if key in names), None)
        if name is None:
            raise ValueError(f"{path} 缺少高度字段 {field!r}（实际：{names}）")
        index = names.index(name)
        for shape, record in zip(reader.shapes(), reader.records(), strict=False):
            h = _height({name: record[index]}, name)
            if h is None:
                continue
            pts = np.asarray(shape.points, dtype=float)
            starts = list(shape.parts) + [len(pts)]
            for a, b in itertools.pairwise(starts):
                ring = pts[a:b]
                if len(ring) < 3:
                    continue
                x, y = frame.to_local(ring[:, 0], ring[:, 1])
                out.append(np.column_stack([x, y]))
                heights.append(h)
    finally:
        reader.close()


def load_buildings(root: str | Path, frame: LocalFrame, field: str = "Height") -> BuildingLayer:
    root = Path(root)
    polygons: list[np.ndarray] = []
    heights: list[float] = []
    if root.is_dir():
        for path in sorted(root.glob("*.geojson")):
            _add_geojson(path, frame, field, polygons, heights)
        for path in sorted(root.glob("*.shp")):
            _add_shapefile(path, frame, field, polygons, heights)
    return BuildingLayer(polygons, heights)


def load_world_terrain(root: str | Path, field: str = "Height") -> TerrainData:
    """读世界目录；没有 ``dem.tif`` 时地面按 0，没有建筑时包络=地面。"""
    root = Path(root)
    region = root / "region.geojson"
    if not region.exists():
        raise FileNotFoundError(f"缺少区域轮廓：{region}")
    frame = frame_from_region(region)
    buildings = load_buildings(root / "building", frame, field=field)
    dem_path = root / "dem.tif"
    dem = load_geotiff_dem(dem_path) if dem_path.exists() else None
    return TerrainData(frame=frame, dem=dem, buildings=buildings)

"""数据加载：经纬度 shp / geojson → ENU 米制 ``Building``，以及区域 bbox。

属于 I/O，放 core 而不是 contract（契约只承接数据，不读文件）。
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar

import shapefile
import shapely
from shapely.geometry import Polygon
from shapely.geometry import shape as shapely_shape

from ..contract import M_PER_DEG_LAT, Building, LocalFrame, NoFlyZone
from ..contract.defaults import (
    DEFAULT_NOFLY_ALTITUDE_MODE,
    MIN_BUILDING_AREA_M2,
    NOFLY_ALTITUDE_ABSOLUTE,
    NOFLY_ALTITUDE_AGL,
)
from .dem_terrain import Dem
from .nofly_altitude import ground_bounds, parse_nofly_altitude, validate_altitude_mode

_Prism = TypeVar("_Prism", Building, NoFlyZone)


def _require_lonlat(lon: float, lat: float, source: str) -> None:
    """挡住投影坐标：本库只吃经纬度（度），喂米坐标会静默算错建筑位置。"""
    if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
        raise ValueError(
            f"{source} 的坐标 ({lon!r}, {lat!r}) 不是经纬度（度）："
            "本库只接受 WGS84 经纬度 (EPSG:4326)；若是 UTM / CGCS2000 等投影米坐标，"
            "请在数据入库前用 pyproj 转成经纬度（参考 docs/后端怎么用-pp1.md §10.2）。"
        )


def frame_from_geojson(path: str | Path) -> LocalFrame:
    """用区域轮廓的 bbox 中心纬度作 ``lat_ref``，锚点取 bbox 左下角。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    geoms = [shapely_shape(f["geometry"]) for f in data["features"] if f.get("geometry")]
    if not geoms:
        raise ValueError(f"no geometry in {path}")
    lon0, lat0, lon1, lat1 = shapely.union_all(geoms).bounds
    _require_lonlat(lon0, lat0, str(path))
    _require_lonlat(lon1, lat1, str(path))
    return LocalFrame(lon0=lon0, lat0=lat0, lat_ref=0.5 * (lat0 + lat1))


def region_bounds_enu(path: str | Path, frame: LocalFrame) -> tuple[float, float, float, float]:
    """区域轮廓的水平 ENU 包络 ``(x0, y0, x1, y1)``，供全域场取覆盖范围。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    geoms = [shapely_shape(f["geometry"]) for f in data["features"] if f.get("geometry")]
    if not geoms:
        raise ValueError(f"no geometry in {path}")
    lon0, lat0, lon1, lat1 = shapely.union_all(geoms).bounds
    corners = [frame.to_local(lon, lat) for lon, lat in ((lon0, lat0), (lon0, lat1), (lon1, lat0), (lon1, lat1))]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return min(xs), min(ys), max(xs), max(ys)


def load_nofly(
    path: str | Path,
    frame: LocalFrame,
    *,
    dem: Dem | None = None,
    altitude_mode: str = DEFAULT_NOFLY_ALTITUDE_MODE,
) -> tuple[NoFlyZone, ...]:
    """读 ``nofly.geojson``（EPSG:4326）→ 局地 ENU 的静态禁飞棱柱。

    输入边界支持两种**互斥**的高度口径，内部只输出绝对 z：

    * ``altitude_mode='absolute'``（默认，库内统一存储口径）：读 ``z_min_m`` / ``z_max_m``，
      缺省 0 / 1000 m；出现 ``agl_*`` 字段直接报错。
    * ``altitude_mode='agl'``：读 ``agl_min_m`` / ``agl_max_m``，用 DEM 把每个 Polygon
      覆盖到的单元四角 min/max 换算成保守的绝对 z；没有 DEM、越出 DEM、或混入 ``z_*`` 都报错。

    ``NoFlyZone`` 本身永远只存绝对 z，建场、SDF、终检不再关心输入是哪种口径。
    """
    altitude_mode = validate_altitude_mode(altitude_mode)
    if altitude_mode == NOFLY_ALTITUDE_AGL and dem is None:
        raise ValueError("nofly_altitude_mode='agl' 需要 DEM：请开启 with_dem=True 并确保 dem.tif 存在")

    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    features = data.get("features", [])
    if not isinstance(features, list):
        raise TypeError(f"{path} 的 features 必须是数组")

    out: list[NoFlyZone] = []
    for index, feature in enumerate(features):
        source = f"{path.name}[{index}]"
        geom = shapely_shape(feature["geometry"])
        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)
        else:
            raise ValueError(f"{source} 的 geometry 必须是 Polygon / MultiPolygon，收到 {geom.geom_type}")
        props = feature.get("properties") or {}
        z_mode, z_low, z_high = parse_nofly_altitude(props, source, altitude_mode)
        reason = props.get("reason", "")
        reason = "" if reason is None else str(reason)

        for poly in polys:
            if poly.is_empty or len(poly.interiors) > 0:
                raise ValueError(f"{source} 不支持空 Polygon 或带内环的 Polygon，请拆成多个禁飞区")
            ring = list(poly.exterior.coords)[:-1]
            if len(ring) < 3:
                raise ValueError(f"{source} 的 Polygon 顶点不足 3 个")
            for lon, lat in ring:
                _require_lonlat(float(lon), float(lat), source)
            footprint = tuple(frame.to_local(float(lon), float(lat)) for lon, lat in ring)
            if Polygon(footprint).area <= 0.0:
                raise ValueError(f"{source} 转成 ENU 后面积为零，请检查坐标顺序")

            if z_mode == NOFLY_ALTITUDE_ABSOLUTE:
                z_min, z_max = z_low, z_high
            else:
                assert dem is not None
                ground_min, ground_max = ground_bounds(poly, dem, source)
                z_min, z_max = ground_min + z_low, ground_max + z_high
            if z_min >= z_max:
                raise ValueError(f"{source} 换算后的 z_min({z_min}) 必须小于 z_max({z_max})")
            out.append(NoFlyZone(footprint=footprint, z_min_m=z_min, z_max_m=z_max, reason=reason))
    return tuple(out)


def _to_float(value: object, default: float = 0.0) -> float:
    """只接受**有限正数**；NaN / Inf / 文本一律走 default。

    Inf 以前会漏过去（``float("inf") > 0`` 为真），栅格膨胀直接把整片空间标成障碍。
    """
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return default
    return out if math.isfinite(out) and out > 0.0 else default


def _largest_polygon(geom: shapely.Geometry) -> Polygon | None:
    if geom.geom_type == "Polygon":
        return geom  # type: ignore[return-value]
    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda g: g.area)  # type: ignore[attr-defined]
    return None


def _height_or_raise(raw: object, source: str, height_field: str) -> float:
    height = _to_float(raw)
    if height <= 0.0:
        raise ValueError(f"{source} 的高度字段 {height_field!r} 缺失或非正：{raw!r}")
    return height


def _poly_ring(
    geom: shapely.Geometry, m_per_deg2: float, min_area_m2: float
) -> list[tuple[float, float]] | None:
    """Polygon / MultiPolygon → 外环顶点；其余几何类型与碎片返回 None。

    MultiPolygon 只取最大一块、带洞按外环填实：两种建筑格式共用这一口径。
    """
    poly = _largest_polygon(geom)
    if poly is None or poly.is_empty or poly.area * m_per_deg2 < min_area_m2:
        return None
    ring = list(poly.exterior.coords)[:-1]
    return ring if len(ring) >= 3 else None


def _load_buildings_shp(
    path: Path,
    frame: LocalFrame,
    height_field: str,
    m_per_deg2: float,
    min_area_m2: float,
    out: list[Building],
) -> None:
    reader = shapefile.Reader(str(path), encoding="gbk", encodingErrors="replace")
    try:
        source = path.stem       # 来源文件名，不是建筑类别；类别字段广州数据里没有
        bx0, by0, bx1, by1 = reader.bbox
        _require_lonlat(bx0, by0, str(path))
        _require_lonlat(bx1, by1, str(path))
        names = [f[0] for f in reader.fields[1:]]
        for shp, rec in zip(reader.shapes(), reader.records(), strict=False):
            fields = dict(zip(names, list(rec)))
            height = _height_or_raise(fields.get(height_field), path.name, height_field)
            geom = shapely_shape(shp.__geo_interface__)
            ring = _poly_ring(geom, m_per_deg2, min_area_m2)
            if ring is not None:
                out.append(
                    Building(
                        footprint=tuple(frame.to_local(lon, lat) for lon, lat in ring),
                        height_m=height,
                        base_z_m=0.0,
                        source=source,
                    )
                )
    finally:
        reader.close()


def _load_buildings_geojson(
    path: Path,
    frame: LocalFrame,
    height_field: str,
    m_per_deg2: float,
    min_area_m2: float,
    out: list[Building],
) -> None:
    """读 ``building/*.geojson``：UTF-8，高度取 ``properties[height_field]``。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    features = data.get("features", [])
    if not isinstance(features, list):
        raise TypeError(f"{path.name} 的 features 必须是数组")
    for index, feature in enumerate(features):
        where = f"{path.name}[{index}]"
        ring = _poly_ring(shapely_shape(feature["geometry"]), m_per_deg2, min_area_m2)
        if ring is None:
            continue
        for lon, lat in ring:
            _require_lonlat(float(lon), float(lat), where)
        props = feature.get("properties") or {}
        height = _height_or_raise(props.get(height_field), where, height_field)
        out.append(
            Building(
                footprint=tuple(frame.to_local(lon, lat) for lon, lat in ring),
                height_m=height,
                base_z_m=0.0,
                source=path.stem,
            )
        )


def load_buildings(
    shp_dir: str | Path,
    frame: LocalFrame,
    height_field: str = "Height",
    min_area_m2: float = MIN_BUILDING_AREA_M2,
) -> list[Building]:
    """读一个目录下所有 ``*.shp``（DBF 为 GBK）与 ``*.geojson``（UTF-8），顶层、不递归。

    两种格式共用口径：高度必须是**有限正数**（米，相对地面），缺失/非正直接抛
    ``ValueError`` 并指出文件；MultiPolygon 只取最大一块；带洞按外环填实；
    面积 < ``min_area_m2`` 的碎片直接丢。坐标都必须是 EPSG:4326 经纬度。

    一栋可用建筑都没有时**报错**：以前只有 shp 被扫描，目录里放 geojson 会静默
    得到空世界（楼不挡路却一路返回 ok），这里直接钉死。
    """
    m_per_deg2 = frame.m_per_deg_lon * M_PER_DEG_LAT
    directory = Path(shp_dir)
    # 大小写不敏感：Windows/GIS 工具导出的 ``Residential.SHP`` / ``Block.GeoJSON`` 不能
    # 静默消失（Linux glob 区分大小写）。楼悄悄少了 = 障碍图缺洞 = 规划“撞楼”，比报错严重得多。
    entries = sorted(p for p in directory.iterdir()
                     if p.is_file() and p.suffix.lower() in (".shp", ".geojson"))
    files = [p for p in entries if p.suffix.lower() == ".shp"] + \
            [p for p in entries if p.suffix.lower() == ".geojson"]
    out: list[Building] = []
    for path in files:
        if path.suffix.lower() == ".geojson":
            _load_buildings_geojson(path, frame, height_field, m_per_deg2, min_area_m2, out)
        else:
            _load_buildings_shp(path, frame, height_field, m_per_deg2, min_area_m2, out)
    if not out:
        raise ValueError(
            f"{directory} 下没有可用建筑（*.shp / *.geojson）：检查文件是否齐全、"
            f"几何是否为 Polygon、高度字段 {height_field!r} 是否为有限正数"
        )
    return out


def filter_by_bbox(
    prisms: Iterable[_Prism],
    bounds: tuple[float, float, float, float],
    pad_m: float = 0.0,
) -> list[_Prism]:
    """粗筛：只留**水平** bbox（``x0, y0, x1, y1``，米）外扩 ``pad_m`` 后相交的棱柱（建筑/禁飞区）。

    这里判的是 footprint 的包络 vs 窗口包络（两个矩形的相交），不做精确实体相交；
    真正的"挡路"由 ``build_occupancy`` 逐格判。所以粗筛数量 ≥ 实际挡路数。

    ``pad_m`` 用于**净空度量**：真几何净空要查路径周边 ``CLEARANCE_SEARCH_RADIUS_M``，
    而路径可以贴着窗口边界走，窗外几米的障碍也算数——不外扩就会把它漏掉、把净空虚报成 30 m。
    """
    x0, y0, x1, y1 = bounds
    x0, y0, x1, y1 = x0 - pad_m, y0 - pad_m, x1 + pad_m, y1 + pad_m
    out: list[_Prism] = []
    for b in prisms:
        bx0 = min(p[0] for p in b.footprint)
        bx1 = max(p[0] for p in b.footprint)
        by0 = min(p[1] for p in b.footprint)
        by1 = max(p[1] for p in b.footprint)
        if bx0 <= x1 and bx1 >= x0 and by0 <= y1 and by1 >= y0:
            out.append(b)
    return out

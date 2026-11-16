"""禁飞区输入高度口径：AGL → 绝对 z 的入口适配。

只服务 ``load_nofly``。库内 ``NoFlyZone`` 永远只存绝对 z；这里做完换算后，
后续 SDF / occupancy / 终检不再出现第二套 AGL 语义。
"""

from __future__ import annotations

import math

import numpy as np
from shapely import contains_xy
from shapely.geometry import Polygon

from ..contract.defaults import (
    NOFLY_ALTITUDE_ABSOLUTE,
    NOFLY_ALTITUDE_AGL,
)
from .dem_terrain import Dem


def validate_altitude_mode(mode: str) -> str:
    """入口只认两种口径；值写错在加载期报，不静默落到某个默认。"""
    if mode not in (NOFLY_ALTITUDE_ABSOLUTE, NOFLY_ALTITUDE_AGL):
        raise ValueError(
            f"nofly_altitude_mode 必须是 {NOFLY_ALTITUDE_ABSOLUTE!r} 或 "
            f"{NOFLY_ALTITUDE_AGL!r}，收到 {mode!r}"
        )
    return mode


def _zone_z(props: dict, key: str, default: float, source: str) -> float:
    """高度字段缺失走契约默认，给了就必须有限，不接受 NaN/Inf。"""
    raw = props.get(key, default)
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{source} 的 {key} 需要有限数值，收到 {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{source} 的 {key} 必须是有限数值，收到 {raw!r}")
    return value


def _dem_extent(dem: Dem) -> tuple[float, float, float, float]:
    """DEM 有效栅格中心的经纬度包络 ``(lon_min, lat_min, lon_max, lat_max)``。"""
    rows, cols = dem.shape
    return (
        dem.lon_ul,
        dem.lat_ul - (rows - 1) * dem.dlat,
        dem.lon_ul + (cols - 1) * dem.dlon,
        dem.lat_ul,
    )


def ground_bounds(poly: Polygon, dem: Dem, source: str) -> tuple[float, float]:
    """AGL→绝对 z 的地面高程范围：多边形顶点 + **覆盖到的 DEM 单元四角**，取 min/max。

    ``Dem.at_lonlat`` 是双线性插值；单元内任一点都是该单元四角值的凸组合，因此四角
    min/max 是该单元地面值的保守包络。只看顶点或格心会在单元内部/边上的极值处漏值。
    """
    lon_min, lat_min, lon_max, lat_max = _dem_extent(dem)
    minx, miny, maxx, maxy = poly.bounds
    tol = 1e-9
    if minx < lon_min - tol or maxx > lon_max + tol or miny < lat_min - tol or maxy > lat_max + tol:
        raise ValueError(
            f"{source} 的 AGL 禁飞区超出 DEM 覆盖范围 "
            f"({lon_min}, {lat_min})–({lon_max}, {lat_max})；AGL 必须有真实地面高程"
        )

    rows = dem.shape[0]
    cols = dem.shape[1]
    if rows < 2 or cols < 2:
        raise ValueError(f"{source} 的 AGL 禁飞区需要至少 2×2 的 DEM 才能取地面包络")

    ring = np.asarray(poly.exterior.coords, dtype=float)
    samples: list[np.ndarray] = [dem.at_lonlat(ring[:, 0], ring[:, 1])]

    # 找到与多边形相交的 DEM 单元：单元中心到多边形的距离不会超过半格对角线。
    # 取这些单元的四角值，得到双线性地面的保守包络（会多包住一些空域）。
    half_diag = 0.5 * math.hypot(dem.dlon, dem.dlat)
    buffered = poly.buffer(half_diag + 1e-12)
    c0 = max(int(np.floor((minx - half_diag - dem.lon_ul) / dem.dlon - 0.5)), 0)
    c1 = min(int(np.ceil((maxx + half_diag - dem.lon_ul) / dem.dlon - 0.5)), cols - 2)
    r0 = max(int(np.floor((dem.lat_ul - maxy - half_diag) / dem.dlat - 0.5)), 0)
    r1 = min(int(np.ceil((dem.lat_ul - miny + half_diag) / dem.dlat - 0.5)), rows - 2)
    if c1 >= c0 and r1 >= r0:
        cell_lon = dem.lon_ul + (np.arange(c0, c1 + 1) + 0.5) * dem.dlon
        cell_lat = dem.lat_ul - (np.arange(r0, r1 + 1) + 0.5) * dem.dlat
        grid_lon, grid_lat = np.meshgrid(cell_lon, cell_lat)
        mask = contains_xy(buffered, grid_lon, grid_lat)
        if mask.any():
            rr, cc = np.nonzero(mask)
            corners = [dem.z_m[r:r + 2, c:c + 2].reshape(-1) for r, c in zip(rr, cc)]
            samples.append(np.concatenate(corners))
        else:
            # 极小多边形可能因浮点边界漏掉格心；退到代表点所在单元的四角。
            rep = poly.representative_point()
            c = min(max(int(np.floor((rep.x - dem.lon_ul) / dem.dlon)), 0), cols - 2)
            r = min(max(int(np.floor((dem.lat_ul - rep.y) / dem.dlat)), 0), rows - 2)
            samples.append(dem.z_m[r:r + 2, c:c + 2].reshape(-1))

    values = np.concatenate(samples)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"{source} 的 AGL 禁飞区在 DEM 上取不到有效地面高程")
    return float(values.min()), float(values.max())


def parse_nofly_altitude(props: dict, source: str, altitude_mode: str) -> tuple[str, float, float]:
    """解析 Feature 的高度字段；只允许一种口径，混用直接报错。"""
    if altitude_mode == NOFLY_ALTITUDE_ABSOLUTE:
        if "agl_min_m" in props or "agl_max_m" in props:
            raise ValueError(
                f"{source} 使用 absolute 模式，不能同时提供 agl_min_m / agl_max_m；"
                "AGL 数据请显式设置 world.nofly_altitude_mode='agl'"
            )
        z_min = _zone_z(props, "z_min_m", 0.0, source)
        z_max = _zone_z(props, "z_max_m", 1000.0, source)
        if z_min >= z_max:
            raise ValueError(f"{source} 的 z_min_m({z_min}) 必须小于 z_max_m({z_max})")
        return NOFLY_ALTITUDE_ABSOLUTE, z_min, z_max

    if "z_min_m" in props or "z_max_m" in props:
        raise ValueError(
            f"{source} 使用 agl 模式，不能同时提供 z_min_m / z_max_m；"
            "绝对高程数据请使用 world.nofly_altitude_mode='absolute'"
        )
    if "agl_min_m" not in props or "agl_max_m" not in props:
        raise ValueError(f"{source} 使用 agl 模式，必须同时提供 agl_min_m 和 agl_max_m")
    agl_min = _zone_z(props, "agl_min_m", 0.0, source)
    agl_max = _zone_z(props, "agl_max_m", 0.0, source)
    if agl_min >= agl_max:
        raise ValueError(f"{source} 的 agl_min_m({agl_min}) 必须小于 agl_max_m({agl_max})")
    return NOFLY_ALTITUDE_AGL, agl_min, agl_max

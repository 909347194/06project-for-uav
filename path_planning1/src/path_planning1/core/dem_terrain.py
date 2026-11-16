"""DEM → 逐点地面高程，以及"地形即障碍"所需的栅格地面。

为什么需要它：广州数据的地面并不平（高程 5–75 m，起降点附近 12–14 m），而飞行带
10–120 m 是**真高（AGL）**。没有地形，起点定在 25 m 时，在 40 m 的山包上就是地下。

口径（与 ``path_planning1/README.md``「实测基线」一致，别在这里改）：
* DEM 用 GeoTIFF 自带的地理标签（EPSG:4326 经纬度），**不引入新依赖**（PIL 就够）。
* 地面当**实体**：体素中心低于地面即占据；净空由 SDF 统一给。
* ``base_z_m`` 取 footprint 内 DEM 的**最大值**：宁可把楼垫高，也不让它被地形切一半。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, map_coordinates

from ..contract import M_PER_DEG_LAT, Building, GridSpec, LocalFrame

_TAG_PIXEL_SCALE = 33550
_TAG_TIEPOINT = 33922
_TAG_NODATA = 42113


@dataclass(frozen=True)
class Dem:
    """经纬度网格上的高程（米）。``z_m`` 已填好 NoData，不含 NaN。"""

    z_m: np.ndarray      # (rows, cols) float32
    lon_ul: float        # 左上角经度
    lat_ul: float        # 左上角纬度
    dlon: float          # 每列经度步长（正）
    dlat: float          # 每行纬度步长（正，向南）

    @property
    def shape(self) -> tuple[int, int]:
        return self.z_m.shape

    def at_lonlat(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """双线性采样（域外按边缘值，不外推）。"""
        cols = (np.asarray(lon, dtype=float) - self.lon_ul) / self.dlon
        rows = (self.lat_ul - np.asarray(lat, dtype=float)) / self.dlat
        return np.asarray(
            map_coordinates(self.z_m, [rows, cols], order=1, mode="nearest"), dtype=float
        )


def load_dem(path: str | Path) -> Dem:
    """读 GeoTIFF（EPSG:4326）。NoData 用最近邻有效格填平，避免边界出现坑。"""
    im = Image.open(path)
    if _TAG_PIXEL_SCALE not in im.tag_v2 or _TAG_TIEPOINT not in im.tag_v2:
        raise ValueError(f"{path} 没有 GeoTIFF 地理标签（33550/33922）")
    scale = im.tag_v2[_TAG_PIXEL_SCALE]
    tie = im.tag_v2[_TAG_TIEPOINT]
    dlon, dlat = float(scale[0]), float(abs(scale[1]))
    lon_ul, lat_ul = float(tie[3]), float(tie[4])

    z = np.asarray(im, dtype=np.float32)
    bad = ~np.isfinite(z)
    nodata = im.tag_v2.get(_TAG_NODATA)
    if nodata is not None:
        bad |= z == np.float32(float(nodata))
    if bad.any():
        idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
        z = z[tuple(idx)]
    return Dem(z_m=z, lon_ul=lon_ul, lat_ul=lat_ul, dlon=dlon, dlat=dlat)


def ground_at_xy(dem: Dem, frame: LocalFrame, xy: np.ndarray) -> np.ndarray:
    """ENU 水平坐标 ``(..., 2)`` → 地面高程（米）。"""
    xy = np.atleast_2d(np.asarray(xy, dtype=float))
    lon = frame.lon0 + xy[:, 0] / frame.m_per_deg_lon
    lat = frame.lat0 + xy[:, 1] / M_PER_DEG_LAT
    return dem.at_lonlat(lon, lat)


def grounds_at_endpoints(
    dem: Dem | None, frame: LocalFrame,
    start_xy: tuple[float, float], goal_xy: tuple[float, float],
) -> tuple[float, float]:
    """两端的 ``z_ground``（米）：DEM 双线性采样；``dem=None`` 取 0（平地）。"""
    if dem is None:
        return 0.0, 0.0
    g = ground_at_xy(dem, frame, np.array([list(start_xy[:2]), list(goal_xy[:2])], dtype=float))
    return float(g[0]), float(g[1])


@dataclass(frozen=True)
class TerrainField:
    """窗口栅格上的地面高程（与 ``GridSpec`` 的 xy 列对齐，``(nx, ny)``）。"""

    grid: GridSpec
    ground_m: np.ndarray

    @property
    def z_range(self) -> tuple[float, float]:
        return float(self.ground_m.min()), float(self.ground_m.max())

    def at(self, pts: np.ndarray) -> np.ndarray:
        """任意 ENU 点 ``(..., 3)`` 处的地面高程（双线性；斜栅格走 ``enu_to_index``）。"""
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        idx = self.grid.enu_to_index(pts)
        return np.asarray(
            map_coordinates(self.ground_m, [idx[:, 0], idx[:, 1]], order=1, mode="nearest"),
            dtype=float,
        )

    def at_xy(self, x: float, y: float) -> float:
        """单点地面高程（m）。取单点时用它，别写 ``at(np.array([[...]]))[0]``。"""
        return float(self.at(np.array([[float(x), float(y), 0.0]]))[0])

    def band_ok(self, pts: np.ndarray, agl_min_m: float, agl_max_m: float) -> np.ndarray:
        """真高（AGL）判定：``ground + agl_min ≤ z ≤ ground + agl_max``。"""
        z = np.atleast_2d(np.asarray(pts, dtype=float))[:, 2]
        g = self.at(pts)
        return (z >= g + float(agl_min_m)) & (z <= g + float(agl_max_m))


def terrain_field(dem: Dem, frame: LocalFrame, grid: GridSpec) -> TerrainField:
    """按栅格的 x/y 铺一张地面高程表（只看水平，与 z 无关）。

    格心 ENU 坐标一律走 ``grid.index_to_enu``：``grid.yaw_rad≠0`` 时窗口是斜的，
    自己用 ``origin + i·sx`` 铺会整体错位（DEM 采样就会错到别处的山包上）。
    """
    nx, ny, _ = grid.shape
    ii, jj = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float), indexing="ij")
    idx = np.stack([ii, jj, np.zeros_like(ii)], axis=-1)
    enu = grid.index_to_enu(idx)
    ground = ground_at_xy(dem, frame, enu.reshape(-1, 3)[:, :2])
    return TerrainField(grid=grid, ground_m=ground.reshape(nx, ny))


def attach_base_z(
    buildings: list[Building], dem: Dem, frame: LocalFrame, min_z_m: float = 0.0
) -> list[Building]:
    """给每栋楼填 ``base_z_m``（底面顶点 + 中心处 DEM 的最大值，不低于 ``min_z_m``）。"""
    out: list[Building] = []
    for b in buildings:
        fp = np.asarray(b.footprint, dtype=float)
        z = float(np.max(ground_at_xy(dem, frame, np.vstack([fp, fp.mean(axis=0)[None, :]]))))
        out.append(replace(b, base_z_m=max(z, float(min_z_m))))
    return out

"""建筑 → ENU 3D occupancy / SDF（窗口内两遍 EDT）。

口径照搬参考实现的三条（见 ``path_planning1/README.md``「口径」）：
**轴 2 = 高度**、**SDF 单位米且正值 = 自由空间**、**在完整高度上算完再裁带**。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates
from shapely import contains_xy
from shapely.geometry import Polygon

from ..contract import Building, GridSpec, NoFlyZone
from .dem_terrain import TerrainField


def _axis(origin: float, spacing: float, n: int) -> np.ndarray:
    return origin + np.arange(n, dtype=float) * spacing


def build_occupancy(
    buildings: Sequence[Building],
    grid: GridSpec,
    terrain: TerrainField | None = None,
    nofly: Sequence[NoFlyZone] = (),
) -> np.ndarray:
    """逐棱柱填充（建筑 + 静态禁飞区）：格心落在棱柱内即视为占据。

    **保守膨胀**（这一条是净空口径的地基，不要改成"裸格心判据"）：
    每个 footprint 先 ``buffer(半格对角线)``、竖直区间各外扩 ``半格`` 再判格心。理由两个：

    * **整栋漏检**：只测格心时，比格距还窄的房子可能一个格心都不落进去。膨胀半个格对角线后，
      任何棱柱至少含一个格心（膨胀后区域半径 ≥ 格点最大空隙 ``√2·s/2``），漏检在数学上不可能发生。
    * **障碍被缩**：裸格心判据让障碍最多"缩"半个格，SDF 于是比真几何乐观；膨胀之后障碍
      只大不小，SDF 变成保守下界，净空不会被虚报。

    禁飞区与建筑走**同一套**保守膨胀，但只填自己的 ``[z_min_m, z_max_m]``：悬浮层不会被
    地形填充成从地面长上来的实心柱；反过来，从它上方/下方绕过仍然合法。

    成本 ∝ 占位面积（只在该棱柱 bbox 的格上做 point-in-polygon），不随窗口体积线性增长。
    ``grid.yaw_rad≠0`` 时格心坐标走 ``grid.index_to_enu``（斜栅格），棱柱先投影到栅格两轴，
    再只测它覆盖的那一小块格。``terrain`` 给出时地面当**实体**：中心低于地面的体素一律占据
    （否则路径会钻地）——这条对所有棱柱一视同仁，地面以下本来就不可飞。
    """
    nx, ny, nz = grid.shape
    sx, sy, sz = (float(v) for v in grid.spacing_m)
    ox, oy, oz = (float(v) for v in grid.origin_enu)
    a0, a1 = grid.axes_xy
    half_xy = 0.5 * float(np.hypot(sx, sy))     # 半格对角线：格心到格内任一点的上界
    half_z = 0.5 * sz
    occ = np.zeros((nx, ny, nz), dtype=bool)
    prisms: list[tuple[tuple[tuple[float, float], ...], float, float]] = [
        (b.footprint, b.base_z_m, b.z_top_m) for b in buildings
    ]
    prisms.extend((z.footprint, z.z_min_m, z.z_max_m) for z in nofly)
    if prisms:
        ui = np.arange(nx, dtype=float) * sx
        vj = np.arange(ny, dtype=float) * sy
        # 全窗口格心的 ENU 坐标（斜栅格 = 原点 + i·sx·a0 + j·sy·a1）
        gx_full = ox + ui[:, None] * a0[0] + vj[None, :] * a1[0]
        gy_full = oy + ui[:, None] * a0[1] + vj[None, :] * a1[1]
        origin_xy = np.array([ox, oy], dtype=float)
        for footprint, z_low, z_high in prisms:
            poly = Polygon(footprint)
            if poly.is_empty:
                continue
            poly = poly.buffer(half_xy, quad_segs=2)
            if poly.is_empty:
                continue
            minx, miny, maxx, maxy = poly.bounds
            corners = np.array([[minx, miny], [maxx, miny], [minx, maxy], [maxx, maxy]])
            pu, pv = (corners - origin_xy) @ a0, (corners - origin_xy) @ a1
            i0 = max(int(np.floor(pu.min() / sx)) - 1, 0)
            i1 = min(int(np.ceil(pu.max() / sx)) + 1, nx)
            j0 = max(int(np.floor(pv.min() / sy)) - 1, 0)
            j1 = min(int(np.ceil(pv.max() / sy)) + 1, ny)
            if i0 >= i1 or j0 >= j1:
                continue
            k0 = max(int(np.ceil((z_low - half_z - oz) / sz)), 0)
            k1 = min(int(np.floor((z_high + half_z - oz) / sz)), nz - 1)
            if k0 > k1:
                continue
            mask = contains_xy(poly, gx_full[i0:i1, j0:j1], gy_full[i0:i1, j0:j1])
            occ[i0:i1, j0:j1, k0 : k1 + 1] |= mask[:, :, None]
    if terrain is not None:
        occ |= terrain.ground_m[:, :, None] >= _axis(oz, sz, nz)[None, None, :]
    return occ


def build_sdf(occupancy: np.ndarray, spacing_m: tuple[float, float, float]) -> np.ndarray:
    """有符号距离（米，正 = 自由空间）。一遍 EDT 拿不到符号，必须两遍。

    ``edt(x)`` 的语义是"到最近 0 体素的距离"，所以：
    自由区的距离 = ``edt(~occ)``（建筑是 0），建筑内部的深度 = ``edt(occ)``（自由是 0）。
    这两个**别接反**——接反了整张 SDF 全是 0，路径会笔直穿楼。
    """
    spacing = tuple(float(v) for v in spacing_m)
    d_body = distance_transform_edt(occupancy, sampling=spacing)
    d_free = distance_transform_edt(~occupancy, sampling=spacing)
    return np.where(occupancy, -d_body, d_free).astype(np.float32)


@dataclass
class SdfField:
    """SDF 的查询封装：三线性插值 + 段内最值，供 RRT / NLP 共用同一份数据。"""

    grid: GridSpec
    sdf: np.ndarray

    def __post_init__(self) -> None:
        if tuple(self.sdf.shape) != tuple(self.grid.shape):
            raise ValueError(f"sdf shape {self.sdf.shape} != grid {self.grid.shape}")

    @property
    def occupied_ratio(self) -> float:
        return float(np.count_nonzero(self.sdf < 0.0)) / float(self.sdf.size)

    def at(self, points: np.ndarray) -> np.ndarray:
        """ENU 连续坐标处的插值 SDF，形状 ``points.shape[:-1]``（窗口外按边缘值，不外推）。"""
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        idx = self.grid.enu_to_index(pts)
        values = map_coordinates(
            self.sdf, [idx[:, 0], idx[:, 1], idx[:, 2]], order=1, mode="nearest"
        )
        return np.asarray(values, dtype=float)

    def segment_min(self, p0: np.ndarray, p1: np.ndarray, sample_step_m: float) -> float:
        """段内采样到的最小 SDF（净空复核的唯一入口，RRT 与结果校验都走它）。"""
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        length = float(np.linalg.norm(p1 - p0))
        n = max(2, int(np.ceil(length / max(sample_step_m, 1e-6))) + 1)
        ts = np.linspace(0.0, 1.0, n)[:, None]
        pts = p0 + ts * (p1 - p0)
        return float(self.at(pts).min())

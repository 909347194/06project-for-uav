"""全域预计算场：整座城市一次算完 occupancy / SDF，规划时只按走廊取样。

以前每个航段都把窗口内的楼重新栅格化 + 两遍 EDT（2.8 km 约 3–4 s）。这里把这一步
挪到**进程启动**：按全域轴对齐栅格算一次，之后每条 OD 只做一次三线性取样
（约 0.2 s），RRT / NLP 的输入形状与以前完全一样。

口径不变（与 ``core/occupancy_sdf.py`` 同源）：轴 2 = 高度、正值 = 自由空间、
建筑保守膨胀半格对角线、地面当实体。全域栅格是**轴对齐**的；走廊窗口仍是斜栅格，
所以取样必然跨格插值——这会改变 SDF 数值（基线要重测），见 README「全域场」一节。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates

from ..contract import Building, GridSpec, LocalFrame, NoFlyZone, RegionOfInterest
from ..contract.defaults import WORLD_FIELD_SAMPLE_BIAS_M
from .corridor_window import corridor_window
from .dem_terrain import Dem, TerrainField, terrain_field
from .occupancy_sdf import build_occupancy, build_sdf

_EPS_INDEX = 1e-6


class FieldContractError(RuntimeError):
    """建场契约被破坏：**部署事故**，不是调用方传错参数，也不该被映射成 400。

    覆盖窗口越界、场产物与当前配置/数据版本不一致、产物损坏都归这一类。它故意不继承
    ``ValueError``：后端通常把 ``ValueError`` 归为"调用方 bug / 4xx"，而这里该做的是
    告警 + 拒绝就绪 + 重新部署，重试同一条请求没有任何意义。
    """


class WindowOutsideField(FieldContractError):
    """窗口超出全域场覆盖范围：应在启动自检暴露，不在线兜底。"""


@dataclass(frozen=True)
class WorldField:
    """全域场：轴对齐栅格上的 occupancy + SDF（米，正 = 自由）。

    ``sdf`` 按 **z-major** 存放 ``(nz, nx, ny)``：按层取样时每个 z 平面是连续内存，
    否则 C 序 ``(nx,ny,nz)`` 的横向步长是 ``nz`` 个元素，6.9 M 点取样会掉到随机访存。
    ``occupancy`` 保持 ``(nx,ny,nz)``，只作离线检查，不在在线路径上。
    """

    grid: GridSpec
    sdf: np.ndarray
    occupancy: np.ndarray

    def __post_init__(self) -> None:
        nx, ny, nz = self.grid.shape
        if tuple(self.sdf.shape) != (nz, nx, ny):
            raise ValueError(f"sdf shape {self.sdf.shape} != {(nz, nx, ny)}")
        if tuple(self.occupancy.shape) != tuple(self.grid.shape):
            raise ValueError(
                f"occupancy shape {self.occupancy.shape} != grid {self.grid.shape}"
            )

    @property
    def nbytes(self) -> int:
        return int(self.sdf.nbytes + self.occupancy.nbytes)

    def _axis_bounds(self) -> tuple[float, float, float, float, float, float]:
        ox, oy, oz = (float(v) for v in self.grid.origin_enu)
        sx, sy, sz = (float(v) for v in self.grid.spacing_m)
        nx, ny, nz = self.grid.shape
        return ox, oy, oz, ox + (nx - 1) * sx, oy + (ny - 1) * sy, oz + (nz - 1) * sz

    def check_coverage(self, window: GridSpec) -> None:
        """窗口 8 个角点必须落在全域场的节点范围内，否则取样会静默取边缘值。

        公开出来给启动自检复用：建完场后应立即验证契约允许的最坏窗口，
        而不是等某条请求撞上来才发现 pad 不够。
        """
        nx, ny, nz = window.shape
        corners = np.array(
            [
                [i, j, k]
                for i in (0.0, nx - 1.0)
                for j in (0.0, ny - 1.0)
                for k in (0.0, nz - 1.0)
            ]
        )
        pts = window.index_to_enu(corners)
        x0, y0, z0, x1, y1, z1 = self._axis_bounds()
        sx, sy, sz = (float(v) for v in self.grid.spacing_m)
        tol = 0.5 * np.array([sx, sy, sz])
        if np.any(pts[:, :2].min(axis=0) < np.array([x0, y0]) - tol[:2]) or np.any(
            pts[:, :2].max(axis=0) > np.array([x1, y1]) + tol[:2]
        ) or float(pts[:, 2].min()) < z0 - tol[2] or float(pts[:, 2].max()) > z1 + tol[2]:
            raise WindowOutsideField(
                f"窗口 ENU 范围 {pts.min(axis=0).round(1)}–{pts.max(axis=0).round(1)} "
                f"超出全域场 {np.round([x0, y0, z0], 1)}–{np.round([x1, y1, z1], 1)}"
            )

    def window_sdf(self, window: GridSpec) -> np.ndarray:
        """按窗口节点取样，返回与 ``window.shape`` 同形的 SDF（逐 z 层采样，内存有界）。"""
        self.check_coverage(window)
        gx0, gy0, gz0 = (float(v) for v in self.grid.origin_enu)
        gsx, gsy, gsz = (float(v) for v in self.grid.spacing_m)
        nx, ny, nz = window.shape
        ox, oy, oz = (float(v) for v in window.origin_enu)
        sx, sy, sz = (float(v) for v in window.spacing_m)
        a0, a1 = window.axes_xy
        ii = np.arange(nx, dtype=float)[:, None] * sx
        jj = np.arange(ny, dtype=float)[None, :] * sy
        x = ox + ii * a0[0] + jj * a1[0]
        y = oy + ii * a0[1] + jj * a1[1]
        gx = ((x - gx0) / gsx).ravel()
        gy = ((y - gy0) / gsy).ravel()

        out = np.empty((nx, ny, nz), dtype=np.float32)
        planes: dict[int, np.ndarray] = {}

        def plane(index: int) -> np.ndarray:
            """按需取一个 z 平面（只取一次；相邻层窗口通常共用同一对全局平面）。"""
            cached = planes.get(index)
            if cached is None:
                cached = map_coordinates(
                    self.sdf[index], [gx, gy], order=1, mode="nearest"
                ).reshape(nx, ny).astype(np.float32, copy=False)
                planes[index] = cached
            return cached

        nz_global = self.grid.shape[2]
        for k in range(nz):
            gz = (oz + k * sz - gz0) / gsz
            k0 = min(max(int(np.floor(gz)), 0), nz_global - 1)
            k1 = min(k0 + 1, nz_global - 1)
            t = float(gz - k0) if k1 != k0 else 0.0
            out[:, :, k] = (1.0 - t) * plane(k0) + t * plane(k1)
            for stale in [key for key in planes if key < k0 - 1]:
                del planes[stale]
        return out

    def at(self, points: np.ndarray) -> np.ndarray:
        """连续 ENU 点的插值 SDF（全域场自身坐标；窗口外按边缘值）。"""
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        idx = self.grid.enu_to_index(pts)
        values = map_coordinates(
            self.sdf, [idx[:, 2], idx[:, 0], idx[:, 1]], order=1, mode="nearest"
        )
        return np.asarray(values, dtype=float)


def field_signature(field: WorldField | None) -> dict[str, Any]:
    """场身份 → ``params_hash(extra=...)``。``None`` 返回空字典（on_demand 不带场额外键）。

    为什么必须进指纹：startup 场版比窗口现算多一层三线性取样并扣
    ``WORLD_FIELD_SAMPLE_BIAS_M``，同一 OD 两种 SDF 基准可以给不同解；场参数变了
    （pad / spacing / shape）也一样。以前两者共用同一个 hash，缓存会静默命中另一口径的结果。

    ``sample_bias_m`` 也在这里：它只作用于场路径（``window_sdf(...) - bias``），改了会换一套
    SDF 基准却不改变格网形状，光靠 origin/spacing/shape 抓不到。放进**场身份**而不是全局
    参数还有个好处：on_demand 不用这个补偿，``field_signature(None)`` 仍是空字典。
    """
    if field is None:
        return {}
    g = field.grid
    return {
        "world_field": {
            "origin_enu": [round(float(v), 6) for v in g.origin_enu],
            "spacing_m": [float(v) for v in g.spacing_m],
            "shape": [int(v) for v in g.shape],
            "sample_bias_m": float(WORLD_FIELD_SAMPLE_BIAS_M),
        }
    }


def check_contract_coverage(
    field: WorldField,
    bounds_m: tuple[float, float, float, float],
    roi: RegionOfInterest,
    spacing_m: tuple[float, float, float],
    max_od_length_m: float,
    *,
    directions: int = 16,
) -> None:
    """启动期自检：契约允许的最坏 OD 的窗口必须落在场内，否则**现在就炸**。

    用 ``corridor_window`` 真实构造窗口（而不是复述 ``required_pad_m`` 的公式），公式写错 /
    配置被改坏都会被抓住。窗口 z 直接取场自身的竖直范围：自检只验证水平覆盖，
    z 在全域建场时已保证包住任何航段的真高带。
    """
    x0, y0, x1, y1 = (float(v) for v in bounds_m)
    gz0 = float(field.grid.origin_enu[2])
    gz1 = gz0 + (field.grid.shape[2] - 1) * float(field.grid.spacing_m[2])
    length = min(float(max_od_length_m), math.hypot(x1 - x0, y1 - y0))
    if length <= 0.0:
        raise FieldContractError(
            "启动自检没构造出任何合法的契约最坏 OD：检查 bounds_m 与 max_od_length_m"
        )
    corners = ((x0, y0), (x1, y0), (x0, y1), (x1, y1))
    n_dir = max(4, int(directions))
    checked = 0
    for k in range(n_dir):
        ang = 2.0 * math.pi * k / n_dir
        ux, uy = math.cos(ang), math.sin(ang)
        for sx0, sy0 in corners:
            gx, gy = sx0 + length * ux, sy0 + length * uy
            if not (x0 <= gx <= x1 and y0 <= gy <= y1):
                continue    # 端点出界：这条 OD 会被准入拒掉，不需要场覆盖
            w = corridor_window(
                (sx0, sy0, gz0), (gx, gy, gz0), roi, spacing_m, gz0, gz1
            )
            field.check_coverage(w.grid)
            checked += 1
    if checked == 0:
        raise FieldContractError(
            "启动自检没构造出任何合法的契约最坏 OD：检查 bounds_m 与 max_od_length_m"
        )


def build_world_field(
    buildings: Sequence[Building],
    frame: LocalFrame,
    dem: Dem | None,
    *,
    spacing_m: tuple[float, float, float],
    agl_min_m: float,
    agl_max_m: float,
    bounds_m: tuple[float, float, float, float],
    pad_m: float = 0.0,
    nofly: Sequence[NoFlyZone] = (),
) -> WorldField:
    """一次算完全域 occupancy / SDF（离线，约 1 分钟 / 0.6 GB）。

    ``bounds_m`` 是 ENU 水平包络 ``(x0, y0, x1, y1)``（``CityDataset.bounds_m``）；
    ``pad_m`` 是额外外扩。竖直包络取 ``[全城最低地面+agl_min, max(最高地面+agl_max,
    最高楼顶)+一格]``，保证任何窗口的真高档都能被覆盖——这比按窗口最低地面定天花板
    更贵（层数 56→90+），但全域只能有一份。禁飞区与建筑一起栅格化进 occupancy，
    但有自己的绝对 z 区间；不在飞行带内的部分不影响这组包络。
    """
    sx, sy, sz = (float(v) for v in spacing_m)
    x0, y0, x1, y1 = (float(v) for v in bounds_m)
    x0, y0 = x0 - pad_m, y0 - pad_m
    x1, y1 = x1 + pad_m, y1 + pad_m
    nx = math.ceil((x1 - x0) / sx) + 1
    ny = math.ceil((y1 - y0) / sy) + 1

    probe_grid = GridSpec(origin_enu=(x0, y0, 0.0), spacing_m=(sx, sy, sz), shape=(nx, ny, 1))
    terrain: TerrainField | None = terrain_field(dem, frame, probe_grid) if dem is not None else None
    ground_min, ground_max = terrain.z_range if terrain is not None else (0.0, 0.0)
    top = max((float(b.z_top_m) for b in buildings), default=ground_max)
    z_lo = ground_min + agl_min_m
    z_hi = max(ground_max + agl_max_m, top) + sz
    nz = math.ceil((z_hi - z_lo) / sz) + 1

    grid = GridSpec(origin_enu=(x0, y0, z_lo), spacing_m=(sx, sy, sz), shape=(nx, ny, nz))
    occupancy = build_occupancy(buildings, grid, terrain, nofly=nofly)
    sdf = build_sdf(occupancy, grid.spacing_m)
    # 转 z-major (nz, nx, ny)：在线按层取样时每个平面连续。转置瞬间多占一份 float32（约 0.7 GB），
    # 换在线取样从 1.4 s 降到亚秒级；建场是一次性的，这笔内存值得花。
    sdf = np.ascontiguousarray(sdf.transpose(2, 0, 1))
    return WorldField(grid=grid, sdf=sdf, occupancy=occupancy)

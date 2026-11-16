"""净空的**第二套实现**：不碰 shapely、不碰 ``BuildingPrisms``，用"边 × 竖直区间"重算一遍。

为什么需要它：``core/clearance_check.py`` 交付的净空（6.47/5.59/7.54 这类数字）只有一份代码
背书。之前所谓的"独立校验"其实是同一公式换个写法（shapely 2D 多边形距离 + 竖直间隔再
``sqrt(d²+dz²)``），能抓编码错误，抓不到口径错误（``base_z``/``z_top`` 用反、点在轮廓内但
竖向在外、凹多边形、邻域粗筛漏楼）。

这里的算法故意换一条路：

* 水平距离按**每条边**算点到线段距离并取最小（不调 shapely 的 polygon distance）；
* 轮廓内外用**射线奇偶**自己判（不调 shapely 的 contains）；
* 竖向间隔照棱柱定义算，最后合成的公式与生产实现一致——**公式一致是必须的**（棱柱的
  真距离就是 ``sqrt(dxy² + dz²)``），独立性来自"算 dxy 的路径不同"。
* 采样步长也自己写一遍，避免"两边共用同一个 densify、采样误差同向"。

口径边界（写清楚，别夸大）：两套实现共享同一批 ``Building`` 对象，所以 ``base_z_m`` 抬错、
footprint 顶点顺序错这类**数据**问题抓不到；它证明的是"几何量法没算错"。
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
from path_planning1.contract import Building


def _densify(path: np.ndarray, step_m: float) -> np.ndarray:
    """按 ``step_m`` 沿线补点（含两端）。"""
    pts = np.asarray(path, dtype=float)
    chunks = [pts[:1]]
    for a, b in itertools.pairwise(pts):
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / max(step_m, 1e-6))) + 1)
        chunks.append(np.linspace(a, b, n)[1:])
    return np.vstack(chunks)


def _point_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """射线奇偶判内（``pts`` (N,2)，``poly`` (M,2)）。"""
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    xi, yi = poly[:, 0][None, :], poly[:, 1][None, :]
    xj = np.roll(poly[:, 0], 1)[None, :]
    yj = np.roll(poly[:, 1], 1)[None, :]
    crosses = ((yi > y) != (yj > y)) & (
        x < (xj - xi) * (y - yi) / np.where(yj == yi, 1e-12, yj - yi) + xi
    )
    return (crosses.sum(axis=1) % 2) == 1


def _segment_dist(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """点到线段距离（逐边向量化）。"""
    ab = b - a
    t = np.clip(((pts - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    nearest = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(pts - nearest, axis=1)


def reference_min_clearance(
    path_enu: np.ndarray,
    buildings: Sequence[Building],
    *,
    sample_step_m: float = 1.0,
    search_radius_m: float = 30.0,
) -> float:
    """与 ``BuildingPrisms.min_clearance`` 同口径的最小 3D 净空，但计算路径不同。"""
    pts = _densify(path_enu, sample_step_m)
    if not buildings:
        return float(search_radius_m)
    best = np.full(len(pts), float(search_radius_m))
    r = float(search_radius_m)
    for b in buildings:
        poly = np.asarray(b.footprint, dtype=float)
        x0, y0 = poly.min(axis=0)
        x1, y1 = poly.max(axis=0)
        # 生产实现用"方框 ±r"粗筛，这里镜像同一语义，保证"邻域内没有楼 ⇒ 报 r"一致
        near = (
            (pts[:, 0] >= x0 - r) & (pts[:, 0] <= x1 + r)
            & (pts[:, 1] >= y0 - r) & (pts[:, 1] <= y1 + r)
        )
        idx = np.flatnonzero(near)
        if idx.size == 0:
            continue
        p = pts[idx, :2]
        inside = _point_in_polygon(p, poly)
        dxy = np.full(len(idx), np.inf)
        for k in range(len(poly)):
            a, bb = poly[k], poly[(k + 1) % len(poly)]
            dxy = np.minimum(dxy, _segment_dist(p, a, bb))
        dxy[inside] = 0.0
        z = pts[idx, 2]
        z0, z1 = float(b.base_z_m), float(b.z_top_m)
        dz = np.where((z >= z0) & (z <= z1), 0.0, np.minimum(np.abs(z - z0), np.abs(z - z1)))
        d = np.sqrt(dxy ** 2 + dz ** 2)
        best[idx] = np.minimum(best[idx], d)
    return float(best.min())


def worst_point(
    path_enu: np.ndarray,
    buildings: Sequence[Building],
    *,
    sample_step_m: float = 1.0,
    search_radius_m: float = 30.0,
) -> tuple[float, np.ndarray]:
    """最小净空 + 取到它的采样点（用来和生产实现的"最差点"逐点对账）。"""
    pts = _densify(path_enu, sample_step_m)
    best = np.full(len(pts), float(search_radius_m))
    r = float(search_radius_m)
    for b in buildings:
        poly = np.asarray(b.footprint, dtype=float)
        x0, y0 = poly.min(axis=0)
        x1, y1 = poly.max(axis=0)
        near = (
            (pts[:, 0] >= x0 - r) & (pts[:, 0] <= x1 + r)
            & (pts[:, 1] >= y0 - r) & (pts[:, 1] <= y1 + r)
        )
        idx = np.flatnonzero(near)
        if idx.size == 0:
            continue
        p = pts[idx, :2]
        inside = _point_in_polygon(p, poly)
        dxy = np.full(len(idx), np.inf)
        for k in range(len(poly)):
            a, bb = poly[k], poly[(k + 1) % len(poly)]
            dxy = np.minimum(dxy, _segment_dist(p, a, bb))
        dxy[inside] = 0.0
        z = pts[idx, 2]
        z0, z1 = float(b.base_z_m), float(b.z_top_m)
        dz = np.where((z >= z0) & (z <= z1), 0.0, np.minimum(np.abs(z - z0), np.abs(z - z1)))
        best[idx] = np.minimum(best[idx], np.sqrt(dxy ** 2 + dz ** 2))
    k = int(np.argmin(best))
    return float(best[k]), pts[k]

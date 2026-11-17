"""建筑底面 + 高度：ENU 点查询（纯 numpy，带均匀网格索引）。"""

from __future__ import annotations

import math

import numpy as np


class BuildingLayer:
    """建筑图层；``heights_at(xs, ys)`` 返回各点屋顶相对地面的高度（米）。"""

    def __init__(self, polygons, heights) -> None:
        polys = [np.asarray(p, dtype=float)[:, :2] for p in polygons]
        hs = np.asarray(heights, dtype=float).reshape(-1)
        if len(polys) != hs.size:
            raise ValueError(f"polygons({len(polys)}) 与 heights({hs.size}) 数量不一致")
        if np.any(~np.isfinite(hs)) or np.any(hs <= 0):
            raise ValueError("建筑高度必须是有限正数")
        self._polys = polys
        self._heights = hs
        self._index = self._build_index()

    @property
    def count(self) -> int:
        return len(self._polys)

    @classmethod
    def from_polygons(cls, polygons, heights) -> BuildingLayer:
        return cls(polygons, heights)

    def _build_index(self) -> dict:
        """按 bbox 把建筑塞进均匀网格；没有建筑时返回空索引。"""
        if not self._polys:
            return {}
        bounds = np.array(
            [
                [p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()]
                for p in self._polys
            ],
            dtype=float,
        )
        x0, y0 = float(bounds[:, 0].min()), float(bounds[:, 1].min())
        x1, y1 = float(bounds[:, 2].max()), float(bounds[:, 3].max())
        width, height = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
        cell = max(32.0, max(width, height) / max(1.0, math.sqrt(len(self._polys))))
        nx = max(1, math.ceil(width / cell))
        ny = max(1, math.ceil(height / cell))
        self._grid = (x0, y0, x1, y1, (x1 - x0) / nx, (y1 - y0) / ny, nx, ny)
        buckets: dict[tuple[int, int], list[int]] = {}
        for idx, (minx, miny, maxx, maxy) in enumerate(bounds):
            ix0 = max(0, min(nx - 1, int((minx - x0) / self._grid[4])))
            ix1 = max(0, min(nx - 1, int((maxx - x0) / self._grid[4])))
            iy0 = max(0, min(ny - 1, int((miny - y0) / self._grid[5])))
            iy1 = max(0, min(ny - 1, int((maxy - y0) / self._grid[5])))
            for ix in range(ix0, ix1 + 1):
                for iy in range(iy0, iy1 + 1):
                    buckets.setdefault((ix, iy), []).append(idx)
        return buckets

    def _candidates(self, x: float, y: float) -> list[int]:
        if not self._index:
            return []
        x0, y0, x1, y1, sx, sy, nx, ny = self._grid
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return []
        ix = max(0, min(nx - 1, int((x - x0) / sx)))
        iy = max(0, min(ny - 1, int((y - y0) / sy)))
        return self._index.get((ix, iy), [])

    @staticmethod
    def _contains(poly: np.ndarray, x: float, y: float) -> bool:
        """射线法；边界点按包含处理（相邻网格点不丢楼顶）。"""
        inside = False
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if (yi > y) != (yj > y):
                x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
                if x < x_cross:
                    inside = not inside
            elif abs(yi - y) < 1e-12 and min(xi, xj) <= x <= max(xi, xj):
                return True
            j = i
        return inside

    def heights_at(self, xs, ys) -> np.ndarray:
        xa = np.asarray(xs, dtype=float)
        ya = np.asarray(ys, dtype=float)
        if xa.shape != ya.shape:
            raise ValueError(f"xs{xa.shape} 与 ys{ya.shape} 形状不一致")
        out = np.zeros(xa.shape, dtype=float)
        for flat_idx, (x, y) in enumerate(zip(xa.ravel(), ya.ravel(), strict=True)):
            for idx in self._candidates(float(x), float(y)):
                if self._contains(self._polys[idx], float(x), float(y)):
                    out.ravel()[flat_idx] = max(out.ravel()[flat_idx], self._heights[idx])
        return out

"""真几何净空：路径折线到建筑棱柱的最小 3D 距离（**不经栅格**，独立于 SDF）。

为什么必须单独有这一份：栅格 SDF 有三个先天误差——格心判据会把障碍最多缩掉半个格
对角线、细楼可能整栋不进栅格、三线性插值再加半个格。未膨胀时实测"SDF 比真几何乐观"
最大 26 m、p99 4.6 m；即使按半格对角线把建筑膨胀之后，仍剩竖直量化（实测可达 2.2 m，
与竖直格距 2 m 同量级）。所以 SDF 读数只是"规划期近似"，**不能当作交付给后端的净空口径**；
这里用 shapely 直接量真几何，得到的是可以拿尺子复核的数（见 README「已知限制」）。

口径（三条，别引申）：

* 建筑视为**垂直棱柱** ``footprint × [base_z_m, z_top_m]``；点在棱柱正上/正下方时用竖直
  间距，其余位置用 ``sqrt(水平距离² + 竖直间距²)``。
* 地形不参与：地面由真高带（10–120 m）约束，路径本身贴着地面飞不了，地面不构成净空约束。
* 只在这个航段的**候选障碍**（建筑 + 禁飞区）上量；候选必须是"窗口外接矩形**再外扩**
  ``search_radius_m``"筛出来的那批（``core/route_segment_planner.py`` 就是这么筛的），
  否则窗外贴边的障碍会漏掉、净空被虚报成 30。
  量不到（这个半径内没有建筑）时返回 ``search_radius_m``，含义是"至少这么远"，不是精确值。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from shapely import STRtree, distance, points
from shapely import box as shapely_box
from shapely.geometry import Point, Polygon

from ..contract import Building, NoFlyZone, defaults


def densify(path: np.ndarray, step_m: float) -> np.ndarray:
    """把折线按 ``step_m`` 补点（含两端），供"沿线采样"类的检查使用。"""
    path = np.asarray(path, dtype=float)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    n = np.maximum(2, np.ceil(seg / max(step_m, 1e-6)).astype(int) + 1)
    chunks = [np.linspace(a, b, k) for a, b, k in zip(path[:-1], path[1:], n)]
    return np.vstack(chunks) if chunks else path


def _z_interval(prism: Building | NoFlyZone) -> tuple[float, float]:
    """建筑从地面长上来；禁飞区是绝对高程的悬浮层——两者都只是竖直区间。"""
    if isinstance(prism, Building):
        return prism.base_z_m, prism.z_top_m
    return prism.z_min_m, prism.z_max_m


class BuildingPrisms:
    """一批静态棱柱（建筑 + 禁飞区）的几何（STRtree + 竖直区间），一次构建、多次量。

    类名沿用历史名字（被实验脚本与测试引用），语义已扩成"所有静态障碍"：禁飞区与建筑
    用同一套距离公式，唯一区别是它的竖直区间由数据直接给出、不经 DEM 抬升。

    ``prisms`` 应是**窗口外接矩形外扩 ``search_radius_m``** 粗筛出的那批
    （见 ``core/city_loader.py::filter_by_bbox(..., pad_m=)`` 与 ``core/route_segment_planner.py``）。
    """

    def __init__(
        self,
        prisms: Sequence[Building | NoFlyZone],
        search_radius_m: float = defaults.CLEARANCE_SEARCH_RADIUS_M,
    ) -> None:
        self._polys = [Polygon(p.footprint) for p in prisms]
        intervals = [_z_interval(p) for p in prisms]
        self._z = np.array(intervals, dtype=float) if intervals else np.zeros((0, 2))
        self._tree = STRtree(self._polys) if self._polys else None
        self._geom = np.array(self._polys, dtype=object)
        self.search_radius_m = float(search_radius_m)

    def __len__(self) -> int:
        return len(self._polys)

    def contains(self, point: np.ndarray) -> bool:
        """点是否落在任一棱柱**内部或边界**（端点落在禁飞区里要单独失败）。"""
        if self._tree is None:
            return False
        x, y, z = (float(v) for v in point)
        hits = np.atleast_1d(self._tree.query(points([x, y])))
        probe = Point(x, y)
        for idx in hits:
            i = int(idx)
            z_lo, z_hi = self._z[i]
            if z_lo <= z <= z_hi and (self._geom[i].contains(probe) or self._geom[i].touches(probe)):
                return True
        return False

    def min_clearance(self, path: np.ndarray, sample_step_m: float = 1.0) -> float:
        """沿路径折线最小的 3D 净空（米）；棱柱内部为 0。

        **口径**：只在 ``search_radius_m`` 邻域内量，结果**一律截到该半径**；返回
        ``search_radius_m`` 就是“至少这么远”（附近什么都没有 / 真实距离超过量程都报它）。
        不能一部分报精确值、一部分报量程上限——那会让同一字段有两种语义，报表不可比。
        """
        return self.worst(path, sample_step_m)[0]

    def worst(self, path: np.ndarray, sample_step_m: float = 1.0) -> tuple[float, np.ndarray]:
        """最小净空 + 取到它的那个采样点（失败时告诉后端"哪一段贴得太近"）。

        邻域查询用**方框粗筛**（bbox 求交）而不是 ``dwithin``：后者会为每个查询点构造缓冲
        多边形，5 km 的航段上要十几秒；方框求交是 O(log n) 的纯索引比较。
        """
        path = np.asarray(path, dtype=float)
        if self._tree is None:
            return self.search_radius_m, path[0]
        pts = densify(path, sample_step_m)
        probe = points(pts[:, 0], pts[:, 1])
        r = self.search_radius_m
        boxes = shapely_box(pts[:, 0] - r, pts[:, 1] - r, pts[:, 0] + r, pts[:, 1] + r)
        hit_pt, hit_poly = self._tree.query(boxes)
        if len(hit_pt) == 0:
            # 整条航段 30 m 内没有楼：返回值是"至少这么远"，没有"最差点"可言，给起点充数
            return self.search_radius_m, path[0]
        d2 = np.asarray(distance(probe[hit_pt], self._geom[hit_poly]), dtype=float)
        z = pts[hit_pt, 2]
        lo, hi = self._z[hit_poly, 0], self._z[hit_poly, 1]
        dz = np.where((z >= lo) & (z <= hi), 0.0, np.minimum(np.abs(z - lo), np.abs(z - hi)))
        dist = np.sqrt(d2**2 + dz**2)
        k = int(np.argmin(dist))
        # bbox 命中但真实几何距离可能 > search_radius_m（L 形/大 footprint 的 bbox 近、
        # 几何远，老城里很常见）。量程外的值一律截到量程上限，与“无命中报量程”同口径。
        return min(float(dist[k]), self.search_radius_m), pts[hit_pt[k]]

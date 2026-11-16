"""RRT 3D（米制、SDF 自适应步长），采样域限制在 ROI 窗口内。

相对参考实现改了 3 处（见 ``path_planning1/README.md``「口径」）：步长全米制、
RNG 按 seed 在每次 ``plan`` 内局部构造、段内采样间隔按栅格自适应（不再固定 2 点）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..contract.defaults import DEFAULT_SEED
from .corridor_window import CorridorWindow
from .dem_terrain import TerrainField
from .occupancy_sdf import SdfField

_EPS_M = 1e-9   # 长度判零容差（米），只用于"两点是否重合"这类退化判断


@dataclass(frozen=True)
class RRTConfig:
    """RRT 运行参数。

    **没有独立默认值**：9 个字段全部必填，只能由 ``RRTConfig.from_planner(planner,
    clearance, seed)`` 派生（``RouteSegmentParams.rrt_config()`` 调用，净空阈值在那里加上
    SDF 容差）。实现细节常数也已上收到 ``PlannerParams``，与可调旋钮走同一条路——
    否则默认值会散成第三份，改 ``PlannerParams`` 却不生效。
    """

    step_m: float
    min_step_m: float
    sdf_step_scale: float
    max_iters: int
    goal_sample_rate: float
    clearance_m: float
    seed: int
    sample_step_m: float    # 段内净空采样间隔（与交付前的真几何复核用同一个步长）
    shortcut_iters: int

    @classmethod
    def from_planner(cls, planner, *, clearance_m: float, seed: int = DEFAULT_SEED) -> RRTConfig:
        """从契约 ``PlannerParams`` + 车辆净空派生；算法参数只在那里定义一次。"""
        return cls(
            step_m=planner.rrt_step_m,
            min_step_m=planner.rrt_min_step_m,
            sdf_step_scale=planner.rrt_sdf_step_scale,
            max_iters=planner.rrt_max_iters,
            goal_sample_rate=planner.rrt_goal_sample_rate,
            clearance_m=float(clearance_m),
            seed=int(seed),
            sample_step_m=planner.clearance_sample_step_m,
            shortcut_iters=planner.rrt_shortcut_iters,
        )


@dataclass(frozen=True)
class RRTResult:
    """一次 RRT 的产物。

    ``nodes`` = 树里留下多少节点；``path_points_before_shortcut`` = 捷径化**前**那条折线
    有几点。两个数不是一回事：旧实现把路径点数叫 ``nodes_before_shortcut``，
    于是"节点数"和"路径点数"混成一个字段。
    """

    path_enu: np.ndarray      # (K, 3)
    iterations: int           # 实际迭代次数（== 采样预算用掉多少）
    nodes: int                # 树节点数（含起点；直连时没有树，为 0）
    path_points_before_shortcut: int


class RRT3D:
    """单次 ``plan`` = 一次 RRT；``plan_many`` = 多初值（不同 seed 跑若干次）。

    ``alt_range_m`` 是**绝对** z 包络（窗口栅格能覆盖的范围）；给了 ``terrain`` +
    ``agl_range_m`` 时，采样还额外受**真高**约束（``ground + agl``），两者取交集。
    """

    def __init__(
        self,
        field: SdfField,
        window: CorridorWindow,
        alt_range_m: tuple[float, float],
        cfg: RRTConfig,
        terrain: TerrainField | None = None,
        agl_range_m: tuple[float, float] | None = None,
    ) -> None:
        self.field = field
        self.window = window
        self.alt_range = (float(alt_range_m[0]), float(alt_range_m[1]))
        self.cfg = cfg
        self.terrain = terrain
        self.agl = None if agl_range_m is None else (float(agl_range_m[0]), float(agl_range_m[1]))
        # 采样在**栅格索引空间**里做，再换回 ENU：窗口是斜的时候（走廊轴向栅格），
        # 在 ENU 外接矩形里均匀采会有一半点落在带外、被 _inside 白白丢掉。
        self._last_idx_xy = np.asarray(window.grid.shape[:2], dtype=float) - 1.0
        sz = float(window.grid.spacing_m[2])
        oz = float(window.grid.origin_enu[2])
        self._k_range = ((self.alt_range[0] - oz) / sz, (self.alt_range[1] - oz) / sz)

    # ---------- 基本判定 ----------

    def _band_ok(self, pts: np.ndarray) -> np.ndarray:
        """真高判定；没有地形信息时视为全通过（退化成纯绝对包络）。"""
        n = len(np.atleast_2d(pts))
        if self.terrain is None or self.agl is None:
            return np.ones(n, dtype=bool)
        return self.terrain.band_ok(pts, *self.agl)

    def _inside(self, pts: np.ndarray) -> bool:
        pts = np.atleast_2d(pts)
        return bool(self.window.inside_grid(pts).all() and self._band_ok(pts).all())

    def _segment_free(self, a: np.ndarray, b: np.ndarray) -> bool:
        """段内采样：窗口内 + SDF ≥ 净空。两端点以外不判（窗口边界不是墙）。"""
        length = float(np.linalg.norm(b - a))
        n = max(2, int(np.ceil(length / self.cfg.sample_step_m)) + 1)
        ts = np.linspace(0.0, 1.0, n)[:, None]
        pts = a + ts * (b - a)
        if not self._inside(pts):
            return False
        if float(np.linalg.norm(b - a)) < _EPS_M:
            return float(self.field.at(a[None, :])[0]) >= self.cfg.clearance_m
        return self.field.segment_min(a, b, self.cfg.sample_step_m) >= self.cfg.clearance_m

    def _sample(self, rng: np.random.Generator) -> np.ndarray:
        """在窗口内均匀采一点（索引空间均匀 → ENU，斜栅格下也不会采到带外）。"""
        idx_xy = rng.uniform(0.0, self._last_idx_xy)
        k = rng.uniform(*self._k_range)
        q = self.window.grid.index_to_enu(np.array([idx_xy[0], idx_xy[1], k]))
        if self.terrain is not None and self.agl is not None:
            q[2] = float(self.terrain.at(q[None, :])[0]) + rng.uniform(*self.agl)
        q[2] = float(np.clip(q[2], self.alt_range[0], self.alt_range[1]))
        return q

    def _steer(self, near: np.ndarray, q: np.ndarray) -> np.ndarray | None:
        delta = q - near
        dist = float(np.linalg.norm(delta))
        if dist < _EPS_M:
            return None
        step = float(np.clip(self.cfg.sdf_step_scale * self.field.at(near[None, :])[0],
                             self.cfg.min_step_m, self.cfg.step_m))
        step = min(step, dist)
        new = near + delta / dist * step
        if new[2] < self.alt_range[0] or new[2] > self.alt_range[1]:
            return None
        if not self._band_ok(new[None, :])[0]:
            return None
        return new

    # ---------- 主循环 ----------

    def plan(
        self,
        start: Sequence[float],
        goal: Sequence[float],
        *,
        seed: int | None = None,
    ) -> RRTResult | None:
        """跑一次 RRT；``seed=None`` 用配置 seed，多初值传 ``cfg.seed + i``。"""
        rng = np.random.default_rng(self.cfg.seed if seed is None else int(seed))
        start_p = np.asarray(start, dtype=float)
        goal_p = np.asarray(goal, dtype=float)
        if not self._segment_free(start_p, start_p) or not self._segment_free(goal_p, goal_p):
            return None
        if self._segment_free(start_p, goal_p):
            return RRTResult(
                path_enu=np.vstack([start_p, goal_p]),
                iterations=0, nodes=0, path_points_before_shortcut=2,
            )

        nodes = [start_p]
        parents = [-1]
        for it in range(1, self.cfg.max_iters + 1):
            if rng.random() < self.cfg.goal_sample_rate:
                q = goal_p
            else:
                q = self._sample(rng)
            arr = np.vstack(nodes)
            nearest = int(np.linalg.norm(arr - q, axis=1).argmin())
            new = self._steer(arr[nearest], q)
            if new is None or not self._segment_free(arr[nearest], new):
                continue
            nodes.append(new)
            parents.append(nearest)
            if self._segment_free(new, goal_p):
                nodes.append(goal_p)
                parents.append(len(nodes) - 2)
                path = self._reconstruct(nodes, parents)
                raw = len(path)
                path = self._shortcut(path)
                return RRTResult(
                    path_enu=path,
                    iterations=it,
                    nodes=len(nodes),
                    path_points_before_shortcut=raw,
                )
        return None

    def plan_many(
        self,
        start: Sequence[float],
        goal: Sequence[float],
        *,
        initializations: int,
    ) -> list[RRTResult]:
        """跑 ``initializations`` 次独立 RRT（seed+i），返回所有找到的候选。

        只负责生成候选，不判断交付净空、不选优；那是规划层的职责。
        """
        n = max(1, int(initializations))
        results: list[RRTResult] = []
        for i in range(n):
            result = self.plan(start, goal, seed=self.cfg.seed + i)
            if result is not None:
                results.append(result)
        return results

    def _reconstruct(self, nodes: list[np.ndarray], parents: list[int]) -> np.ndarray:
        chain: list[np.ndarray] = []
        idx = len(nodes) - 1
        while idx != -1:
            chain.append(nodes[idx])
            idx = parents[idx]
        return np.vstack(chain[::-1])

    def _shortcut(self, path: np.ndarray) -> np.ndarray:
        """贪心捷径：每段都过 ``_segment_free``，**省下的预算不换"没校验的直连"**。

        以前这里有个坑：预算（``shortcut_iters``）用尽时直接 ``append(path[-1])``，
        等于把"当前点 → 终点"的一条**未经校验的直线**接到路径尾部。预算为 0 时更极端——
        路径退化成 `[start, goal]` 一条直线，穿楼而过却仍是 ``status=ok``（实测净空 -11.70 m）。
        现在预算用尽就把剩下的原始节点原样接上：原始边都是 RRT 生长时校验过的自由边。

        ``shortcut_iters`` 是**软上限**：内层必须一直退到相邻点（``j == i+1``）才停，
        因为只有相邻边保证校验过；所以单段尝试次数会略超预算。这里宁可少省几次，
        也不接一条没校验的边。
        """
        if len(path) <= 2:
            return path
        kept = [path[0]]
        i = 0
        budget = self.cfg.shortcut_iters
        while i < len(path) - 1 and budget > 0:
            j = len(path) - 1
            while j > i + 1:
                budget -= 1
                if self._segment_free(path[i], path[j]):
                    break
                j -= 1
            kept.append(path[j])   # j == i+1 时是原始相邻边，已校验过
            i = j
        if i < len(path) - 1:
            kept.extend(path[i + 1:])
        return np.vstack(kept)

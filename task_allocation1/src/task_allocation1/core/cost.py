"""代价访问：节点键编码 + 注入式距离函数 + 代价矩阵。

节点键三套空间互不冲突（对齐参考项目 ``cost/nest_keys.py`` / ``provider.py``）：
    任务键 = task.id（非负整数）
    机巢键 = DEFAULT_NEST_OFFSET + nest.id
    UAV 起点键 = -(1 + uav.id)

:class:`CostProvider.cost` 是**有向**模型代价：命中 pairwise 用其值，否则用
distance_fn(pa, pb) 现算；绝不回退反向键、不静默换欧氏（口径唯一）。
:meth:`CostProvider.travel` 是物理航程（米），专供 ETA / 时间窗，与 cost 同源。
本模块零文件 I/O：距离口径要么 pairwise 给定，要么从注入的 distance_fn 现算。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..contract.defaults import DEFAULT_NEST_OFFSET
from ..contract.distance import DistanceFn, Position, euclidean3d
from ..contract.entities import UAV, Nest, Task

INF = float("inf")


def nest_key(nest_id: int, offset: int = DEFAULT_NEST_OFFSET) -> int:
    return offset + int(nest_id)


def uav_start_key(uav_id: int) -> int:
    return -(1 + int(uav_id))


@dataclass
class CostProvider:
    """有向代价访问器。pairwise 可为普通 dict，或任意惰性 Mapping（活视图直通）。"""

    pairwise: Mapping[tuple[int, int], float] = field(default_factory=dict)
    positions: dict[int, Position] = field(default_factory=dict)
    nest_offset: int = DEFAULT_NEST_OFFSET
    distance_fn: DistanceFn | None = None

    def key(self, nest_id: int) -> int:
        return nest_key(nest_id, self.nest_offset)

    def _distance(self, a: Position, b: Position) -> float:
        fn = self.distance_fn or euclidean3d
        try:
            return float(fn(a, b))
        except (ValueError, TypeError, IndexError, OverflowError):
            # 注入函数在特定坐标对上失败时退回欧氏，保持纯计算不中断
            return float(euclidean3d(a, b))

    def cost(self, id_a: int, id_b: int) -> float:
        """有向代价：pairwise 命中优先；否则按两端位置用 distance_fn 现算。"""
        pair = (int(id_a), int(id_b))
        if pair in self.pairwise:
            return float(self.pairwise[pair])
        pa = self.positions.get(pair[0])
        pb = self.positions.get(pair[1])
        if pa is None or pb is None:
            return INF
        return self._distance(pa, pb)

    def travel(self, a: Position, b: Position) -> float:
        """物理航程（米），与时间窗 ETA 同源，不读 pairwise。"""
        return self._distance(a, b)

    def cost_by_position(self, a: Position, b: Position) -> float:
        """按坐标直接取有向代价（与 cost 同口径），供起点快照等无键场景。"""
        return self._distance(a, b)


def build_cost_provider(
    tasks: Sequence[Task],
    nests: Sequence[Nest],
    uavs: Sequence[UAV],
    pairwise: Mapping[tuple[int, int], float] | None = None,
    nest_offset: int = DEFAULT_NEST_OFFSET,
    distance_fn: DistanceFn | None = None,
) -> CostProvider:
    """装配代价访问器：注册 task / nest / uav 起点三类位置，绑定距离口径。"""
    positions: dict[int, Position] = {}
    for t in tasks:
        positions[int(t.id)] = t.position
    for n in nests:
        positions[nest_key(n.id, nest_offset)] = n.position
    for u in uavs:
        positions[uav_start_key(u.id)] = u.position
    # 非 dict Mapping（惰性活视图）直通共享，禁止 dict() 物化（会 O(n²) 回潮）
    pw: Mapping[tuple[int, int], float] = pairwise if pairwise is not None else {}
    return CostProvider(
        pairwise=pw,
        positions=positions,
        nest_offset=nest_offset,
        distance_fn=distance_fn,
    )


@dataclass
class CostMatrixBuild:
    """方案 A 代价矩阵：列只有任务，终点机巢不进编码（同巢起降亦无需进）。"""

    provider: CostProvider
    uav_target: np.ndarray  # (K, N)
    task_task: np.ndarray  # (N, N)
    participating: list[UAV] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)

    def combined(self) -> np.ndarray:
        """(K+N, N)：上 K 行 UAV→任务，下 N 行 任务→任务。"""
        if self.uav_target.size == 0 and self.task_task.size == 0:
            return np.zeros((0, 0), dtype=float)
        return np.vstack([self.uav_target, self.task_task])


def build_cost_matrix(
    tasks: Sequence[Task],
    uavs: Sequence[UAV],
    nests: Sequence[Nest],
    distance_fn: DistanceFn | None = None,
    nest_offset: int = DEFAULT_NEST_OFFSET,
    uav_target: np.ndarray | None = None,
    pairwise: Mapping[tuple[int, int], float] | None = None,
) -> CostMatrixBuild:
    """构造 (K,N)/(N,N) 代价矩阵。

    ``uav_target`` 可由选机层预筛传入（形状 (K,N)）；缺省按全部 uavs×tasks 现算。
    """
    tl = list(tasks)
    ul = list(uavs)
    prov = build_cost_provider(tl, nests, ul, pairwise=pairwise,
                               nest_offset=nest_offset, distance_fn=distance_fn)
    n = len(tl)
    k = len(ul)
    if uav_target is None:
        ut = np.zeros((k, n), dtype=float)
        for i, u in enumerate(ul):
            for j, t in enumerate(tl):
                ut[i, j] = prov.cost(uav_start_key(u.id), int(t.id))
    else:
        ut = np.asarray(uav_target, dtype=float)
    tt = np.zeros((n, n), dtype=float)
    for i, ti in enumerate(tl):
        for j, tj in enumerate(tl):
            tt[i, j] = 0.0 if i == j else prov.cost(int(ti.id), int(tj.id))
    return CostMatrixBuild(provider=prov, uav_target=ut, task_task=tt,
                           participating=ul, tasks=tl)


def finite_max(values: Any) -> float:
    """有限值上界（供 big-M 推导）。"""
    finite = [abs(float(v)) for v in values if np.isfinite(v)]
    return max(finite, default=0.0)

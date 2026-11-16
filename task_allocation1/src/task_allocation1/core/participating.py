"""参与机选择：本周期派 K 架机进精细规划（与任务数解耦）。

三层裁剪：① K ≤ N-1（SRP 不变式）+ 运营上限；② 策略排序
（cheapest / first / marginal / demand）；③ demand 做设施选址式最大覆盖。
终点巢不参与选机——同巢起降下每机终点恒为起飞巢，无跨巢竞争。

``uav_target`` 形状 (n_avail, N)；``row_of_uav`` 把 UAV id 映射到行号。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ..contract.config import SolverConfig
from ..contract.entities import UAV


def select_participating_uavs(uavs: Sequence[UAV], n_tasks: int,
                              config: SolverConfig,
                              uav_target: np.ndarray | None = None,
                              row_of_uav: Mapping[int, int] | None = None) -> list[UAV]:
    if not uavs or n_tasks <= 0:
        return list(uavs)

    k_cap = max(1, n_tasks - 1)  # SRP 不变式
    if config.max_participating_uavs is not None:
        k_cap = min(k_cap, max(1, int(config.max_participating_uavs)))
    k = max(1, min(k_cap, len(uavs)))

    policy = config.participating_policy
    if policy == "first" or uav_target is None or row_of_uav is None:
        picked = list(uavs[:k])
    elif policy == "cheapest":
        picked = _by_cheapest(uavs, k, uav_target, row_of_uav)
    elif policy == "marginal":
        picked = _by_marginal(uavs, k, uav_target, row_of_uav)
    elif policy == "demand":
        picked = _by_demand(uavs, k, config, uav_target, row_of_uav)
    else:
        picked = _by_cheapest(uavs, k, uav_target, row_of_uav)

    # 保持原始顺序（下游 cost 矩阵行号稳定）
    order = {u.id: i for i, u in enumerate(uavs)}
    return sorted(picked, key=lambda u: order[u.id])


def _by_cheapest(uavs: Sequence[UAV], k: int, ut: np.ndarray,
                 row_of_uav: Mapping[int, int]) -> list[UAV]:
    def mincost(u: UAV) -> float:
        r = row_of_uav[u.id]
        row = ut[r]
        finite = row[np.isfinite(row)]
        return float(finite.min()) if finite.size else float("inf")
    return sorted(uavs, key=mincost)[:k]


def _by_marginal(uavs: Sequence[UAV], k: int, ut: np.ndarray,
                 row_of_uav: Mapping[int, int]) -> list[UAV]:
    """该机是几个任务的最近机（近距离份额）越多越优先。"""
    shares = {u.id: 0 for u in uavs}
    for col in range(ut.shape[1]):
        col_vals = ut[:, col]
        if not np.isfinite(col_vals).any():
            continue
        r = int(np.argmin(np.where(np.isfinite(col_vals), col_vals, np.inf)))
        for u in uavs:
            if row_of_uav[u.id] == r:
                shares[u.id] += 1
    return sorted(uavs, key=lambda u: -shares[u.id])[:k]


def _by_demand(uavs: Sequence[UAV], k: int, config: SolverConfig,
               ut: np.ndarray, row_of_uav: Mapping[int, int]) -> list[UAV]:
    """贪心最大覆盖：每轮选覆盖"尚未被覆盖任务"最多、代价最低的机。"""
    n_tasks = ut.shape[1]
    radius = {u.id: max(float(u.remaining_range) * config.demand_range_factor, 1e-6)
              for u in uavs}
    covered: set[int] = set()
    picked: list[UAV] = []
    while len(picked) < k:
        best: tuple[int, float, UAV] | None = None
        for u in uavs:
            if u in picked:
                continue
            r = row_of_uav[u.id]
            reach = [c for c in range(n_tasks)
                     if np.isfinite(ut[r, c]) and ut[r, c] <= radius[u.id]]
            new = len(set(reach) - covered)
            total = float(np.nansum(ut[r, reach])) if reach else float("inf")
            key = (-new, total)
            if best is None or key < best[:2]:
                best = (key[0], key[1], u)
        if best is None:
            break
        u = best[2]
        picked.append(u)
        r = row_of_uav[u.id]
        covered |= {c for c in range(n_tasks)
                    if np.isfinite(ut[r, c]) and ut[r, c] <= radius[u.id]}
    return picked

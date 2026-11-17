"""离散侧布局不变量：航次段规范化。

``transform.py`` 只负责连续→离散；这里守护 SRP 的唯一不变量：
- 起始基因按 UAV 下标升序占据前 K 槽，其后紧跟属于该 UAV 的巡游基因；
- 每个目标恰好出现一次（一个任务只被一架机完成），缺漏目标挂到最后一个非空 UAV。

``repair_invalid`` / ``mask_*`` / ``nearest_match_adaptive`` 曾服务于已停用的
balanced/overloaded 通路，随该通路一并删除（任务唯一分配后不再需要按模型分支掩码）。
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from .encoding import Gene


def recalc_tour_costs(genes: list[Gene], matrix: np.ndarray, n_uavs: int) -> None:
    """巡游边扰动后按新前驱重算 Gene.cost，保持与顺序一致。"""
    prev: int | None = None
    for i, g in enumerate(genes):
        if g.uav_id >= 0:
            prev = g.target_id
            continue
        if prev is not None:
            genes[i] = Gene(-1, g.target_id, float(matrix[n_uavs + prev, g.target_id]))
        prev = g.target_id


def normalize_srp_segments(genes: list[Gene], matrix: np.ndarray, n_uavs: int, n_targets: int) -> list[Gene]:
    """重建严格航次分段：每个 UAV 起始基因在前，其后紧跟属于它的巡游基因。

    方案 A 不生成终止基因；起始基因按 UAV 下标升序，基因总数恒为 N。
    """
    routes: dict[int, list[int]] = {u: [] for u in range(n_uavs)}
    current = -1
    for g in genes:
        if g.uav_id >= 0:
            current = g.uav_id
            if 0 <= current < n_uavs and g.target_id not in routes[current]:
                routes[current].append(g.target_id)
        elif 0 <= current < n_uavs and g.target_id not in routes[current]:
            routes[current].append(g.target_id)

    out: list[Gene] = []
    for u in range(n_uavs):
        tids = routes[u]
        if not tids:
            continue
        out.append(Gene(u, tids[0], float(matrix[u, tids[0]])))
        for prev, tid in pairwise(tids):
            out.append(Gene(-1, tid, float(matrix[n_uavs + prev, tid])))
    # 保证目标不重复：缺失目标挂到最后一个非空 UAV
    seen = {g.target_id for g in out}
    missing = [t for t in range(n_targets) if t not in seen]
    if missing and out:
        anchor = max((g for g in out if g.uav_id >= 0), key=lambda g: g.uav_id)
        prev = anchor.target_id
        for t in missing:
            out.append(Gene(-1, t, float(matrix[n_uavs + prev, t])))
            prev = t
    return out

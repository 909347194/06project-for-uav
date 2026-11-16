"""反映射修复规则：最近邻自适应匹配、唯一性掩码、无效值修补、段规范化。

``transform.py`` 只负责连续→离散；这里守护离散侧的布局不变量：
- balanced：UAV↔Target 一一对应；
- overloaded：UAV 不重复、目标至少出现一次；
- srp：起始基因必须按 UAV 下标升序占据前 K 槽，段内巡游基因紧随其后。
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from .encoding import Gene


def nearest_match_adaptive(
    cv: float, matrix: np.ndarray, mask: np.ndarray, *, temperature: float,
    rng: np.random.Generator,
) -> tuple[int, int, float] | None:
    """在未被掩码的格子里，找代价最接近 cv 的 (row, col)；温度控制 softmax 探索。"""
    diffs = np.abs(np.where(mask, np.inf, matrix) - cv)
    if not np.isfinite(diffs).any():
        return None
    if temperature < 0.1:
        flat = int(np.argmin(diffs))
    else:
        d = diffs.copy()
        d[~np.isfinite(d)] = np.inf
        minv = float(np.nanmin(np.where(np.isfinite(d), d, np.inf)))
        if not np.isfinite(minv):
            return None
        spread = max(float(np.nanstd(d[np.isfinite(d)])), 1e-6)
        logits = -(d - minv) / spread / max(0.01, temperature * 2.0)
        logits[~np.isfinite(logits)] = -1e9
        p = np.exp(logits - logits.max())
        p /= p.sum()
        flat = int(rng.choice(p.size, p=p.ravel()))
    row, col = divmod(flat, matrix.shape[1])
    return int(row), int(col), float(matrix[row, col])


def mask_balanced(mask: np.ndarray, row: int, col: int) -> None:
    mask[row, :] = True
    mask[:, col] = True


def mask_overloaded(mask: np.ndarray, row: int, col: int) -> None:
    mask[row, :] = True  # UAV 不重复；目标可重复


def repair_invalid(matrix: np.ndarray, mask: np.ndarray, count: int, *,
                   model_type, n_uavs: int, rng: np.random.Generator) -> list[tuple[int, int, float]]:
    """规则 3.6：无效基因（nan/负/越界）随机补到可用格。"""
    out: list[tuple[int, int, float]] = []
    rows = [r for r in range(matrix.shape[0]) if not mask[r, :].all()]
    if not rows:
        return out
    for _ in range(count):
        row = int(rng.choice(rows))
        cols = [c for c in range(matrix.shape[1]) if not mask[row, c] and np.isfinite(matrix[row, c])]
        if not cols:
            continue
        col = int(rng.choice(cols))
        out.append((row, col, float(matrix[row, col])))
        mask_balanced(mask, row, col) if model_type.value == "balanced" else mask_overloaded(mask, row, col)
    return out


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

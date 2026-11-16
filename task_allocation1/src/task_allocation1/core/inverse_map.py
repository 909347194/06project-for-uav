"""反映射 φ'（公式 3-7）：连续代价值向量 → 离散个体。

- balanced/overloaded：只在「UAV 行 × 任务列」子矩阵匹配，掩码保唯一性；
- srp：起始槽位逐 UAV 匹配，巡游槽位按“当前已访问目标行”迭代匹配
  （Ming 2017 Rule 2(c)），令**访问顺序真正进入 DE 搜索空间**。

温度 ``T = 1 - g/S``：早期大扰动探索、后期贪心开发。终点机巢不进编码。
"""

from __future__ import annotations

import numpy as np

from ..contract.enums import ModelType
from .encoding import Gene, Individual
from .repair_rules import (
    mask_balanced,
    mask_overloaded,
    nearest_match_adaptive,
    normalize_srp_segments,
    recalc_tour_costs,
    repair_invalid,
)


def inverse_phi(cost_vector: np.ndarray, cost_matrix: np.ndarray, k: int, n: int,
                model: ModelType, rng: np.random.Generator, *, temperature: float = 0.5,
                assign_factor: float = 0.4, tour_factor: float = 0.5,
                temp_gate: float = 0.1) -> Individual:
    if model == ModelType.SRP:
        return _srp(cost_vector, cost_matrix, k, n, rng, temperature, tour_factor, temp_gate)
    return _standard(cost_vector, cost_matrix, k, n, model, rng,
                     temperature, assign_factor, temp_gate)


def _standard(cv: np.ndarray, cm: np.ndarray, k: int, n: int, model: ModelType,
              rng: np.random.Generator, temperature: float, factor: float,
              gate: float) -> Individual:
    work = np.asarray(cm, dtype=float).copy()
    mask = np.zeros_like(work, dtype=bool)
    mask[k:, :] = True   # 只用 UAV 行
    mask[:, n:] = True   # 只用任务列
    genes: list[Gene] = []
    invalid = 0
    for idx, value in enumerate(cv):
        if not np.isfinite(value) or value < 0:
            invalid += 1
            continue
        hit = nearest_match_adaptive(float(value), work, mask,
                                     temperature=temperature, rng=rng)
        if hit is None:
            invalid += 1
            continue
        row, col, cost = hit
        genes.append(Gene(row, col, cost))
        if model == ModelType.BALANCED:
            mask_balanced(mask, row, col)
        else:
            mask_overloaded(mask, row, col)

    if invalid:
        genes.extend(Gene(r, c, co) for r, c, co in
                     repair_invalid(work, mask, invalid, model_type=model,
                                    n_uavs=k, rng=rng))

    if temperature > gate and len(genes) >= 2:
        prob = temperature * factor
        if rng.random() < prob:
            i, j = rng.choice(len(genes), size=2, replace=False)
            gi, gj = genes[i], genes[j]
            genes[i] = Gene(gi.uav_id, gj.target_id, float(work[gi.uav_id, gj.target_id]))
            genes[j] = Gene(gj.uav_id, gi.target_id, float(work[gj.uav_id, gi.target_id]))

    if model == ModelType.BALANCED:
        _fix_balanced(genes, work, n)
    elif model == ModelType.OVERLOADED:
        _fix_overloaded(genes, work, n, rng)
    return Individual(genes=genes, model_type=model)


def _fix_balanced(genes: list[Gene], cm: np.ndarray, n: int) -> None:
    seen: set[int] = set()
    dups: list[int] = []
    for i, g in enumerate(genes):
        if g.target_id in seen:
            dups.append(i)
        else:
            seen.add(g.target_id)
    missing = [t for t in range(n) if t not in seen]
    for i in dups:
        if not missing:
            break
        g = genes[i]; t = missing.pop(0)
        genes[i] = Gene(g.uav_id, t, float(cm[g.uav_id, t]))


def _fix_overloaded(genes: list[Gene], cm: np.ndarray, n: int,
                    rng: np.random.Generator) -> None:
    counts: dict[int, int] = {}
    for g in genes:
        counts[g.target_id] = counts.get(g.target_id, 0) + 1
    uncovered = [t for t in range(n) if counts.get(t, 0) == 0]
    replaceable = [t for t, c in counts.items() if c > 1]
    for t in uncovered:
        if not replaceable:
            break
        src = int(rng.choice(replaceable))
        idxs = [i for i, g in enumerate(genes) if g.target_id == src]
        worst = max(idxs, key=lambda i: genes[i].cost)
        g = genes[worst]
        genes[worst] = Gene(g.uav_id, t, float(cm[g.uav_id, t]))
        counts[src] -= 1; counts[t] = counts.get(t, 0) + 1
        if counts[src] <= 1:
            replaceable.remove(src)


def _srp(cv: np.ndarray, cm: np.ndarray, k: int, n: int, rng: np.random.Generator,
         temperature: float, factor: float, gate: float) -> Individual:
    work = np.asarray(cm, dtype=float).copy()
    assigned: set[int] = set()
    last: dict[int, int] = {}
    genes: list[Gene] = []

    for i in range(min(k, len(cv))):
        value = float(cv[i])
        avail = [t for t in range(n) if t not in assigned]
        if not avail:
            break
        reach = [t for t in avail if np.isfinite(work[i, t])]
        pool = reach or avail
        if np.isfinite(value) and value >= 0 and pool:
            diffs = np.abs(np.array([work[i, t] for t in pool]) - value)
            if temperature < 0.1:
                t = pool[int(np.argmin(diffs))]
            else:
                spread = max(float(np.std(diffs)), 1e-6)
                logits = -diffs / spread / max(0.01, temperature * 2.0)
                probs = np.exp(logits - logits.max()); probs /= probs.sum()
                t = pool[int(rng.choice(len(pool), p=probs))]
        else:
            t = int(rng.choice(pool))
        genes.append(Gene(i, t, float(work[i, t])))
        assigned.add(t); last[i] = t

    remaining = [t for t in range(n) if t not in assigned]
    for j in range(len(remaining)):
        idx = k + j
        value = float(cv[idx]) if idx < len(cv) else float("nan")
        u, t, cost = _match_tour(value, work, k, last, remaining, temperature, rng)
        if u < 0:
            break
        genes.append(Gene(-1, t, cost))
        last[u] = t
        remaining.remove(t)

    if temperature > gate and sum(1 for g in genes if g.uav_id < 0) >= 2:
        _perturb_tour(genes, work, k, temperature, factor, rng)

    genes = normalize_srp_segments(genes, work, k, n)
    return Individual(genes=genes, model_type=ModelType.SRP)


def _match_tour(value: float, cm: np.ndarray, k: int, last: dict[int, int],
                candidates: list[int], temperature: float,
                rng: np.random.Generator) -> tuple[int, int, float]:
    rows: list[int] = []
    uavs: list[int] = []
    for u in sorted(last):
        r = k + last[u]
        if 0 <= r < cm.shape[0]:
            rows.append(r); uavs.append(u)
    cols = [c for c in candidates if 0 <= c < cm.shape[1]]
    if not rows or not cols:
        return -1, -1, float("inf")
    costs = cm[np.ix_(rows, cols)]
    costs = np.where(np.isfinite(costs), costs, np.inf)
    diffs = np.abs(costs - value) if np.isfinite(value) and value >= 0 else costs.copy()
    if temperature < 0.1 or not np.isfinite(diffs).all():
        flat = int(np.argmin(diffs))
    else:
        spread = max(float(np.std(diffs)), 1e-6)
        logits = (-diffs / spread / max(0.01, temperature * 2.0)).ravel()
        logits -= logits.max()
        p = np.exp(logits); p /= p.sum()
        flat = int(rng.choice(p.size, p=p))
    r, c = divmod(flat, len(cols))
    return uavs[r], cols[c], float(costs[r, c])


def _perturb_tour(genes: list[Gene], cm: np.ndarray, k: int, temperature: float,
                  factor: float, rng: np.random.Generator) -> None:
    idxs = [i for i, g in enumerate(genes) if g.uav_id < 0]
    if rng.random() >= temperature * factor:
        return
    op = str(rng.choice(["swap", "insert", "reverse"]))
    if op == "swap" and len(idxs) >= 2:
        i, j = rng.choice(idxs, size=2, replace=False)
        genes[i], genes[j] = Gene(-1, genes[j].target_id, genes[i].cost), \
                             Gene(-1, genes[i].target_id, genes[j].cost)
    elif op == "insert" and len(idxs) >= 2:
        s, d = int(rng.choice(idxs)), int(rng.choice(idxs))
        if s != d:
            g = genes.pop(s)
            genes.insert(d if d < s else d - 1, g)
    elif op == "reverse" and len(idxs) >= 3:
        i, j = sorted(int(x) for x in rng.choice(idxs, size=2, replace=False))
        genes[i:j + 1] = list(reversed(genes[i:j + 1]))
    recalc_tour_costs(genes, cm, k)

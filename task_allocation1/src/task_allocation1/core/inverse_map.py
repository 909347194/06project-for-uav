"""反映射 φ'（公式 3-7）：连续代价值向量 → 离散个体。

只有 srp 一条通路：起始槽位逐 UAV 匹配，巡游槽位按“当前已访问目标行”迭代匹配
（Ming 2017 Rule 2(c)），令**访问顺序真正进入 DE 搜索空间**。任务唯一性由
``remaining`` 未分配池保证，``normalize_srp_segments`` 只兜补缺漏。

温度 ``T = 1 - g/S``：早期大扰动探索、后期贪心开发。终点机巢不进编码。
"""

from __future__ import annotations

import numpy as np

from ..contract.enums import ModelType
from .encoding import Gene, Individual
from .repair_rules import (
    normalize_srp_segments,
    recalc_tour_costs,
)


def inverse_phi(cost_vector: np.ndarray, cost_matrix: np.ndarray, k: int, n: int,
                model: ModelType, rng: np.random.Generator, *, temperature: float = 0.5,
                tour_factor: float = 0.5, temp_gate: float = 0.1) -> Individual:
    """``tour_factor`` 控制巡游顺序扰动幅度，``temp_gate`` 是扰动温度门限。

    原先的 ``assign_factor``（分配槽交换扰动）随 BALANCED/OVERLOADED 通路一并删除：
    SRP 的起始槽位只由温度控制匹配，没有“改派”这一层。
    """
    if model is not ModelType.SRP:
        raise ValueError(
            f"反映射只支持 SRP：收到 model={model!r}。BALANCED(K=N) 与 OVERLOADED(K>N) "
            f"已停用（任务唯一分配、一机可串飞多任务）。"
        )
    return _srp(cost_vector, cost_matrix, k, n, rng, temperature, tour_factor, temp_gate)


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

"""DMDE 进化算子：动态交叉率 CR（3-9）、混合差分策略（3-10）、
动态缩放因子 F（3-11）、GMR 灭绝重启（3-12）。

差分都在**飞行代价空间**进行；同一代所有个体共用 CR，F 逐个体独立采样。
随机性只取注入的 Generator；无全局随机状态。
"""

from __future__ import annotations

import math

import numpy as np


def dynamic_crossover_rate(gen: int, total_gens: int, zeta: int = 3) -> float:
    """公式 3-9：CR(g) = 1 - (log g / log S)^ζ，g 从 1 起。"""
    if total_gens <= 1 or gen <= 1:
        return 1.0
    g = min(max(gen, 1), total_gens)
    return float(1.0 - (math.log(g) / math.log(total_gens)) ** zeta)


def dynamic_scale_factor(cr: float, rng: np.random.Generator) -> float:
    """公式 3-11：rand<=CR → F=CR；否则 F=(1-CR)/2。"""
    return float(cr) if rng.random() <= cr else float((1.0 - cr) / 2.0)


def hybrid_differential_population(vectors: np.ndarray, best_idx: int, cr: float,
                                   rng: np.random.Generator) -> np.ndarray:
    """公式 3-10：CR 高（探索）用 DE/rand/1，CR 低（开发）用 DE/best/2。

    ``DE/rand/1``: v = x[r1] + F·(x[r2] - x[r3])
    ``DE/best/2``: v = x[best] + F·(x[r1]-x[r2]) + F·(x[r3]-x[r4])
    越界分量裁剪回原向量值（保持代价非负语义）。
    """
    pop = len(vectors)
    if pop < 4:
        return vectors.copy()
    out = np.empty_like(vectors)
    for i in range(pop):
        f = dynamic_scale_factor(cr, rng)
        idxs = [j for j in range(pop) if j != i]
        if cr >= 0.5:  # 探索
            r1, r2, r3 = rng.choice(idxs, size=3, replace=False)
            trial = vectors[r1] + f * (vectors[r2] - vectors[r3])
        else:          # 开发
            rs = rng.choice(idxs, size=4, replace=False)
            trial = (vectors[best_idx]
                     + f * (vectors[rs[0]] - vectors[rs[1]])
                     + f * (vectors[rs[2]] - vectors[rs[3]]))
        valid = np.isfinite(trial) & (trial >= 0)
        out[i] = np.where(valid, trial, vectors[i])
    return out


def should_extinct(cr: float, gen: int, total_gens: int, delta: float | None,
                   rng: np.random.Generator) -> bool:
    """GMR（3-12）：原式概率 = 1 - CR；给定 delta 则仅搜索后期触发。"""
    if delta is None:
        return bool(rng.random() < (1.0 - cr))
    return bool(gen >= delta * total_gens and rng.random() < (1.0 - cr))


def apply_extinction(population: list, keep: int, encoder_generate,
                     rng: np.random.Generator) -> None:
    """灭绝：保留最优 keep 个，其余用编码器重播种（原地替换）。"""
    if len(population) <= keep:
        return
    fresh = encoder_generate(len(population) - keep, rng)
    population[keep:] = fresh

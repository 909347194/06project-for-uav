"""DMDE 主循环（算法 3.1）：初始化 → 变异 → 反映射 → 评价 → 贪婪选择 → 灭绝。

差分在代价空间进行，反映射后每个个体经同一套解码修复链评价。
随机性单一通道：一个注入的 Generator，同种子逐位可复现。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..contract.enums import ModelType
from .encoding import Individual, PopulationEncoder, model_type_of
from .inverse_map import inverse_phi
from .operators import (
    dynamic_crossover_rate,
    hybrid_differential_population,
    should_extinct,
)


@dataclass
class DMDEResult:
    best: Individual | None = None
    cost_history: list[float] = field(default_factory=list)
    n_evals: int = 0


class DMDESolver:
    def __init__(self, *, pop_size: int, max_generations: int, zeta: int,
                 extinction_delta: float | None, seed: int | None,
                 tour_factor: float, temp_gate: float) -> None:
        self.pop_size = max(4, int(pop_size))
        self.max_generations = max(0, int(max_generations))
        self.zeta = int(zeta)
        self.delta = extinction_delta
        self.seed = seed
        self.tour_factor = float(tour_factor)
        self.temp_gate = float(temp_gate)

    def solve(self, cost_matrix: np.ndarray, k: int, n: int,
              evaluate, *, model: ModelType | None = None) -> DMDEResult:
        rng = np.random.default_rng(self.seed)
        model = model or model_type_of(k, n)
        result = DMDEResult()
        if n == 0 or cost_matrix.size == 0:
            return result

        encoder = PopulationEncoder(cost_matrix, k, n)
        encoder.model = model
        population = encoder.generate(self.pop_size, rng)

        def eval_one(ind: Individual) -> None:
            ind.fitness = float(evaluate(ind))
            result.n_evals += 1

        for ind in population:
            eval_one(ind)
        best = min(population, key=lambda x: x.fitness).copy()
        result.cost_history.append(best.fitness)

        for gen in range(1, self.max_generations + 1):
            cr = dynamic_crossover_rate(gen, self.max_generations, self.zeta)
            best_idx = int(np.argmin([ind.fitness for ind in population]))
            vectors = np.vstack([ind.cost_vector for ind in population])
            trials = hybrid_differential_population(vectors, best_idx, cr, rng)
            temperature = 1.0 - gen / self.max_generations

            for i, ind in enumerate(population):
                child = inverse_phi(
                    trials[i], cost_matrix, k, n, model, rng,
                    temperature=temperature,
                    tour_factor=self.tour_factor,
                    temp_gate=self.temp_gate,
                )
                eval_one(child)
                if child.fitness < ind.fitness:
                    child.model_type = model
                    population[i] = child
                    if child.fitness < best.fitness:
                        best = child.copy()

            if should_extinct(cr, gen, self.max_generations, self.delta, rng):
                sorted_pop = sorted(population, key=lambda x: x.fitness)
                keep = max(1, self.pop_size // 4)
                fresh = encoder.generate(self.pop_size - keep, rng)
                population[:] = sorted_pop[:keep] + fresh

            result.cost_history.append(best.fitness)

        result.best = best
        return result

"""编码：三元组基因 / 个体 / 种群初始化（论文规则 3.1~3.3，公式 3-3/3-4）。

基因 ``(uav_local, task_local, cost)``；``uav_local=-1`` 表示巡游边（任务→任务）。
``Individual.cost_vector`` 采用固定布局供 DE 差分：
    ``[0,K)``  起始基因（UAV→首任务），``[K,N)`` 巡游基因（任务→任务）。
方案 A：终点机巢不进编码——同巢起降下终点恒=起飞巢，更无需入基因。
基因长度恒为 N：K 个起始基因 + N-K 个巡游基因（N=1 时巡游维为 0）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contract.enums import ModelType


@dataclass(frozen=True)
class Gene:
    uav_id: int       # 局部 UAV 下标；-1 = 巡游边
    target_id: int    # 局部任务下标
    cost: float

    def __repr__(self) -> str:
        return f"Gene(U={self.uav_id}, T={self.target_id}, C={self.cost:.1f})"


@dataclass
class Individual:
    genes: list[Gene]
    fitness: float = float("inf")
    model_type: ModelType = ModelType.SRP

    @property
    def assignment(self) -> list[tuple[int, int]]:
        """(uav_local, task_local) 列表；巡游基因归属最近的前一个起始基因 UAV。"""
        out: list[tuple[int, int]] = []
        current = -1
        for g in self.genes:
            if g.uav_id >= 0:
                current = g.uav_id
                out.append((g.uav_id, g.target_id))
            else:
                out.append((current, g.target_id))
        return out

    @property
    def cost_vector(self) -> np.ndarray:
        starts = [g.cost for g in self.genes if g.uav_id >= 0]
        patrols = [g.cost for g in self.genes if g.uav_id < 0]
        return np.asarray(starts + patrols, dtype=float)

    def copy(self) -> Individual:
        return Individual(genes=list(self.genes), fitness=self.fitness,
                          model_type=self.model_type)


def model_type_of(k: int, n: int) -> ModelType:
    """按本库口径判定编码模型：只会返回 SRP 或 EMPTY。

    口径（2026-11 起固定）：任务唯一分配——一个任务只能被一架无人机完成；
    参与机数 ``K`` 恒满足 ``K <= max(1, N-1)``（``N=1`` 时 ``K=1``，巡游维为 0）；
    无人机多于任务时，未选中的机闲置。

    因此 BALANCED（``K == N``）与 OVERLOADED（``K > N``）在本库不再产生。这里不是
    “恰好不发生”，而是显式拒绝：越界即抛错，避免上游配置漂移后静默走进
    “一任务多机”的旧分支。
    """
    k = int(k)
    n = int(n)
    if n <= 0 or k <= 0:
        return ModelType.EMPTY
    k_cap = max(1, n - 1)
    if k > k_cap:
        raise ValueError(
            f"编码模型口径冲突：K={k}, N={n}, 上界={k_cap}。本库禁止 BALANCED(K=N) 与 "
            f"OVERLOADED(K>N)：任务唯一分配，参与机数必须由 "
            f"select_participating_uavs 裁剪到 max(1, N-1) 以内。"
        )
    return ModelType.SRP


class PopulationEncoder:
    """按代价矩阵生成初始种群；随机性只取注入的 ``np.random.Generator``。"""

    def __init__(self, cost_matrix: np.ndarray, n_uavs: int, n_targets: int) -> None:
        self.cm = cost_matrix
        self.k = n_uavs
        self.n = n_targets
        self.model = model_type_of(k=n_uavs, n=n_targets)

    def generate(self, pop_size: int, rng: np.random.Generator) -> list[Individual]:
        if self.model is ModelType.EMPTY:
            return []
        return [self._srp(rng) for _ in range(pop_size)]

    def _srp(self, rng: np.random.Generator) -> Individual:
        targets = list(rng.permutation(self.n))
        groups: dict[int, list[int]] = {i: [int(targets[i])] for i in range(self.k)}
        for i in range(self.k, self.n):
            groups[int(rng.integers(0, self.k))].append(int(targets[i]))
        genes: list[Gene] = []
        for u, tids in groups.items():
            ordered = self._greedy_order(u, tids)
            for seq, tid in enumerate(ordered):
                if seq == 0:
                    genes.append(Gene(u, tid, float(self.cm[u, tid])))
                else:
                    prev = ordered[seq - 1]
                    genes.append(Gene(-1, tid, float(self.cm[self.k + prev, tid])))
        return Individual(genes, model_type=self.model)

    def _greedy_order(self, u: int, tids: list[int]) -> list[int]:
        if len(tids) <= 1:
            return tids
        rem = set(tids)
        first = min(rem, key=lambda t: self.cm[u, t])
        out = [first]; rem.remove(first)
        while rem:
            last = out[-1]
            nxt = min(rem, key=lambda t: self.cm[self.k + last, t])
            out.append(nxt); rem.remove(nxt)
        return out

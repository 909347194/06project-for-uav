"""覆盖门禁：判定"能装却没装"（voluntary_skip）与能力不足（capacity_shortfall）。

语义：合法解 = S 内不存在「未交付且可被某机开班/插入」的任务。
一阶检验：若存在空闲机可单任务开班，或某现有航次插入后仍满足航程预算 →
voluntary_skip（非法）；否则 capacity_shortfall（物理装不下，允许不交付）。
终点巢固定为起飞巢，插入/开班代价用同巢返航口径核算。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..contract.solution import AllocationSolution
from .problem import SelectiveProblem
from .repair import DecodeRepair
from .tour import tour_path_cost


@dataclass
class CoverageStats:
    n_skip: int = 0
    n_shortfall: int = 0
    skip_ids: list[int] = field(default_factory=list)
    shortfall_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_skip": self.n_skip,
            "n_shortfall": self.n_shortfall,
            "skip_ids": list(self.skip_ids),
            "shortfall_ids": list(self.shortfall_ids),
        }


def audit_coverage(problem: SelectiveProblem, decoder: DecodeRepair,
                   solution: AllocationSolution) -> CoverageStats:
    stats = CoverageStats()
    executed = set(solution.executed_task_ids)
    cand = set(solution.candidate_task_ids) or set(problem.task_by_id)
    routes = {t.uav_id: list(t.task_ids) for t in solution.tours}

    for tid in sorted(cand - executed):
        if tid not in problem.task_by_id:
            continue
        can = False
        # 插入现有航次
        for uid, tids in routes.items():
            if not tids:
                continue
            u = problem.uav_by_id.get(uid)
            if u is None:
                continue
            for pos in range(len(tids) + 1):
                cand_tids = list(tids); cand_tids.insert(pos, tid)
                c = tour_path_cost(problem.costs, uid, cand_tids, u.nest_id, u.position)
                if c <= float(u.remaining_range) + 1e-6:
                    can = True
                    break
            if can:
                break
        # 空闲机开班
        if not can:
            for t in solution.tours:
                u = problem.uav_by_id.get(t.uav_id)
                if u is not None and not t.task_ids:
                    c = tour_path_cost(problem.costs, u.id, [tid], u.nest_id, u.position)
                    if c <= float(u.remaining_range) + 1e-6:
                        can = True
                        break
        if can:
            stats.n_skip += 1
            stats.skip_ids.append(tid)
        else:
            stats.n_shortfall += 1
            stats.shortfall_ids.append(tid)
    return stats

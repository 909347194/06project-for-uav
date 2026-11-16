"""问题实例：实体索引 + 航次代价 + 可行性判定 + 四层适应度（覆盖门禁口径）。

适应度（DMDE 最小化，门禁为唯一主口径）：
    f = W_hard·n_hard + w_exc·excess_m + W_skip·n_skip + w_late·L_tot + λ·path
固定表述：**搜索用 f 不含 −m·F1；F1 只作事后指标**（完成是约束、成本是目标）。

同巢起降硬约束（本库新增，恒检查）：任何非空航次终点机巢必须等于起飞巢；
违约记入 T0 硬违约，且 ``home_return_ok=False`` 直接使解不可行。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..contract.config import ProblemConfig
from ..contract.entities import UAV, Nest, Task
from ..contract.solution import AllocationSolution, UAVTour
from .cost import CostProvider, finite_max
from .tour import arrival_times, lateness_of, tour_path_cost

HARD_HOME_TAG = "home_return_violation"


@dataclass
class Feasibility:
    feasible: bool = True
    n_hard: int = 0
    range_excess: float = 0.0
    soft_violation: float = 0.0
    penalty: float = 0.0
    violations: list[str] = field(default_factory=list)
    lateness: dict[int, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "feasible": self.feasible,
            "n_hard": self.n_hard,
            "range_excess": self.range_excess,
            "soft_violation": self.soft_violation,
            "penalty": self.penalty,
            "violations": list(self.violations),
        }


@dataclass
class SelectiveProblem:
    """单周期选择性 MUAS 问题实例（同巢口径）。"""

    tasks: list[Task]
    uavs: list[UAV]
    nests: list[Nest]
    costs: CostProvider
    config: ProblemConfig = field(default_factory=ProblemConfig)
    fleet_uavs: list[UAV] = field(default_factory=list)
    task_by_id: dict[int, Task] = field(init=False, repr=False)
    uav_by_id: dict[int, UAV] = field(init=False, repr=False)
    nest_by_id: dict[int, Nest] = field(init=False, repr=False)
    _m_cache: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.task_by_id = {t.id: t for t in self.tasks}
        self.uav_by_id = {u.id: u for u in self.uavs}
        self.nest_by_id = {n.id: n for n in self.nests}
        if not self.fleet_uavs:
            self.fleet_uavs = list(self.uavs)

    # ---- 基础量 ---------------------------------------------------------
    def tour_cost(self, tour: UAVTour) -> float:
        u = self.uav_by_id.get(tour.uav_id)
        if u is None:
            return 0.0
        return tour_path_cost(self.costs, tour.uav_id, tour.task_ids,
                              tour.end_nest_id, tour.start_position)

    def path_cost(self, solution: AllocationSolution) -> float:
        return float(sum(self.tour_cost(t) for t in solution.tours))

    def f1_reward(self, solution: AllocationSolution) -> float:
        return float(sum(max(self.task_by_id[t].reward, 0.0)
                         for t in solution.executed_task_ids if t in self.task_by_id))

    def empty_solution(self) -> AllocationSolution:
        return AllocationSolution(
            tours=[UAVTour.from_uav(u) for u in self.uavs],
            candidate_task_ids=[t.id for t in self.tasks],
        )

    # ---- big-M / 权重推导 -----------------------------------------------
    def f1_weight(self) -> float:
        if self.config.big_m is not None:
            return max(1.0, float(self.config.big_m))
        if self._m_cache is None:
            pos = [float(t.reward) for t in self.tasks if float(t.reward) > 0]
            r_min = min(pos) if pos else 1.0
            edge = finite_max(list(self.costs.pairwise.values())) if self.costs.pairwise else 0.0
            pts = [p.position for p in self.tasks] + [p.position for p in self.uavs]
            pts += [p.position for p in self.nests]
            if pts:
                diag = math.sqrt(
                    (max(p[0] for p in pts) - min(p[0] for p in pts)) ** 2
                    + (max(p[1] for p in pts) - min(p[1] for p in pts)) ** 2
                    + (max(p[2] for p in pts) - min(p[2] for p in pts)) ** 2
                )
                edge = max(edge, diag, 1.0)
            legs = len(self.tasks) + 2 * len(self.uavs) + 2
            speed_min = min((max(float(u.speed), 1e-6) for u in self.uavs), default=1.0)
            c_bound = edge * legs
            v_bound = len(self.tasks) * (c_bound / speed_min)
            secondary = self.config.lambda_cost * c_bound
            secondary += (float(self.config.w_violation) if self.config.w_violation is not None
                          else self.config.lambda_cost * self.config.w_penalty) * v_bound
            self._m_cache = 1.0 + secondary / r_min
        return float(self._m_cache)

    def hard_weight(self) -> float:
        if self.config.w_hard_violation is not None:
            return max(0.0, float(self.config.w_hard_violation))
        r_pool = sum(max(float(t.reward), 0.0) for t in self.tasks)
        return float(self.config.hard_violation_kappa) * self.f1_weight() * r_pool

    def skip_weight(self) -> float:
        if self.config.w_skip_violation is not None:
            return max(0.0, float(self.config.w_skip_violation))
        return self.hard_weight()

    def excess_weight(self) -> float:
        if self.config.w_range_excess is not None:
            return max(0.0, float(self.config.w_range_excess))
        return 1.0 / max(self.f1_weight(), 1.0)

    # ---- 可行性 ---------------------------------------------------------
    def check_feasible(self, solution: AllocationSolution) -> Feasibility:
        cons = self.config.constraints
        rep = Feasibility()
        seen: set[int] = set()
        for tour in solution.tours:
            for tid in tour.task_ids:
                if tid in seen and cons.duplicate_task:
                    rep.violations.append("duplicate_task")
                seen.add(tid)
        if cons.duplicate_task:
            if set(solution.executed_task_ids) != seen:
                rep.violations.append("sum_x_neq_y")
            for tid in solution.executed_task_ids:
                if tid not in self.task_by_id:
                    rep.violations.append(f"unknown_task_{tid}")

        for tour in solution.tours:
            u = self.uav_by_id.get(tour.uav_id)
            if u is None:
                rep.violations.append(f"unknown_uav_{tour.uav_id}")
                continue
            # 同巢起降硬约束：终点 == 起飞巢（空航次无返航义务，豁免）
            if (not tour.is_idle
                    and (tour.end_nest_id is None
                         or tour.end_nest_id != tour.start_nest_id
                         or tour.end_nest_id != u.nest_id)):
                rep.violations.append(f"{HARD_HOME_TAG}_{u.id}")
            pc = self.tour_cost(tour)
            tour_ok = True
            if cons.range and tour.task_ids:
                rv = max(0.0, pc - float(u.remaining_range))
                if rv > 0:
                    rep.range_excess += rv
                    rep.violations.append(f"range_uav_{u.id}")
                    tour_ok = False
            if cons.endurance and tour.task_ids:
                limit = self._uav_max_time(u)
                if limit is not None:
                    from .tour import tour_endurance
                    ttime = tour_endurance(u, tour, self.task_by_id, provider=self.costs,
                                           current_time=float(self.config.current_time),
                                           service_time=float(self.config.default_service_time))
                    if ttime > limit + 1e-6:
                        rep.violations.append(f"endurance_uav_{u.id}")
                        tour_ok = False
            if cons.nest_availability and tour.end_nest_id is not None:
                nest = self.nest_by_id.get(tour.end_nest_id)
                if nest is None or not nest.available:
                    rep.violations.append(f"nest_unavailable_{tour.end_nest_id}")
                    tour_ok = False
            if cons.end_nest_reachability and tour.task_ids and tour.end_nest_id is not None:
                from .tour import return_leg_cost
                if not math.isfinite(return_leg_cost(self.costs, tour.task_ids[-1], tour.end_nest_id)):
                    rep.violations.append(f"end_nest_unreachable_{tour.end_nest_id}")
                    tour_ok = False
            if cons.time_window and tour.task_ids:
                seq = [self.task_by_id[tid] for tid in tour.task_ids if tid in self.task_by_id]
                arrivals = arrival_times(tour.start_position, seq, speed=float(u.speed),
                                         current_time=float(self.config.current_time),
                                         service_time=float(self.config.default_service_time),
                                         travel=self.costs.travel)
                _, _, per = lateness_of(arrivals, seq)
                for tid, late in per.items():
                    rep.lateness[tid] = late
                    if late > 0:
                        rep.soft_violation += late
            tour.feasible = tour_ok

        if cons.nest_capacity and self.nests:
            occ: dict[int, int] = {}
            for u in self.fleet_uavs:
                if u.nest_id is not None:
                    occ[u.nest_id] = occ.get(u.nest_id, 0) + 1
            for n in self.nests:
                if occ.get(n.id, 0) > int(n.capacity):
                    rep.violations.append(f"nest_overcapacity_{n.id}")

        rep.n_hard = len(rep.violations)
        wv = self.config.w_violation
        rep.penalty = (self.hard_weight() * rep.n_hard
                       + float(wv) * rep.soft_violation if wv is not None
                       else self.hard_weight() * rep.n_hard)
        rep.feasible = rep.n_hard == 0 and solution.home_return_ok()
        return rep

    def _uav_max_time(self, u: UAV) -> float | None:
        if u.max_time is not None:
            return float(u.max_time)
        rng = u.full_charge_range(1e9)
        return rng / max(float(u.speed), 1e-6) if rng < 1e18 else None

    # ---- 适应度 ---------------------------------------------------------
    def scalar_fitness(self, solution: AllocationSolution, rep: Feasibility | None = None) -> float:
        rep = rep or self.check_feasible(solution)
        n_skip = int((solution.meta.get("gate_stats") or {}).get("n_skip", 0))
        wv = self.config.w_violation
        late_pen = 0.0 if wv is None else float(wv) * rep.soft_violation
        return float(self.hard_weight() * rep.n_hard
                     + self.excess_weight() * rep.range_excess
                     + self.skip_weight() * n_skip
                     + late_pen
                     + self.config.lambda_cost * self.path_cost(solution))

    def evaluate(self, solution: AllocationSolution) -> AllocationSolution:
        rep = self.check_feasible(solution)
        solution.meta["feasibility"] = rep.as_dict()
        solution.meta["feasibility_obj"] = rep
        solution.feasible = rep.feasible
        solution.f1_reward = self.f1_reward(solution)
        solution.f2_cost = self.path_cost(solution)
        solution.meta["path_m"] = solution.f2_cost
        from .tour import return_leg_cost
        solution.meta["return_leg_m"] = sum(
            return_leg_cost(self.costs, t.task_ids[-1], t.end_nest_id)
            for t in solution.tours if t.task_ids
        )
        for tour in solution.tours:
            tour.est_cost = self.tour_cost(tour)
            tour.est_reward = sum(self.task_by_id[t].reward for t in tour.task_ids
                                  if t in self.task_by_id)
        return solution

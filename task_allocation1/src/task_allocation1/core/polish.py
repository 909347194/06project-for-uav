"""交付后局部优化：归属改动 + 组内 2-opt/Or-opt（四道门，全程无随机）。

四道门：① 航程门（改后各航次 ≤ 剩余航程）；② 迟到门（迟到数、迟到总量逐分量不劣化）；
③ 执行集门（覆盖只增不减）；④ 成本门（总航程严格下降 > ε）。
同巢口径：所有候选航次终点巢恒 = 起飞巢。
"""

from __future__ import annotations

from ..contract.config import SolverConfig
from ..contract.solution import AllocationSolution
from .problem import SelectiveProblem
from .repair import DecodeRepair
from .tour import arrival_times, lateness_of, tour_path_cost


class RoutePolisher:
    def __init__(self, problem: SelectiveProblem, decoder: DecodeRepair,
                 config: SolverConfig) -> None:
        self.problem = problem
        self.decoder = decoder
        self.config = config

    def polish(self, solution: AllocationSolution) -> AllocationSolution:
        if not self.config.polish_enabled:
            return solution
        routes = {t.uav_id: list(t.task_ids) for t in solution.tours}
        base_exec = set(solution.executed_task_ids)
        base_path = self._path(routes)
        base_late = self._lateness(routes)
        eps = float(self.config.cost_polish_eps)

        for _ in range(max(1, self.config.polish_rounds)):
            improved = False
            # 层 1：跨机 relocate / swap
            for uid, tids in list(routes.items()):
                for pos in range(len(tids)):
                    tid = tids[pos]
                    for other in list(routes):
                        if other == uid:
                            continue
                        for op in ("relocate", "swap"):
                            cand = self._apply(routes, uid, other, tid, pos, op)
                            if cand is None:
                                continue
                            if self._accept(cand, routes, base_exec, base_late, base_path, eps):
                                routes = cand
                                base_path = self._path(routes)
                                base_late = self._lateness(routes)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # 层 2：组内 2-opt / Or-opt
            for uid, tids in list(routes.items()):
                if len(tids) < 2:
                    continue
                for i in range(len(tids)):
                    for j in range(i + 1, len(tids)):
                        cand = dict(routes)
                        seq = list(tids)
                        seq[i:j + 1] = list(reversed(seq[i:j + 1]))
                        cand[uid] = seq
                        if self._accept(cand, routes, base_exec, base_late, base_path, eps):
                            routes = cand
                            base_path = self._path(routes)
                            base_late = self._lateness(routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return self.decoder.assemble_external(routes)

    # ---- 候选与门 -------------------------------------------------------
    def _apply(self, routes: dict[int, list[int]], src: int, dst: int,
               tid: int, pos: int, op: str) -> dict[int, list[int]] | None:
        cand = {u: list(t) for u, t in routes.items()}
        if op == "relocate":
            cand[src].remove(tid)
            cand[dst].append(tid)
        else:  # swap：把 src 的 tid 与 dst 的最后一个任务对换
            if not cand[dst]:
                return None
            other = cand[dst][-1]
            cand[src][pos] = other
            cand[dst][-1] = tid
        return cand

    def _accept(self, cand: dict[int, list[int]], base: dict[int, list[int]],
                base_exec: set[int], base_late: tuple[int, float],
                base_path: float, eps: float) -> bool:
        # 执行集门：只增不减
        exec_now = {t for tids in cand.values() for t in tids}
        if not base_exec.issubset(exec_now):
            return False
        # 航程门
        for uid, tids in cand.items():
            u = self.problem.uav_by_id.get(uid)
            if u is None:
                return False
            c = tour_path_cost(self.problem.costs, uid, tids, u.nest_id, u.position)
            if c > float(u.remaining_range) + 1e-6:
                return False
        # 迟到门（逐分量不劣化）
        late = self._lateness(cand)
        if late[0] > base_late[0] or (late[0] == base_late[0] and late[1] > base_late[1] + 1e-6):
            return False
        # 成本门：严格下降
        return self._path(cand) < base_path - eps

    def _path(self, routes: dict[int, list[int]]) -> float:
        total = 0.0
        for uid, tids in routes.items():
            u = self.problem.uav_by_id.get(uid)
            if u is not None:
                total += tour_path_cost(self.problem.costs, uid, tids, u.nest_id, u.position)
        return float(total)

    def _lateness(self, routes: dict[int, list[int]]) -> tuple[int, float]:
        n = 0; total = 0.0
        for uid, tids in routes.items():
            u = self.problem.uav_by_id.get(uid)
            if u is None or not tids:
                continue
            seq = [self.problem.task_by_id[t] for t in tids if t in self.problem.task_by_id]
            arrivals = arrival_times(u.position, seq, speed=float(u.speed),
                                     current_time=float(self.problem.config.current_time),
                                     service_time=float(self.problem.config.default_service_time),
                                     travel=self.problem.costs.travel)
            a, b, _ = lateness_of(arrivals, seq)
            n += a; total += b
        return n, float(total)

"""解码修复链：原始 assignment → 可行 AllocationSolution（修到不动点）。

同巢口径：终点机巢在装配期恒写为 ``uav.nest_id``（=起点），不做任何异巢选择、
不做最小费用流。返航代价照常计入航程与目标。修复链算子：
    _consolidate → _repair_range → _complete_coverage → _reorder_for_deadlines
    → _repair_range → _assemble
被弹掉的任务改派救援（reflow）：插入现有航次 / 闲置机开班 / 换机腾位。
"""

from __future__ import annotations

from collections.abc import Sequence

from ..contract.config import SolverConfig
from ..contract.entities import UAV, Task
from ..contract.solution import AllocationSolution, UAVTour
from .cost import CostProvider
from .problem import SelectiveProblem
from .tour import arrival_times, lateness_of, tour_path_cost


class DecodeRepair:
    """解码修复执行器：持有问题实例 + 参与机 + 局部下标↔实体映射。"""

    def __init__(self, problem: SelectiveProblem, participating: list[UAV],
                 tasks: list[Task], config: SolverConfig) -> None:
        self.problem = problem
        self.participating = participating
        self.tasks = tasks
        self.config = config
        self.costs: CostProvider = problem.costs

    # ---- 主入口 ---------------------------------------------------------
    def build_solution(self, assignment: Sequence[tuple[int, int]]) -> AllocationSolution:
        routes = self._to_routes(assignment)
        for _ in range(max(1, self.config.repair_max_passes)):
            before = self._signature(routes)
            routes = self._consolidate(routes)
            routes = self._repair_range(routes)
            routes = self._complete_coverage(routes)
            if self._signature(routes) == before:
                break
        routes = self._reorder_for_deadlines(routes)
        routes = self._repair_range(routes)
        return self._assemble(routes)

    def _signature(self, routes: dict[int, list[int]]) -> tuple:
        return tuple((u, tuple(t)) for u, t in sorted(routes.items()))

    def _to_routes(self, assignment: Sequence[tuple[int, int]]) -> dict[int, list[int]]:
        """局部下标 assignment → 真实 UAV id 的 routes（任务去重、越界丢弃）。"""
        routes: dict[int, list[int]] = {}
        seen: set[int] = set()
        k = len(self.participating)
        for u_local, t_local in assignment:
            if not (0 <= u_local < k) or not (0 <= t_local < len(self.tasks)):
                continue
            tid = int(self.tasks[t_local].id)
            if tid in seen:
                continue
            seen.add(tid)
            uid = self.participating[int(u_local)].id
            routes.setdefault(uid, []).append(tid)
        # 保证每个参与机都有键（含空航次）
        for u in self.participating:
            routes.setdefault(u.id, [])
        return routes

    # ---- 航次代价 / 可行性 ----------------------------------------------
    def _tour_cost(self, uid: int, tids: Sequence[int]) -> float:
        u = self.problem.uav_by_id.get(uid)
        if u is None:
            return float("inf")
        return tour_path_cost(self.costs, uid, tids, u.nest_id, u.position)

    def _range_ok(self, uid: int, tids: Sequence[int]) -> bool:
        u = self.problem.uav_by_id.get(uid)
        if u is None:
            return False
        return self._tour_cost(uid, tids) <= float(u.remaining_range) + 1e-6

    def _deadline_penalty(self, uid: int, tids: Sequence[int]) -> tuple[int, float]:
        u = self.problem.uav_by_id.get(uid)
        if u is None or not tids:
            return 0, 0.0
        seq = [self.problem.task_by_id[t] for t in tids if t in self.problem.task_by_id]
        arrivals = arrival_times(u.position, seq, speed=float(u.speed),
                                 current_time=float(self.problem.config.current_time),
                                 service_time=float(self.problem.config.default_service_time),
                                 travel=self.costs.travel)
        n, total, _ = lateness_of(arrivals, seq)
        return n, total

    # ---- 算子：合并 -----------------------------------------------------
    def _consolidate(self, routes: dict[int, list[int]]) -> dict[int, list[int]]:
        """任务少时向更少机合并（只改派不删任务；合并须落进受体航程预算）。"""
        work = {u: list(t) for u, t in routes.items()}
        active = [u for u in work if work[u]]
        if len(active) <= 1:
            return work
        donors = sorted(active, key=lambda u: len(work[u]))
        for src in donors:
            if not work[src]:
                continue
            for tid in list(work[src]):
                placed = False
                for dst in sorted(active):
                    if dst == src or not work[dst]:
                        continue
                    cand = list(work[dst]) + [tid]
                    if self._range_ok(dst, cand):
                        work[dst] = cand
                        work[src].remove(tid)
                        placed = True
                        break
                if placed:
                    break
        return work

    # ---- 算子：航程修复 + 改派 ------------------------------------------
    def _repair_range(self, routes: dict[int, list[int]]) -> dict[int, list[int]]:
        work = {u: list(t) for u, t in routes.items()}
        for _ in range(len(self.tasks) + 1):
            over = [u for u in work if work[u] and not self._range_ok(u, work[u])]
            if not over:
                break
            moved = False
            for uid in over:
                tids = work[uid]
                # 移除“移除后节省最多”的任务
                base = self._tour_cost(uid, tids)
                tid = max(tids, key=lambda t: base - self._tour_cost(uid, [x for x in tids if x != t]))
                rest = [x for x in tids if x != tid]
                if self._reflow(tid, uid, work):
                    work[uid] = rest
                    moved = True
                    break
            if not moved:
                break  # 全不可行：留在航次，由 feasibility 记违约（证明性暂缓）
        return work

    def _reflow(self, tid: int, exclude_uid: int, work: dict[int, list[int]]) -> bool:
        # ① 插入现有航次（最省代价的位置）
        best: tuple[float, int, int] | None = None
        for uid, tids in work.items():
            if uid == exclude_uid or not tids:
                continue
            for pos in range(len(tids) + 1):
                cand = list(tids); cand.insert(pos, tid)
                if self._range_ok(uid, cand):
                    c = self._tour_cost(uid, cand)
                    if best is None or c < best[0]:
                        best = (c, uid, pos)
        if best is not None:
            _, uid, pos = best
            work[uid].insert(pos, tid)
            return True
        # ② 闲置机开班
        idle = [u for u in work if not work[u] and u != exclude_uid]
        for uid in sorted(idle):
            if self._range_ok(uid, [tid]):
                work[uid] = [tid]
                return True
        # ③ 换机腾位：把受体机最后一个任务换给第三机
        for uid, tids in work.items():
            if uid == exclude_uid or not tids:
                continue
            tail = tids[-1]
            for other, other_tids in work.items():
                if other in (uid, exclude_uid) or not self._range_ok(other, other_tids + [tail]):
                    continue
                cand = list(tids[:-1]) + [tid]
                if self._range_ok(uid, cand):
                    work[other] = work[other] + [tail]
                    work[uid] = cand
                    return True
        return False

    # ---- 算子：覆盖补全 -------------------------------------------------
    def _complete_coverage(self, routes: dict[int, list[int]]) -> dict[int, list[int]]:
        assigned = {t for tids in routes.values() for t in tids}
        work = routes
        for task in self.tasks:
            if int(task.id) in assigned:
                continue
            for uid in sorted(work):
                for pos in range(len(work[uid]) + 1):
                    cand = list(work[uid]); cand.insert(pos, int(task.id))
                    if self._range_ok(uid, cand):
                        work[uid] = cand
                        assigned.add(int(task.id))
                        break
                if int(task.id) in assigned:
                    break
        return work

    # ---- 算子：时间窗重排（纯调序）--------------------------------------
    def _reorder_for_deadlines(self, routes: dict[int, list[int]]) -> dict[int, list[int]]:
        work = {u: list(t) for u, t in routes.items()}
        for uid, tids in work.items():
            if len(tids) < 2:
                continue
            base = self._deadline_penalty(uid, tids)
            best_seq, best_key = tids, base
            for i in range(len(tids)):
                for j in range(i + 1, len(tids)):
                    cand = list(tids); cand[i], cand[j] = cand[j], cand[i]
                    key = self._deadline_penalty(uid, cand)
                    if key < best_key:
                        best_seq, best_key = cand, key
            work[uid] = best_seq
        return work

    # ---- 装配：终点巢恒 = 起飞巢 ----------------------------------------
    def assemble_external(self, routes: dict[int, list[int]]) -> AllocationSolution:
        """供 polish 等外部算子装配：同一步骤、同一同巢口径。"""
        return self._assemble(routes)

    def _assemble(self, routes: dict[int, list[int]]) -> AllocationSolution:
        tours: list[UAVTour] = []
        executed: list[int] = []
        for u in self.participating:
            tids = [t for t in routes.get(u.id, []) if t in self.problem.task_by_id]
            # 同巢：end_nest_id 显式写为起飞巢（UAVTour.from_uav 缺省即此）
            tour = UAVTour.from_uav(u, task_ids=tids, end_nest_id=u.nest_id)
            tours.append(tour)
            executed.extend(tids)
        sol = AllocationSolution(
            tours=tours,
            candidate_task_ids=[t.id for t in self.tasks],
            executed_task_ids=list(dict.fromkeys(executed)),
        )
        self.problem.evaluate(sol)
        return sol

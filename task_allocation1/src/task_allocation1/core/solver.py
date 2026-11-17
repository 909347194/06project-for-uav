"""顶层编排：单周期分配主入口 ``allocate(request) -> AllocateResult``。

流程：
    可用机筛 → 建代价矩阵（距离 fn 注入）→ 选参与机 K
    → 组装 (K+N,N) 代价矩阵 → DMDE 搜索（每体经解码修复链评价）
    → 重建交付解 → 覆盖门禁审计 → 交付后 polish → 交付闸 → 结果。

终点机巢全程恒 = 起飞巢（同巢起降）；不提供异巢降落分支。
本模块零 I/O：所有实体在内存、距离由注入函数给出。
"""

from __future__ import annotations

import numpy as np

from ..contract.config import ProblemConfig, SolverConfig
from ..contract.distance import DistanceFn
from ..contract.entities import UAV, Nest, Task
from ..contract.enums import ModelType
from ..contract.request import AllocateRequest
from ..contract.result import AllocateResult
from ..contract.solution import AllocationSolution
from .cost import build_cost_matrix
from .coverage import audit_coverage
from .dmde import DMDESolver
from .encoding import model_type_of
from .participating import select_participating_uavs
from .polish import RoutePolisher
from .problem import Feasibility, SelectiveProblem
from .repair import DecodeRepair
from .vertical_section import build_distance_fn


def allocate(request: AllocateRequest) -> AllocateResult:
    """执行一次单周期任务分配（同巢起降）。纯计算：无 I/O、无全局状态。"""
    tasks: list[Task] = list(request.tasks)
    all_uavs: list[UAV] = list(request.uavs)
    nests: list[Nest] = list(request.nests)
    avail = [u for u in all_uavs if u.is_available]
    fleet = list(request.fleet_uavs) if request.fleet_uavs is not None else all_uavs

    pcfg: ProblemConfig = request.problem_config
    if request.current_time is not None:
        from dataclasses import replace
        pcfg = replace(pcfg, current_time=float(request.current_time))
    scfg: SolverConfig = request.solver_config
    # 距离口径：显式注入的 fn 优先；否则按 cost_config 解析（默认垂直切面）
    dist = _resolve_distance(request)

    if not tasks or not avail:
        problem = SelectiveProblem(tasks, avail, nests,
                                   build_cost_matrix(tasks, avail, nests,
                                                     dist).provider,
                                   config=pcfg, fleet_uavs=fleet)
        empty = problem.empty_solution()
        problem.evaluate(empty)
        return AllocateResult(solution=empty, model_type=ModelType.EMPTY,
                              diagnostics={"reason": "empty_tasks_or_uavs"})

    # ① 代价矩阵（全部可用机 × 任务）与选机
    base = build_cost_matrix(tasks, avail, nests, distance_fn=dist)
    row_of = {u.id: i for i, u in enumerate(avail)}
    part = select_participating_uavs(avail, len(tasks), scfg,
                                     uav_target=base.uav_target, row_of_uav=row_of)
    k = len(part)
    ut = np.vstack([base.uav_target[row_of[u.id]] for u in part])
    cost_build = build_cost_matrix(tasks, part, nests,
                                   distance_fn=dist,
                                   uav_target=ut)

    problem = SelectiveProblem(tasks, part, nests, cost_build.provider,
                               config=pcfg, fleet_uavs=fleet)
    decoder = DecodeRepair(problem, part, tasks, scfg)

    def evaluate(ind) -> float:
        sol = decoder.build_solution(ind.assignment)
        _audit_and_stamp(problem, decoder, sol)
        return problem.scalar_fitness(sol)

    # ② DMDE 搜索
    model = model_type_of(k, len(tasks))
    solver = DMDESolver(
        pop_size=scfg.pop_size, max_generations=scfg.max_generations,
        zeta=scfg.zeta, extinction_delta=scfg.extinction_delta,
        seed=scfg.seed, assign_factor=scfg.assign_perturb_factor,
        tour_factor=scfg.tour_perturb_factor, temp_gate=scfg.perturb_temp_gate,
    )
    dmde = solver.solve(cost_build.combined(), k, len(tasks), evaluate, model=model)

    # ③ 交付解：重建 → 门禁审计 → polish → 再审计
    if dmde.best is None:
        final = problem.empty_solution()
        problem.evaluate(final)
        return AllocateResult(solution=final, model_type=model)
    final = decoder.build_solution(dmde.best.assignment)
    _audit_and_stamp(problem, decoder, final)
    polisher = RoutePolisher(problem, decoder, scfg)
    polished = polisher.polish(final)
    if ((polished.feasible or not final.feasible)
            and problem.scalar_fitness(polished) <= problem.scalar_fitness(final)):
        final = polished
    _audit_and_stamp(problem, decoder, final)

    hard = int((final.meta.get("feasibility") or {}).get("n_hard", 0))
    return AllocateResult(
        solution=final,
        participating_uav_ids=[u.id for u in part],
        model_type=model,
        n_evals=dmde.n_evals,
        cost_history=list(dmde.cost_history),
        diagnostics={
            "n_hard": hard,
            "home_return_ok": final.home_return_ok(),
            "repair_status": "OK" if hard == 0 else "REPAIR_FAILED",
            "gate_stats": dict(final.meta.get("gate_stats") or {}),
        },
    )


def _resolve_distance(request: AllocateRequest) -> DistanceFn:
    """解析距离口径：显式 ``distance_fn`` 优先；否则按 ``cost_config`` 现装配。

    ``vertical_section``（默认口径）的采样器由调用方给（``ground_z`` / ``obstacle_top_z``）；
    两者都没给时明确报错，而不是悄悄退回欧氏——口径错误必须暴露。无地形数据请显式
    传 ``CostConfig(model="euclid3d")``。
    """
    if request.distance_fn is not None:
        return request.distance_fn
    return build_distance_fn(
        request.cost_config,
        ground_z=request.ground_z,
        obstacle_top_z=request.obstacle_top_z,
    )


def _audit_and_stamp(problem: SelectiveProblem, decoder: DecodeRepair,
                     solution: AllocationSolution) -> None:
    """覆盖门禁审计 → 写 gate_stats(n_skip) → 重算可行性/适应度元数据。"""
    stats = audit_coverage(problem, decoder, solution)
    solution.meta["gate_stats"] = stats.to_dict()
    solution.meta["feasibility_obj"] = Feasibility()
    problem.evaluate(solution)

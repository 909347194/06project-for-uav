"""同巢起降核心契约（本库的立身之本）。

1. UAVTour.from_uav 缺省终点 = 起飞巢；
2. 端到端：任何解每条非空航次 end_nest_id == start_nest_id；
3. 终点不是自由变量：即使外部塞入异巢终点，feasibility 记硬违约、解不可行；
4. 不同场景种子下不变量恒成立。
"""

from __future__ import annotations

import pytest
from task_allocation1.core.problem import HARD_HOME_TAG

import task_allocation1 as ta1


def _scenario(seed: int = 0):
    tasks = [ta1.Task(id=i, x=float((i % 6) * 700), y=float((i // 6) * 700),
                      reward=1.0, latest=1e6) for i in range(9)]
    uavs = [
        ta1.UAV(id=0, x=0, y=0, nest_id=0, speed=15.0, remaining_range=1e6),
        ta1.UAV(id=1, x=2100, y=0, nest_id=1, speed=15.0, remaining_range=1e6),
        ta1.UAV(id=2, x=0, y=700, nest_id=2, speed=15.0, remaining_range=1e6),
    ]
    nests = [
        ta1.Nest(id=0, x=0, y=0, capacity=5),
        ta1.Nest(id=1, x=2100, y=0, capacity=5),
        ta1.Nest(id=2, x=0, y=700, capacity=5),
    ]
    cfg = ta1.SolverConfig(pop_size=10, max_generations=8, seed=seed)
    return ta1.AllocateRequest(tasks=tasks, uavs=uavs, nests=nests, solver_config=cfg)


def test_from_uav_default_end_nest_is_home() -> None:
    u = ta1.UAV(id=3, x=10, y=20, nest_id=7)
    tour = ta1.UAVTour.from_uav(u, task_ids=[1])
    assert tour.start_nest_id == 7
    assert tour.end_nest_id == 7
    assert tour.is_home_return()


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_end_to_end_home_return_invariant(seed: int) -> None:
    res = ta1.allocate(_scenario(seed))
    assert res.home_return_ok is True
    assert res.solution.home_return_ok() is True
    for tour in res.solution.tours:
        if tour.task_ids:
            assert tour.end_nest_id == tour.start_nest_id
            assert tour.end_nest_id in {0, 1, 2}


def test_foreign_end_nest_is_hard_violation() -> None:
    """外部篡改终点为异巢 → 硬违约且解不可行（同巢是硬约束，不是软偏好）。"""
    req = _scenario(0)
    res = ta1.allocate(req)
    sol = res.solution
    active = next(t for t in sol.tours if t.task_ids)
    active.end_nest_id = (active.start_nest_id + 1) % 3  # 异巢

    from task_allocation1.core.cost import build_cost_matrix
    from task_allocation1.core.problem import SelectiveProblem
    provider = build_cost_matrix(req.tasks, req.uavs, req.nests).provider
    problem = SelectiveProblem(req.tasks, req.uavs, req.nests, provider)
    rep = problem.check_feasible(sol)
    assert not sol.home_return_ok()
    assert any(HARD_HOME_TAG in v for v in rep.violations)
    assert rep.feasible is False


def test_empty_scenario_preserves_invariant() -> None:
    req = ta1.AllocateRequest(tasks=[], solver_config=ta1.SolverConfig(seed=0))
    res = ta1.allocate(req)
    assert res.solution.home_return_ok()
    assert res.model_type == ta1.ModelType.EMPTY

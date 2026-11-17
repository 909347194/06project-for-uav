"""端到端求解：完成度、可复现、三段主线（搜索+修复+polish）行为、边界。"""

from __future__ import annotations

import task_allocation1 as ta1


def _req(seed: int = 0, gens: int = 6):
    tasks = [ta1.Task(id=i, x=float(i * 600), y=0.0, reward=2.0, latest=1e7)
             for i in range(8)]
    uavs = [
        ta1.UAV(id=0, x=0, y=0, nest_id=0, speed=12.0, remaining_range=1e6),
        ta1.UAV(id=1, x=3000, y=0, nest_id=1, speed=12.0, remaining_range=1e6),
    ]
    nests = [ta1.Nest(id=0, x=0, y=0, capacity=4), ta1.Nest(id=1, x=3000, y=0, capacity=4)]
    return ta1.AllocateRequest(
        tasks=tasks, uavs=uavs, nests=nests,
        cost_config=ta1.CostConfig(model="euclid3d"),  # 平坦合成场景：显式欧氏
        solver_config=ta1.SolverConfig(pop_size=8, max_generations=gens, seed=seed),
    )


def test_all_tasks_delivered_when_capacity_allows() -> None:
    res = ta1.allocate(_req())
    assert sorted(res.solution.executed_task_ids) == list(range(8))
    assert res.solution.validate_unique_tasks()
    assert res.home_return_ok
    assert res.diagnostics["repair_status"] == "OK"


def test_same_seed_is_bit_reproducible() -> None:
    a = ta1.allocate(_req(seed=11, gens=4))
    b = ta1.allocate(_req(seed=11, gens=4))
    assert a.solution.executed_task_ids == b.solution.executed_task_ids
    assert a.solution.tours[0].task_ids == b.solution.tours[0].task_ids
    assert a.cost_history == b.cost_history
    assert a.n_evals == b.n_evals


def test_search_budget_changes_n_evals() -> None:
    small = ta1.allocate(_req(seed=3, gens=2))
    large = ta1.allocate(_req(seed=3, gens=10))
    assert large.n_evals > small.n_evals
    assert len(large.cost_history) == 11  # 初始 + 每代一条


def test_polish_off_still_feasible_and_home() -> None:
    req = _req()
    req.solver_config = ta1.SolverConfig(pop_size=6, max_generations=3, seed=2,
                                         polish_enabled=False)
    res = ta1.allocate(req)
    assert res.solution.feasible or res.diagnostics["n_hard"] == 0
    assert res.home_return_ok


def test_cost_history_non_increasing() -> None:
    res = ta1.allocate(_req(seed=5, gens=10))
    h = res.cost_history
    assert all(h[i + 1] <= h[i] + 1e-6 for i in range(len(h) - 1))

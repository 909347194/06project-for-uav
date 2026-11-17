"""参与口径不变量（端到端）：任务唯一分配、一机可串飞、机多于任务时允许闲置。

本库口径：永不使用 BALANCED / OVERLOADED —— 一个任务只能被一架无人机完成；
参与机数 K <= max(1, N-1)；无人机多于任务时，未选中的机闲置。
"""

from __future__ import annotations

import pytest

import task_allocation1 as ta1


def _req(n_tasks: int, n_uavs: int, *, remaining_range: float = 1e5):
    tasks = [ta1.Task(id=i, x=500.0 * (i + 1), y=0.0, reward=1.0, latest=1e6)
             for i in range(n_tasks)]
    uavs = [ta1.UAV(id=i, x=0.0, y=0.0, nest_id=0, speed=15.0,
                    remaining_range=remaining_range) for i in range(n_uavs)]
    nests = [ta1.Nest(id=0, x=0.0, y=0.0, capacity=max(1, n_uavs))]
    return ta1.AllocateRequest(
        tasks=tasks, uavs=uavs, nests=nests,
        cost_config=ta1.CostConfig(model="euclid3d"),
        solver_config=ta1.SolverConfig(pop_size=20, max_generations=30, seed=3),
    )


def test_single_task_goes_through_srp() -> None:
    """N=1 不再走 BALANCED：model_type 必须是 SRP，巡游维为 0。"""
    res = ta1.allocate(_req(1, 3))
    assert res.model_type == ta1.ModelType.SRP
    m = res.metrics_dict()
    assert m["n_participating"] == 1
    assert len(res.solution.tours) == 1
    assert res.solution.tours[0].task_ids == [0]
    assert m["path_m"] == pytest.approx(1000.0, abs=1e-6)  # 500 去 + 500 回


@pytest.mark.parametrize("n_tasks,n_uavs", [(2, 5), (5, 5), (6, 4), (10, 3)])
def test_participating_capped_and_idle_allowed(n_tasks: int, n_uavs: int) -> None:
    """K <= max(1, N-1) 恒成立；机多于任务时必然有闲置机。"""
    res = ta1.allocate(_req(n_tasks, n_uavs))
    assert res.model_type == ta1.ModelType.SRP
    k = len(res.participating_uav_ids)
    assert k <= max(1, n_tasks - 1)
    assert k <= n_uavs
    assert set(res.participating_uav_ids) <= set(range(n_uavs))
    if n_uavs > n_tasks:
        assert len(set(range(n_uavs)) - set(res.participating_uav_ids)) >= 1


@pytest.mark.parametrize("n_tasks,n_uavs", [(2, 5), (5, 5), (6, 4), (10, 3)])
def test_each_task_assigned_exactly_once(n_tasks: int, n_uavs: int) -> None:
    """任务唯一分配：每个 task_id 在所有航次里恰好出现一次，不重不漏。"""
    res = ta1.allocate(_req(n_tasks, n_uavs))
    ids = [tid for tour in res.solution.tours for tid in tour.task_ids]
    assert sorted(ids) == list(range(n_tasks))


def test_one_uav_can_carry_multiple_tasks() -> None:
    """任务数多于机数时，一机串飞多任务是允许形态（SRP 的“巡游”）。"""
    res = ta1.allocate(_req(4, 2))
    assert res.model_type == ta1.ModelType.SRP
    counts = [len(t.task_ids) for t in res.solution.tours if t.task_ids]
    assert max(counts) >= 2

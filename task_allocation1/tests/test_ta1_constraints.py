"""约束与反映射单测：编码/反映射布局、航程约束、距离注入下的可行性。"""

from __future__ import annotations

import numpy as np
import pytest
from task_allocation1.contract.entities import UAV, Nest, Task
from task_allocation1.contract.enums import ModelType
from task_allocation1.core.cost import build_cost_matrix
from task_allocation1.core.encoding import PopulationEncoder, model_type_of
from task_allocation1.core.inverse_map import inverse_phi
from task_allocation1.core.problem import SelectiveProblem


def test_model_type_only_srp_and_empty() -> None:
    """口径：任务唯一分配、K <= max(1, N-1)；本库只产 SRP / EMPTY。"""
    assert model_type_of(2, 5) == ModelType.SRP
    assert model_type_of(1, 2) == ModelType.SRP
    assert model_type_of(1, 1) == ModelType.SRP      # N=1：K=1，巡游维为 0
    assert model_type_of(3, 5) == ModelType.SRP
    assert model_type_of(0, 0) == ModelType.EMPTY
    assert model_type_of(0, 3) == ModelType.EMPTY


@pytest.mark.parametrize("k,n", [(3, 3), (5, 3), (2, 1)])
def test_model_type_rejects_balanced_and_overloaded(k: int, n: int) -> None:
    """K=N（balanced）与 K>N（overloaded，含 N=1 时 K>1）必须显式拒绝。"""
    with pytest.raises(ValueError):
        model_type_of(k, n)


def test_encoding_cost_vector_fixed_layout() -> None:
    cm = np.array([[1., 2., 3., 4., 5.], [9., 8., 7., 6., 5.],
                   [0., 10., 11., 12., 13.], [10., 0., 14., 15., 16.],
                   [11., 14., 0., 17., 18.]])
    enc = PopulationEncoder(cm, n_uavs=2, n_targets=3)
    rng = np.random.default_rng(0)
    pop = enc.generate(4, rng)
    assert len(pop) == 4
    for ind in pop:
        assert len(ind.cost_vector) == 3  # 2 起始 + 1 巡游


def test_inverse_phi_returns_unique_targets() -> None:
    cm = np.array([[100., 200., 300.],
                   [150., 250., 350.],
                   [0., 120., 220.],
                   [120., 0., 180.],
                   [220., 180., 0.]])
    cv = np.array([105., 255., 125.])
    ind = inverse_phi(cv, cm, 2, 3, ModelType.SRP, np.random.default_rng(0))
    assert ind.assignment
    tids = [t for _, t in ind.assignment]
    assert len(tids) == len(set(tids)) == 3   # 每个任务恰好一次


@pytest.mark.parametrize("model", [ModelType.BALANCED, ModelType.OVERLOADED])
def test_inverse_phi_rejects_disabled_models(model: ModelType) -> None:
    cm = np.eye(4)
    cv = np.array([1., 1.])
    with pytest.raises(ValueError):
        inverse_phi(cv, cm, 2, 2, model, np.random.default_rng(0))


def test_range_violation_detected() -> None:
    """注入极大距离 → 航程超限 → n_hard>0 且不可行。"""
    tasks = [Task(id=0, x=5e6, y=0), Task(id=1, x=5e6, y=5e6)]
    uavs = [UAV(id=0, x=0, y=0, nest_id=0, speed=10.0, remaining_range=1e3)]
    nests = [Nest(id=0, x=0, y=0)]
    provider = build_cost_matrix(tasks, uavs, nests).provider
    problem = SelectiveProblem(tasks, uavs, nests, provider)
    from task_allocation1.contract.solution import AllocationSolution, UAVTour
    u = uavs[0]
    sol = AllocationSolution(
        tours=[UAVTour.from_uav(u, task_ids=[0, 1], end_nest_id=u.nest_id)],
        executed_task_ids=[0, 1],
    )
    rep = problem.check_feasible(sol)
    assert rep.range_excess > 0
    assert rep.n_hard > 0
    assert not rep.feasible

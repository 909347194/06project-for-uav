"""顶层门面守卫：后端/上层只需 ``import task_allocation1 as ta1``。"""

from __future__ import annotations

from task_allocation1.core.solver import allocate as core_allocate

import task_allocation1 as ta1

_REQUIRED = {
    "ALGORITHM_ID", "AllocateRequest", "AllocateResult", "AllocationSolution",
    "ConstraintConfig", "CostConfig", "DistanceFn", "EndNestPolicy", "ModelType",
    "Nest", "ProblemConfig", "SolverConfig", "SurfaceSampler", "Task", "TaskStatus",
    "UAV", "UAVStatus", "UAVTour", "VerticalSectionCost", "__version__",
    "allocate", "build_distance_fn", "contract", "defaults", "euclidean3d", "terrain",
}


def test_public_api_is_flat_and_complete() -> None:
    assert set(ta1.__all__) == _REQUIRED
    for name in ta1.__all__:
        assert getattr(ta1, name) is not None, name


def test_facade_reexports_real_implementation() -> None:
    assert ta1.allocate is core_allocate
    assert ta1.ALGORITHM_ID == "ta1-dmde-home-nest"
    assert ta1.__version__ == "0.2.0"


def test_backend_can_build_request_from_one_import() -> None:
    req = ta1.AllocateRequest(
        tasks=[ta1.Task(id=1, x=0, y=0)],
        uavs=[ta1.UAV(id=0, x=0, y=0, nest_id=0)],
        nests=[ta1.Nest(id=0, x=0, y=0)],
        solver_config=ta1.SolverConfig(pop_size=6, max_generations=2, seed=1),
    )
    assert len(req.tasks) == 1
    assert callable(ta1.allocate)
    assert ta1.euclidean3d((0, 0, 0), (3, 4, 0)) == 5.0

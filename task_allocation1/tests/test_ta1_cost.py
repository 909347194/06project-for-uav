"""代价模型：节点键编码、有向性、距离注入、pairwise 直通。"""

from __future__ import annotations

import math

from task_allocation1.contract.entities import UAV, Nest, Task
from task_allocation1.core.cost import (
    build_cost_matrix,
    build_cost_provider,
    nest_key,
    uav_start_key,
)


def _scene():
    tasks = [Task(id=0, x=0, y=0), Task(id=1, x=100, y=0)]
    nests = [Nest(id=0, x=0, y=0), Nest(id=7, x=50, y=50)]
    uavs = [UAV(id=0, x=0, y=0, nest_id=0)]
    return tasks, nests, uavs


def test_node_keys_disjoint() -> None:
    assert nest_key(0) == 1_000_000
    assert nest_key(7) == 1_000_007
    assert uav_start_key(0) == -1
    assert uav_start_key(3) == -4
    assert len({nest_key(0), uav_start_key(0), 0}) == 3


def test_distance_fn_injected_and_directed() -> None:
    tasks, nests, uavs = _scene()

    # 有向距离：A→B = 10，B→A = 99（模拟爬升不对称）
    def directed(a, b):
        return 10.0 if a[0] < b[0] else 99.0

    p = build_cost_provider(tasks, nests, uavs, distance_fn=directed)
    assert p.cost(0, 1) == 10.0
    assert p.cost(1, 0) == 99.0  # 绝不回退反向键/欧氏
    assert p.travel((0, 0, 0), (100, 0, 0)) == 10.0  # travel 同样走注入的有向 fn
    assert p.travel((100, 0, 0), (0, 0, 0)) == 99.0
    assert p.key(7) == 1_000_007


def test_pairwise_hit_and_missing_sentinel() -> None:
    tasks, nests, uavs = _scene()
    pw = {(0, 1): 42.0}
    p = build_cost_provider(tasks, nests, uavs, pairwise=pw)
    assert p.cost(0, 1) == 42.0
    assert math.isinf(p.cost(0, 999))  # 两端位置缺失 → 惩罚哨兵


def test_cost_matrix_shape_and_idle_default() -> None:
    tasks, nests, uavs = _scene()
    b = build_cost_matrix(tasks, uavs, nests)
    assert b.uav_target.shape == (1, 2)
    assert b.task_task.shape == (2, 2)
    assert b.task_task[0, 0] == 0.0
    assert b.combined().shape == (3, 2)

"""航次原语：航程代价、到达/迟到时间、返航腿（同巢）。口径唯一来源。

返航腿固定到 UAV 的起飞巢：
    ``return_to = tour.end_nest_id``（同巢口径下恒 == tour.start_nest_id == uav.nest_id）

空航次恒 0：不起飞不产生航程（与问题层 tour_path_cost 一致）。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import pairwise

from ..contract.entities import UAV, Task
from ..contract.solution import UAVTour
from .cost import CostProvider, uav_start_key


def return_leg_cost(provider: CostProvider, last_task_id: int, end_nest_id: int | None) -> float:
    """末任务 → 终点机巢的返航代价；无终点记 +inf（缺返航义务不可行）。"""
    if end_nest_id is None:
        return math.inf
    return float(provider.cost(int(last_task_id), provider.key(int(end_nest_id))))


def tour_path_cost(provider: CostProvider, uav_id: int, task_ids: Sequence[int],
                   end_nest_id: int | None, start_position: Sequence[float] | None = None) -> float:
    """航次总代价：起点 → 任务链 → 终点巢。空航次恒 0。"""
    if not task_ids:
        return 0.0
    total = 0.0
    if start_position is not None:
        first: float = provider.cost_by_position(start_position,
                                                  provider.positions.get(int(task_ids[0]), (0, 0, 0)))
        total += first
    else:
        prev = uav_start_key(uav_id)
        total += float(provider.cost(prev, int(task_ids[0])))
    for a, b in pairwise(task_ids):
        total += float(provider.cost(int(a), int(b)))
    total += return_leg_cost(provider, task_ids[-1], end_nest_id)
    return float(total)


def arrival_times(start_position: Sequence[float], tasks: Sequence[Task], *, speed: float,
                  current_time: float, service_time: float,
                  travel) -> list[float]:
    """逐个任务的到达时刻（含早到等待 + 服务时间）。

    ``travel(pos_a, pos_b) -> 米``；到达 = max(前一离开时刻 + 段长/速度, earliest)；
    离开 = 到达 + service_time。返回每个任务的到达时刻列表。
    """
    v = max(float(speed), 1e-6)
    cursor = float(current_time)
    out: list[float] = []
    prev_pos = start_position
    for t in tasks:
        leg = float(travel(prev_pos, t.position)) / v
        arrive = cursor + leg
        if t.earliest is not None:
            arrive = max(arrive, float(t.earliest))
        out.append(float(arrive))
        cursor = arrive + float(service_time)
        prev_pos = t.position
    return out


def lateness_of(arrivals: Sequence[float], tasks: Sequence[Task]) -> tuple[int, float, dict[int, float]]:
    """返回 (迟到任务数, 迟到秒数和, {task_id: 迟到秒数})。"""
    n_late = 0
    total = 0.0
    per: dict[int, float] = {}
    for arrive, t in zip(arrivals, tasks):
        if t.latest is None:
            per[int(t.id)] = 0.0
            continue
        late = max(0.0, float(arrive) - float(t.latest))
        per[int(t.id)] = late
        if late > 0:
            n_late += 1
            total += late
    return n_late, float(total), per


def tour_endurance(uav: UAV, tour: UAVTour, task_by_id: Mapping[int, Task], *,
                   provider: CostProvider, current_time: float, service_time: float) -> float:
    """整航迹时长（秒）：飞行 + 早到等待 + 服务 + 返巢；诊断与续航判定共用。"""
    if tour.is_idle:
        return 0.0
    seq = [task_by_id[tid] for tid in tour.task_ids if tid in task_by_id]
    arrivals = arrival_times(tour.start_position, seq, speed=float(uav.speed),
                             current_time=current_time, service_time=service_time,
                             travel=provider.travel)
    if not arrivals:
        return 0.0
    cursor = arrivals[-1] + float(service_time)
    if tour.end_nest_id is not None and seq:
        nest_pos = provider.positions.get(provider.key(tour.end_nest_id))
        if nest_pos is not None:
            cursor += provider.travel(seq[-1].position, nest_pos) / max(float(uav.speed), 1e-6)
    return float(cursor - current_time)

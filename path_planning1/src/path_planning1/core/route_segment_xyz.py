"""绝对 ENU 端点 ``(x, y, z)`` 入口：只做 3 元组校验，然后原样透传。

真正的入口 ``plan_route_segment`` 本身就收绝对 z（真高由库内部按 DEM 算），
所以这里不再做任何换算；留着它只是为了"一次给一个完整的 (x, y, z)"这种写法更顺手，
以及把"少传一个数"这类错误挡在调用前。
"""

from __future__ import annotations

from collections.abc import Sequence

from ..contract import Building, LocalFrame, NoFlyZone
from .dem_terrain import Dem
from .route_segment_outcome import RouteSegmentPlan
from .route_segment_params import RouteSegmentParams
from .route_segment_planner import plan_route_segment
from .world_field import WorldField


def plan_route_segment_xyz(
    buildings: list[Building],
    frame: LocalFrame,
    dem: Dem | None,
    start_xyz: tuple[float, float, float],
    goal_xyz: tuple[float, float, float],
    params: RouteSegmentParams | None = None,
    buildings_total: int | None = None,
    *,
    world_field: WorldField | None = None,
    nofly: Sequence[NoFlyZone] = (),
) -> RouteSegmentPlan:
    """端点用 **绝对 ENU 米** ``(x, y, z)`` 的一条航段（与 ``plan_route_segment`` 等价）。

    端点 z 低于地面时真高为负，按 ``out_of_flight_band`` 返回，不会静默抬高。
    """
    start = tuple(float(v) for v in start_xyz)
    goal = tuple(float(v) for v in goal_xyz)
    for label, pt in (("start_xyz", start), ("goal_xyz", goal)):
        if len(pt) != 3:
            raise ValueError(f"{label} 必须是 3 个数 (x, y, z)（收到 {pt!r}）")
    return plan_route_segment(
        buildings, frame, dem, start[:2], goal[:2], start[2],
        params, buildings_total, goal_z_m=goal[2], world_field=world_field, nofly=nofly,
    )

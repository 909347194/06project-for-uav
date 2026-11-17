"""两库的「接口桥」：**懒加载**，谁都不被写进依赖。

设计约束（本项目的硬规矩）：

- ``task_allocation1`` 与 ``path_planning1`` 是两个**各自独立**的计算库，互不 import；
- 本包也不 import 它们（运行时依赖只有标准库）；
- 三者之所以能对接，靠的是**数据契约**：同一套 ENU 米、同一个 ``frame_hash``、同一份
  ``world.json``。下面这些函数只是把契约"翻译"成各自的入参，**import 写在函数体内**，
  装了哪个库才用得到对应的那座桥。

桥清单：``to_pp1_world_kwargs``（世界→pp1）、``to_ta1_request``（任务包→ta1 请求）、
``to_ta1_cost_samplers``（世界→ta1 垂直切面尺子的地面/建筑包络采样器）、
``check_world_with_pp1``（对拍 frame 与指纹）。

所以：只用 ta1 的服务不会因为这里而背上 shapely/casadi；只用 pp1 的服务也不会装上 ta1。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from .mission import Mission
from .world import World


def to_pp1_world_kwargs(world: World, **overrides: Any) -> dict[str, Any]:
    """世界 → pp1 的 ``world`` 段。纯 dict，**不 import pp1**。

    ``overrides`` 直接透传（如 ``with_dem=False`` / ``max_od_length_m=6000.0``）。
    """
    return world.world_kwargs(**overrides)


def to_ta1_request(
    mission: Mission,
    *,
    distance_fn: Callable[[Sequence[float], Sequence[float]], float] | None = None,
    cost_config: Any = None,
    ground_z: Any = None,
    obstacle_top_z: Any = None,
    solver_config: Any = None,
    problem_config: Any = None,
):
    """任务包 → ``ta1.AllocateRequest``（懒 import ta1）。

    ``distance_fn`` 是 ta1 的底层注入点；不传则按 ``cost_config`` 解析（默认垂直切面）。
    ``cost_config`` / ``ground_z`` / ``obstacle_top_z`` 用于 ``model="vertical_section"``：
    前两个采样器由 :func:`to_ta1_cost_samplers` 从世界现造；默认口径下至少给一个，
    否则 ta1 直接报错。无地形数据请显式传 ``ta1.CostConfig(model="euclid3d")``。
    ``solver_config`` / ``problem_config`` 不传时用任务包 ``mission.json`` 里的
    ``solver`` / ``problem`` 段。
    """
    try:
        import task_allocation1 as ta1
    except ModuleNotFoundError as exc:  # pragma: no cover - 环境问题，不是逻辑分支
        raise ModuleNotFoundError(
            "to_ta1_request 需要 task_allocation1：pip install -e task_allocation1"
            "（本包不硬依赖它，所以没有自动装上）"
        ) from exc

    return ta1.AllocateRequest(
        tasks=[ta1.Task(id=t.id, x=t.x, y=t.y, z=t.z, reward=t.reward,
                        earliest=t.earliest, latest=t.latest) for t in mission.tasks],
        uavs=[ta1.UAV(id=u.id, x=u.x, y=u.y, z=u.z, speed=u.speed,
                      remaining_range=u.remaining_range, max_time=u.max_time,
                      max_range=u.max_range, nest_id=u.nest_id,
                      status=ta1.UAVStatus(u.status)) for u in mission.uavs],
        nests=[ta1.Nest(id=n.id, x=n.x, y=n.y, z=n.z, capacity=n.capacity,
                        available=n.available) for n in mission.nests],
        distance_fn=distance_fn,
        cost_config=(cost_config if cost_config is not None
                     else ta1.CostConfig.from_dict(mission.cost)),
        ground_z=ground_z,
        obstacle_top_z=obstacle_top_z,
        problem_config=problem_config or ta1.ProblemConfig.from_dict(mission.problem),
        solver_config=solver_config or ta1.SolverConfig.from_dict(mission.solver),
        current_time=mission.current_time,
    )


def to_ta1_cost_samplers(
    world: World,
    *,
    field_mode: str = "on_demand",
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """世界 → ta1 垂直切面尺子需要的两个采样器 ``(ground_z, obstacle_top_z)``。

    返回的两个函数都接受 ``(xs, ys)``（水平 ENU 坐标序列），返回逐点**绝对 z**：

    - ``ground_z``：该点地面高程（DEM 双线性；无 DEM 时全 0）；
    - ``obstacle_top_z``：该点障碍包络顶 = ``max(地面, 其所在建筑屋顶 z_top)``。

    放进 ``ta1.AllocateRequest(..., cost_config=ta1.CostConfig(model="vertical_section"),
    ground_z=..., obstacle_top_z=...)`` 即生效；ta1 本身仍不读文件。

    地理数据的读取交给 ``path_planning1``（DEM/建筑解析在那边只维护一份），
    所以本函数懒 import pp1 + numpy + shapely：只装 ta1 的用户不会因为 import
    本包而背上 GIS 栈。``field_mode="on_demand"`` 时只建窗口不预计算场。
    """
    try:
        import numpy as np
        from path_planning1.core.dem_terrain import ground_at_xy
        from shapely import STRtree
        from shapely import points as shapely_points
        from shapely.geometry import Polygon

        import path_planning1 as pp1
    except ModuleNotFoundError as exc:  # pragma: no cover - 环境问题，不是逻辑分支
        raise ModuleNotFoundError(
            "to_ta1_cost_samplers 需要 path_planning1 + numpy + shapely："
            "pip install -e path_planning1（本包不硬依赖它们）"
        ) from exc

    cfg = {"world": {**world.world_kwargs(), "field_mode": field_mode}, "params": {}}
    dataset = pp1.create_context(cfg).dataset
    dem, frame, buildings = dataset.dem, dataset.frame, dataset.buildings
    polys = [Polygon(b.footprint) for b in buildings]
    tops = np.array([float(b.z_top_m) for b in buildings], dtype=float)
    tree = STRtree(polys) if polys else None

    def _xy(xs, ys):
        return np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])

    def ground_z(xs, ys):
        xy = _xy(xs, ys)
        if dem is None:
            return np.zeros(len(xy))
        return ground_at_xy(dem, frame, xy)

    def obstacle_top_z(xs, ys):
        base = np.asarray(ground_z(xs, ys), dtype=float)
        if tree is None or tops.size == 0:
            return base
        point_idx, tree_idx = tree.query(shapely_points(_xy(xs, ys)))
        np.maximum.at(base, point_idx, tops[tree_idx])
        return base

    return ground_z, obstacle_top_z


def check_world_with_pp1(world: World) -> dict[str, Any]:
    """**对拍**：用 pp1 自己的实现复核本包算出的 frame 与数据集指纹（懒 import pp1）。

    只给测试 / CI / 数据入库时用，不进请求路径：它要 import shapely 与 pyshp。
    返回 ``{"frame_ok", "digest_ok", "frame_pp1", "digest_pp1"}``。
    """
    try:
        from path_planning1.core.city_loader import frame_from_geojson
        from path_planning1.core.dataset_identity import dataset_digest
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "check_world_with_pp1 需要 path_planning1：pip install -e path_planning1"
        ) from exc

    pp1_frame = frame_from_geojson(world.region_path)
    same = (round(pp1_frame.lon0, 9), round(pp1_frame.lat0, 9),
            round(pp1_frame.lat_reference, 9)) == (
        round(world.frame.lon0, 9), round(world.frame.lat0, 9),
        round(world.frame.lat_reference, 9))

    pp1_digest = dataset_digest(world.world_kwargs())
    return {
        "frame_ok": same,
        "digest_ok": world.dataset_digest == pp1_digest,
        "frame_pp1": (pp1_frame.lon0, pp1_frame.lat0, pp1_frame.lat_reference),
        "digest_pp1": pp1_digest,
    }

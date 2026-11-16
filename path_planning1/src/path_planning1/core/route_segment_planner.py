"""一条航段的完整流程：ROI 走廊 → 窗口栅格 → SDF → RRT → 净空终检 → 时间域 NLP → 指标。

这是**核心规划器**：收水平坐标 + 真高，返回几何折线与可行轨迹，不暴露窗口/SDF 细节。
在线入口是 ``PlanningContext.plan``（固定传场、先准入）；这里不判城市边界与建场契约，
也不做缓存。默认数值只在 ``contract/defaults.py``，结果形状见 ``route_segment_outcome.py``。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from ..contract import (
    Building,
    Diagnostics,
    FailureCode,
    LocalFrame,
    Metrics,
    NoFlyZone,
    Status,
    defaults,
)
from .city_loader import filter_by_bbox
from .clearance_check import BuildingPrisms
from .dem_terrain import Dem, grounds_at_endpoints
from .occupancy_sdf import SdfField, build_occupancy, build_sdf
from .params_fingerprint import digest_for
from .request_check import check_request, check_vehicle
from .route_segment_outcome import RouteSegmentPlan, RouteSegmentStats
from .route_segment_params import RouteSegmentParams
from .route_segment_trajectory import fit_time_profile
from .route_segment_window import build_segment_window
from .rrt_candidates import evaluate_candidates, select_candidate
from .rrt_path_search import RRT3D
from .world_field import WorldField, field_signature


def plan_route_segment(
    buildings: list[Building],
    frame: LocalFrame,
    dem: Dem | None,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    start_z_m: float,
    params: RouteSegmentParams | None = None,
    buildings_total: int | None = None,
    *,
    goal_z_m: float | None = None,
    world_field: WorldField | None = None,
    nofly: Sequence[NoFlyZone] = (),
) -> RouteSegmentPlan:
    """规划一条航段。长度单位米，时间单位秒。

    **端点高度统一为绝对 z**（与 ``x/y`` 同一 ENU 高程基准）；真高是内部派生量：
    ``agl = z − z_ground(x, y)``（DEM 采样，无 DEM 取 0），飞行带仍按真高判。

    ``goal_z_m`` 省略 = 与起点同高（两端等高的空中航段是常见情形）。
    ``buildings_total`` 用于说明"窗口里筛掉了多少"，省略时按 ``len(buildings)`` 计。

    ``nofly`` 是静态禁飞棱柱（绝对 z、可悬浮），与建筑共用障碍通道：场路径在建场期已栅格化，
    ``on_demand`` 才现算；两类障碍都进真几何终检，端点命中禁飞区返回 ``ENDPOINT_IN_NOFLY``。

    ``world_field`` 给了就只按走廊取样，不给则是显式 ``on_demand`` 现算；两条路径 SDF 基准
    不同，已分别进 ``params_hash`` 的 ``extra``。窗口越界属建场契约错误（``FieldContractError``），
    由启动自检挡住；运行期不判断、不重建 SDF、不回退。
    """
    params = params or RouteSegmentParams()
    start_z_m = float(start_z_m)
    goal_z_m = start_z_m if goal_z_m is None else float(goal_z_m)
    check_request(start_xy, goal_xy, params.spacing_m)
    check_vehicle(params.vehicle.thrust_max_n, params.vehicle.mass_kg)
    timings: dict[str, float] = {}
    started = t0 = time.perf_counter()   # started 是墙钟起点；t0 会被各段反复重置

    # 纯入参合法性先查（不碰数据）：非法高度不该先白算 4 s 窗口 SDF。绝对 z → 真高只查 DEM，O(1)。
    digest = digest_for(params, field_signature(world_field))
    g_start, g_goal = grounds_at_endpoints(dem, frame, start_xy, goal_xy)
    for label, z, g in (("起点", start_z_m, g_start), ("终点", goal_z_m, g_goal)):
        agl = z - g
        if not (params.agl_min_m <= agl <= params.agl_max_m):
            elapsed = time.perf_counter() - started
            diag = Diagnostics(roi_used=params.roi, params_hash=digest, elapsed_s=elapsed)
            return RouteSegmentPlan(
                Status.INFEASIBLE, FailureCode.OUT_OF_FLIGHT_BAND,
                f"{label}绝对 z={z:.2f} m → 真高 {agl:.2f} m（地面 {g:.2f} m），"
                f"不在 [{params.agl_min_m}, {params.agl_max_m}] 内",
                None, Metrics(), RouteSegmentStats(), diag, {"total_s": elapsed}, digest)

    # 窗口建两次（见 route_segment_window），只是把 z 包络与地形一并装配好。
    t0 = time.perf_counter()
    window, terrain, z_lo, z_hi = build_segment_window(dem, frame, start_xy, goal_xy, params)
    timings["window_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    # 粗筛也外扩：build_occupancy 会膨胀 footprint 半格对角线，窗外贴边的楼仍可能盖住
    # 窗口内格心，漏掉就使边缘 SDF 虚高。此 pad 与下面净空度量的搜索半径不是一回事。
    occ_pad_m = float(np.hypot(*window.grid.spacing_m[:2]))
    local = filter_by_bbox(buildings, window.enu_bounds, occ_pad_m)
    local_nofly = filter_by_bbox(nofly, window.enu_bounds, occ_pad_m)
    if world_field is None:
        # on_demand：显式的低内存/调试模式，窗口现算是**唯一**路径，不是异常兜底。
        occ = build_occupancy(local, window.grid, terrain, nofly=local_nofly)
        sdf = build_sdf(occ, window.grid.spacing_m)
    else:
        # startup：只按走廊取样。覆盖够不够是建场契约，启动自检已验证，这里不判断、
        # 不重建 SDF。真越界就让 FieldContractError 冒泡成部署事故，绝不当场降级成另一套口径。
        # 全域场是轴对齐格、窗口是斜格，取样多一层三线性插值；扣掉固定补偿量，
        # 否则交付净空比窗口现算低 1–2 m（见 defaults.WORLD_FIELD_SAMPLE_BIAS_M）。
        sdf = world_field.window_sdf(window.grid) - defaults.WORLD_FIELD_SAMPLE_BIAS_M
    field = SdfField(window.grid, sdf)
    # 净空度量的粗筛要**外扩**一个邻域半径：路径可以贴窗口边界走，窗外几米的楼同样算数。
    # 不外扩 = 把"贴边的楼"漏掉、把净空虚报成 30 m（见 core/clearance_check.py 的口径第 3 条）。
    nearby_nofly = filter_by_bbox(nofly, window.enu_bounds, defaults.CLEARANCE_SEARCH_RADIUS_M)
    prisms = BuildingPrisms(
        filter_by_bbox(buildings, window.enu_bounds, defaults.CLEARANCE_SEARCH_RADIUS_M)
        + nearby_nofly,
        defaults.CLEARANCE_SEARCH_RADIUS_M,
    )
    # 端点命中禁飞区要单独报码；与"端点贴楼"不是一回事，别共用 START_OR_GOAL_BLOCKED。
    nofly_prisms = BuildingPrisms(nearby_nofly, defaults.CLEARANCE_SEARCH_RADIUS_M)
    timings["grid_s"] = time.perf_counter() - t0

    ground_start = 0.0 if terrain is None else terrain.at_xy(*start_xy)
    ground_goal = 0.0 if terrain is None else terrain.at_xy(*goal_xy)
    start = np.array([start_xy[0], start_xy[1], start_z_m])
    goal = np.array([goal_xy[0], goal_xy[1], goal_z_m])
    stats = RouteSegmentStats(
        horizontal_m=float(window.horizontal_length_m),
        straight_m=float(np.linalg.norm(goal - start)),
        corridor_width_m=2 * float(window.half_width_m),
        window_width_m=2 * float(window.lateral_m),
        window_shape=tuple(int(v) for v in window.grid.shape),
        window_cells=int(np.prod(window.grid.shape)),
        buildings_total=int(len(buildings) if buildings_total is None else buildings_total),
        buildings_in_window=len(local),
        occupied_ratio=float(field.occupied_ratio),
        ground_start_m=float(ground_start),
        ground_goal_m=float(ground_goal),
        z_envelope_m=(float(z_lo), float(z_hi)),
    )
    diagnostics = Diagnostics(roi_used=params.roi, params_hash=digest)

    def fail(code: FailureCode, message: str) -> RouteSegmentPlan:
        """统一的失败出口：补总耗时、形状与成功路径完全一致（只差 path 为 None）。"""
        # 总耗时取墙钟：分段之间（构造 stats / SDF 查询 / 建 RRT）不计入任何段，和会少报。
        timings["total_s"] = time.perf_counter() - started
        diagnostics.elapsed_s = timings["total_s"]
        return RouteSegmentPlan(
            Status.INFEASIBLE, code, message, None, Metrics(), stats, diagnostics, timings, digest
        )

    for label, point in (("起点", start), ("终点", goal)):
        if nofly_prisms.contains(point):
            msg = (f"{label} ({point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f}) 落在静态禁飞区内"
                   "（绝对高程口径）；库不会静默抬升或平移端点")
            return fail(FailureCode.ENDPOINT_IN_NOFLY, msg)

    # 剩下两种失败都要数据才能判：z 包络靠窗口内最低地面，净空靠 SDF。
    if not (z_lo <= start[2] <= z_hi and z_lo <= goal[2] <= z_hi):
        # 天花板按窗口最低地面定，高地上空真高受限（见 z_envelope_from_ground）。
        return fail(
            FailureCode.OUT_OF_FLIGHT_BAND,
            f"起终点 z=({start[2]:.1f}, {goal[2]:.1f}) 超出窗口 z 包络 [{z_lo:.1f}, {z_hi:.1f}]"
            "（天花板按窗口最低地面定，高地上空真高受限）",
        )
    need = params.clearance_m + params.sdf_tolerance_m
    have = min(float(field.at(start[None, :])[0]), float(field.at(goal[None, :])[0]))
    if have < need:
        return fail(
            FailureCode.START_OR_GOAL_BLOCKED,
            f"起终点 SDF 净空 {have:.2f} m < 阈值 {need:.2f} m"
            f"（净空 {params.clearance_m:.1f} + 竖直格距容差 {params.sdf_tolerance_m:.1f}；"
            "栅格已保守膨胀，仍不够就是这里真的没空间）",
        )

    rrt = RRT3D(
        field, window, (z_lo, z_hi), params.rrt_config(),
        terrain=terrain, agl_range_m=(params.agl_min_m, params.agl_max_m),
    )
    n_init = int(params.planner.rrt_initializations)
    t0 = time.perf_counter()
    results = rrt.plan_many(start, goal, initializations=n_init)
    timings["rrt_s"] = time.perf_counter() - t0
    diagnostics.candidates = n_init
    if not results:
        return fail(
            FailureCode.RRT_NO_PATH,
            f"RRT 在走廊带内（宽 {2 * window.half_width_m:.0f} m，含 margin 实际 "
            f"{2 * window.lateral_m:.0f} m）跑了 {n_init} 个初值仍找不到满足净空/真高带的解："
            "先查走廊是否太窄、分辨率与净空是否合适，别先调 RRT 参数",
        )

    # 真几何净空：SDF 负责"快"，真几何负责"真"。多初值先全部量一遍，再选一条给 NLP。
    # 这一步同时是最后一道闸：真几何不够就按失败返回，绝不静默给错路径。
    t_geo = time.perf_counter()
    candidates = evaluate_candidates(
        results,
        prisms,
        window,
        sample_step_m=params.planner.clearance_sample_step_m,
        nlp_need_m=need,
    )
    timings["clearance_s"] = time.perf_counter() - t_geo
    selected, feasible_count = select_candidate(candidates, clearance_m=params.clearance_m)
    diagnostics.feasible_candidates = feasible_count
    if selected is None:
        closest = max(candidates, key=lambda c: c.clearance_m)
        lateral = float(window.lateral_of(closest.worst_pt[None, :])[0])
        return fail(
            FailureCode.PATH_CLEARANCE_VIOLATED,
            f"{n_init} 个初值中找到 {len(candidates)} 条 RRT 候选，但没有一条通过真几何净空"
            f" {params.clearance_m:.1f} m；最接近的 {closest.clearance_m:.2f} m"
            f"（最差点 ({closest.worst_pt[0]:.1f}, {closest.worst_pt[1]:.1f}, "
            f"{closest.worst_pt[2]:.1f})、离中轴 {lateral:.1f} m）——"
            "先查分辨率与净空是否匹配，再怀疑求解器",
        )

    path = selected.result.path_enu
    length = selected.length_m
    clearance = selected.clearance_m
    # ---- 时间域层：几何路径 → 带时间的可行轨迹（最短飞行时间，论文 2.4 节口径）----
    # 解不出来只降级（status 仍 OK、trajectory=None、warnings 说明），因为"几何可飞、时间剖面
    # 没解出来"与"这条航段飞不过去"是两件事——后端据此决定"照路径飞"还是"这条航段先别派"。
    # 真失败只有一种：折线出了半宽、NLP 在半宽内也解不出可交付轨迹（ROI_NO_PATH）。
    t0 = time.perf_counter()
    traj, traj_clearance, failure = fit_time_profile(
        path, field, window, params, need_m=need, prisms=prisms, diagnostics=diagnostics
    )
    timings["trajectory_s"] = time.perf_counter() - t0
    if failure is not None:
        return fail(*failure)

    # 轨迹的真几何净空由 fit_time_profile 量过（含"不达标就抬阈值重解一次"）；无轨迹时退回折线的。
    min_clearance = float(clearance) if traj is None else float(traj_clearance)
    path_lateral_m = float(window.lateral_of(path).max())
    traj_lateral_m: float | None = None
    traj_thrust_n: float | None = None
    duration_s: float | None = None
    max_airspeed_mps: float | None = None
    max_acceleration_mps2: float | None = None
    if traj is not None:
        # 交付物是轨迹，横向偏移与推力都量轨迹的真值。半宽在 NLP 里已是硬约束，这里再量一次
        # 既是闸也是证据；推力由 a 换算（托住重力），否则"有没有咬到上限"只能靠文档嘴说。
        traj_lateral_m = float(window.lateral_of(traj.position_enu).max())
        if traj_lateral_m > window.half_width_m + 1e-6:
            diagnostics.warnings.append(
                f"轨迹离中轴 {traj_lateral_m:.1f} m 超出走廊半宽 {window.half_width_m:.1f} m"
                "（NLP 有该约束，超出即求解器容差量级，先查这个数再判 corridor 失效）"
            )
        g_vec = np.array([0.0, 0.0, defaults.GRAVITY_MPS2])
        traj_thrust_n = float(params.vehicle.mass_kg * float(np.max(np.linalg.norm(
            traj.acceleration_enu + g_vec, axis=1
        ))))
        duration_s = float(traj.t_s[-1])
        max_airspeed_mps = float(np.max(np.linalg.norm(traj.velocity_enu, axis=1)))
        max_acceleration_mps2 = float(np.max(np.linalg.norm(traj.acceleration_enu, axis=1)))
    # 真高/高度一律量"要飞的那条线"：有轨迹取轨迹，降级才退回折线（原来固定取折线，
    # 绕高墙时折线爬到 52 m、轨迹只到 26 m，真高误报 25 m）。
    delivered = path if traj is None else traj.position_enu
    agl = delivered[:, 2] - (terrain.at(delivered) if terrain is not None else 0.0)
    max_lateral_m = path_lateral_m if traj_lateral_m is None else traj_lateral_m
    stats = replace(
        stats,
        points=len(path),
        length_ratio=length / stats.straight_m if stats.straight_m > 0 else 0.0,
        max_lateral_m=max_lateral_m,
        path_max_lateral_m=path_lateral_m,
        inside_corridor=bool(max_lateral_m <= window.half_width_m + 1e-6),
        max_thrust_n=0.0 if traj_thrust_n is None else traj_thrust_n,
        thrust_ratio=(
            0.0 if traj_thrust_n is None
            else traj_thrust_n / params.vehicle.thrust_max_n
        ),
        agl_min_m=float(agl.min()),
        agl_max_m=float(agl.max()),
        z_min_m=float(delivered[:, 2].min()),
        z_max_m=float(delivered[:, 2].max()),
    )
    # rrt_iterations / rrt_nodes 报选中候选自身的规模；总搜索耗时看 timings["rrt_s"]。
    diagnostics.rrt_iterations = int(selected.result.iterations)
    diagnostics.rrt_nodes = int(selected.result.nodes)
    metrics = Metrics(
        length_m=length,                    # 几何路径长度（RRT 折线）——不是轨迹弧长
        min_clearance_m=min_clearance,       # 有轨迹就报轨迹的（shapely 量棱柱），否则报几何路径的
        duration_s=duration_s,
        max_airspeed_mps=max_airspeed_mps,
        max_acceleration_mps2=max_acceleration_mps2,
        # energy_j 留空：没接风场，能耗口径不成立（平均高度这类派生量由后端自己算）
    )
    # 放在最后取，把 stats / metrics 的构造也计入墙钟；分段仍保留做耗时归因。
    timings["total_s"] = time.perf_counter() - started
    diagnostics.elapsed_s = timings["total_s"]
    return RouteSegmentPlan(
        Status.OK, FailureCode.NONE, "", path, metrics, stats, diagnostics, timings, digest,
        trajectory=traj,
    )

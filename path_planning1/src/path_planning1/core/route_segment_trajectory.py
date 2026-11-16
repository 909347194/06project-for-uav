"""时间域层：几何折线 → 带时间的可行轨迹（最短飞行时间），含净空重解与走廊失败判定。

单独成文件是为了让 ``route_segment_planner`` 保持"一条流水线"的形状：入口只说
"几何 → 时间剖面 → 指标"，而这里负责"时间剖面到底解出来了没有、解出来的东西交付得出去吗"。

三件事只有这里知道：

* **净空阈值**进 NLP 时已经含 SDF 量化容差（``need_m`` = 净空 + 竖直格距）；
* **SDF 的乐观误差**（屋顶附近实测 2.2 m > 容差 2 m）靠**一次重解**兜住：真几何不达标就把
  阈值抬 ``_CLEARANCE_REPAIR_M`` 再解一遍；还不过才是真交付不了；
* **走廊半宽**是交付物约束：NLP 里是硬约束，折线（内部初值）不保证；折线出半宽且 NLP 解不出
  可交付轨迹时，返回 ``ROI_NO_PATH`` 而不是"降级成一条没有轨迹的 ok"。

失败返回 ``(None, None, (code, message))``，由入口统一 ``fail()``；这样 ``Diagnostics``
里的 solver 状态只有一份出处，入口不用重复拼装。
"""

from __future__ import annotations

from ..contract import Diagnostics, FailureCode, Trajectory
from .clearance_check import BuildingPrisms
from .corridor_window import CorridorWindow
from .route_segment_params import RouteSegmentParams
from .trajectory_optimization import NlpConfig, optimize_trajectory

# 净空重解抬升量：实测 SDF 乐观误差 2.2 m（4/4/2 栅格、屋顶附近三线性 + 距离场非线性）。
_CLEARANCE_REPAIR_M = 2.0
_EPS_CLEARANCE_M = 1e-9


def fit_time_profile(
    path, field, window: CorridorWindow, params: RouteSegmentParams, *,
    need_m: float, prisms: BuildingPrisms, diagnostics: Diagnostics,
) -> tuple[Trajectory | None, float | None, tuple[FailureCode, str] | None]:
    """解时间剖面；返回 ``(traj, 轨迹真几何净空, 失败)``。"""
    path_lateral_m = float(window.lateral_of(path).max())
    traj = None
    traj_clearance: float | None = None
    nlp = None
    for extra_m in (0.0, _CLEARANCE_REPAIR_M):
        cfg = NlpConfig.from_planner(
            params.planner, params.vehicle, clearance_m=need_m + extra_m,
            # 半宽进 NLP = 走廊从"算过的范围"变成"交付轨迹的约束"（Lin 2021 的 corridor /
            # GeoWind2Plan 的 Ω_task(r)）。只约束轨迹：RRT 折线是内部初值，改它的采样会换随机流。
            half_width_m=window.half_width_m,
            corridor_normal_xy=(float(window.normal[0]), float(window.normal[1])),
            corridor_pivot_xy=window.start_xy,
        )
        nlp = optimize_trajectory(path, field, cfg)
        diagnostics.solver_status = nlp.solver_status
        diagnostics.solver_iterations = nlp.iterations
        traj = nlp.trajectory
        if traj is None:
            diagnostics.warnings.append(f"NLP 未给出轨迹（{nlp.solver_status}）：{nlp.message}")
            break
        traj_clearance, worst = prisms.worst(
            traj.position_enu, params.planner.clearance_sample_step_m
        )
        if traj_clearance >= params.clearance_m - _EPS_CLEARANCE_M:
            if nlp.message:      # 退让过（热启动/降节点/加预算）：能收敛就行，但要留痕
                diagnostics.warnings.append(f"NLP 降级：{nlp.message}")
            break
        if extra_m > 0.0:
            lateral = float(window.lateral_of(worst[None, :])[0])
            return None, None, (
                FailureCode.PATH_CLEARANCE_VIOLATED,
                (
                    f"轨迹层输出的真几何净空 {traj_clearance:.2f} m < 要求 {params.clearance_m:.1f} m"
                    f"（最差点 ({worst[0]:.1f}, {worst[1]:.1f}, {worst[2]:.1f})、离中轴 {lateral:.1f} m）；"
                    f"净空阈值抬到 {need_m + extra_m:.1f} m 重解仍不过——SDF 的乐观超出容差"
                ),
            )
        # 交付净空不达标就抬阈值再解一次；基线第一次就过，不会为此多花一次求解。
        diagnostics.warnings.append(
            f"SDF 乐观：轨迹真几何净空 {traj_clearance:.2f} m < 要求 {params.clearance_m:.1f} m，"
            f"净空阈值 +{_CLEARANCE_REPAIR_M:.1f} m 重解"
        )

    if traj is None and path_lateral_m > window.half_width_m + _EPS_CLEARANCE_M:
        # 折线自己就出了半宽、NLP 在半宽约束下又解不出来 ⇒ 这条走廊交付不出东西。
        # 这不是"NLP 数值没收敛"（那只是降级）：走廊是 route 的语义边界。
        return None, None, (
            FailureCode.ROI_NO_PATH,
            (
                f"走廊半宽 {window.half_width_m:.1f} m 内没有可交付轨迹：折线最远 "
                f"{path_lateral_m:.1f} m（窗口含 margin 实际 {window.lateral_m:.1f} m），"
                f"NLP 在半宽约束下无解（{diagnostics.solver_status}）。半宽要同时容纳净空"
                f"（SDF 用的是 {need_m:.1f} m），先把 ratio 放宽或把 spacing_m 调细；"
                "margin_m 只垫栅格，不放宽交付边界"
            ),
        )
    return traj, traj_clearance, None

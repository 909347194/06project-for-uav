"""一条航段的**结果类型**：统计、扁平指标、状态与失败码。

从 ``route_segment_planner.py`` 里分出来，只为一件事：让"算"（流程）与"报"（结果形状）各有归属。
形状对齐契约 ``PlanResult``——接 ``PlanAlgorithm`` 时不用改下游。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..contract import Diagnostics, FailureCode, Metrics, Status, Trajectory


@dataclass(frozen=True)
class RouteSegmentStats:
    """窗口与几何统计（"这次算了多大地方、走了多远"，不是优化指标）。

    计数用 ``int``、长度用 ``_m``、比例无量纲。以前这些都塞进 ``dict[str, float]``，
    ``points=5.0`` 那种写法会让下游分不清计数与长度。
    """

    horizontal_m: float = 0.0            # 起终点水平距离
    straight_m: float = 0.0              # 起终点三维直线距离
    corridor_width_m: float = 0.0        # 用户口径的走廊宽（2×半宽）
    window_width_m: float = 0.0          # 实际窗口宽（含 margin）
    window_shape: tuple[int, int, int] = (0, 0, 0)
    window_cells: int = 0
    buildings_total: int = 0
    buildings_in_window: int = 0
    occupied_ratio: float = 0.0          # 窗口内占据格占比
    ground_start_m: float = 0.0
    ground_goal_m: float = 0.0
    z_envelope_m: tuple[float, float] = (0.0, 0.0)
    points: int = 0
    length_ratio: float = 0.0            # RRT 折线长度 / 三维直线（折线口径）
    max_lateral_m: float = 0.0           # 交付物（有轨迹就轨迹，否则折线）离中轴最大水平偏移
    path_max_lateral_m: float = 0.0      # RRT 折线离中轴最大水平偏移（只作初值，可以大于半宽）
    inside_corridor: bool = False        # 交付物是否真的落在 half_width 内（NLP 已强约束）
    max_thrust_n: float = 0.0            # 轨迹节点最大所需推力 m‖a+g·ê_z‖（没有轨迹就是 0）
    thrust_ratio: float = 0.0            # 上者 / thrust_max_n；>0.9 说明机型已接近上限
    # 以下四项都是**交付物**口径（有轨迹取轨迹，降级才取折线）：真高 = z − 地面。
    # 别拿 RRT 折线的 z 去校验要飞的轨迹——折线只是 NLP 的初值。
    agl_min_m: float = 0.0
    agl_max_m: float = 0.0
    z_min_m: float = 0.0
    z_max_m: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """扁平字典，键名与历史 ``summary_*.json`` 一致（便于对账与绘图）。"""
        return {
            "horizontal_m": round(self.horizontal_m, 1),
            "straight_m": round(self.straight_m, 1),
            "corridor_width_m": round(self.corridor_width_m, 1),
            "window_width_m": round(self.window_width_m, 1),
            "window_shape": list(self.window_shape),
            "window_shape_cells": self.window_cells,
            "buildings_total": self.buildings_total,
            "buildings_in_window": self.buildings_in_window,
            "occupied_ratio": round(self.occupied_ratio, 4),
            "ground_start_m": round(self.ground_start_m, 1),
            "ground_goal_m": round(self.ground_goal_m, 1),
            "z_envelope_m": [round(self.z_envelope_m[0], 1), round(self.z_envelope_m[1], 1)],
            "points": self.points,
            "length_ratio": round(self.length_ratio, 3),
            "max_lateral_m": round(self.max_lateral_m, 1),
            "path_max_lateral_m": round(self.path_max_lateral_m, 1),
            "inside_corridor": self.inside_corridor,
            "max_thrust_n": round(self.max_thrust_n, 1),
            "thrust_ratio": round(self.thrust_ratio, 3),
            "agl_min_m": round(self.agl_min_m, 1),
            "agl_max_m": round(self.agl_max_m, 1),
            "z_min_m": round(self.z_min_m, 1),
            "z_max_m": round(self.z_max_m, 1),
        }


@dataclass
class RouteSegmentPlan:
    """一条航段的结果。``status`` 为 ``INFEASIBLE`` 时 ``path_enu`` 是 ``None``。

    两个几何产物别混：``path_enu`` 是**几何路径**（RRT 折线，可飞性由净空闸保证），
    ``trajectory`` 是**时空轨迹**（NLP 在走廊内解出的 x/v/a + 均匀 Δt，目标最短飞行时间）。
    轨迹存在时它才是"要飞的那条线"（切线连续 C¹）；几何路径保留为兜底与对照。
    ``metrics.duration_s / max_airspeed_mps / ...`` 只描述轨迹——没有轨迹时保持 ``None``，
    不拿"路径长度 / 某个默认速度"凑一个假时长。
    """

    status: Status
    failure_code: FailureCode = FailureCode.NONE
    message: str = ""
    path_enu: np.ndarray | None = None
    metrics: Metrics = field(default_factory=Metrics)
    stats: RouteSegmentStats = field(default_factory=RouteSegmentStats)
    diagnostics: Diagnostics = field(default_factory=Diagnostics)
    timings: dict[str, float] = field(default_factory=dict)
    params_hash: str = ""
    trajectory: Trajectory | None = None   # 放最后：老调用方（含测试）按位置传参不受影响

    @property
    def ok(self) -> bool:
        return self.status is Status.OK and self.path_enu is not None

    def metrics_dict(self) -> dict[str, object]:
        """扁平指标：契约 ``Metrics`` + ``RouteSegmentStats`` + 口径指纹，CLI / JSON / 绘图统一从这里取。

        ``params_hash`` 必须在这里：后端常常只序列化这份扁平字典，丢掉它就分不清
        startup 场版与 on_demand 现算版两套 SDF 口径，缓存键会串。
        """
        return {
            "status": self.status.value,
            "failure_code": self.failure_code.value,
            "params_hash": self.params_hash,
            "length_m": None if self.metrics.length_m is None else round(self.metrics.length_m, 1),
            "min_clearance_m": (
                None if self.metrics.min_clearance_m is None
                else round(self.metrics.min_clearance_m, 2)
            ),
            "duration_s": self.metrics.duration_s,
            "max_airspeed_mps": self.metrics.max_airspeed_mps,
            "max_acceleration_mps2": self.metrics.max_acceleration_mps2,
            "energy_j": self.metrics.energy_j,
            **self.stats.to_dict(),
        }


def rejected_plan(code: FailureCode, message: str, digest: str = "") -> RouteSegmentPlan:
    """装配层的**准入拒绝**（端点出界 / OD 超契约）。

    形状与规划器的失败结果完全一致：后端按 ``status`` + ``failure_code`` 一条路处理，
    不用为"没进核心规划"的请求写特例。``total_s=0`` 是因为真的没算，不是漏计时。
    """
    return RouteSegmentPlan(
        Status.INFEASIBLE, code, message, None,
        Metrics(), RouteSegmentStats(), Diagnostics(params_hash=digest),
        {"total_s": 0.0}, digest,
    )

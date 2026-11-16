"""算法参数（进 ``params_hash``，决定缓存能否复用）。

**这里只有字段和默认值，数值本身来自** ``contract/defaults.py``（唯一配置清单）。
三条边界，别越界：净空 → ``vehicle.VehicleLimits``；飞行带 → ``defaults``；
栅格间距 → ``core/route_segment_planner.RouteSegmentParams``（请求级口径，与算法无关）。

所有旋钮都从 ``PlannerParams`` 走：想临时改口径就 ``replace(PlannerParams(), ...)``，
改默认值就改 ``defaults.py``——两条路都只经过一个地方。
"""

from __future__ import annotations

from dataclasses import dataclass

from . import defaults


@dataclass(frozen=True)
class PlannerParams:
    """本包的算法参数。改任何一项都必须反映到 ``params_hash``。

    换算表与取舍理由见 ``path_planning1/README.md``「口径」。
    """

    # ---- RRT 可调旋钮（在用）----------------------------------------
    rrt_step_m: float = defaults.RRT_STEP_M
    rrt_min_step_m: float = defaults.RRT_MIN_STEP_M
    rrt_sdf_step_scale: float = defaults.RRT_SDF_STEP_SCALE
    rrt_max_iters: int = defaults.RRT_MAX_ITERS
    rrt_goal_sample_rate: float = defaults.RRT_GOAL_SAMPLE_RATE

    # ---- RRT 实现细节（也在用，改了输出会变，所以一样可配、一样进 hash）----
    rrt_shortcut_iters: int = defaults.RRT_SHORTCUT_ITERS    # 捷径化的尝试次数
    rrt_initializations: int = defaults.RRT_INITIALIZATIONS  # 多初值次数（seed+i）
    # 沿线检查步长：RRT 判段内净空用它，交付前的真几何复核也用它——**同一个旋钮**，
    # 不做"检查用 2 m、复核用 1 m"的两套步长（那样会自相矛盾：求解器自认为合格、复核判不合格）
    clearance_sample_step_m: float = defaults.CLEARANCE_SAMPLE_STEP_M

    # ---- 轨迹层 NLP（在用：节点数 / 时间步 / IPOPT 预算）----------------
    n_resample: int = defaults.NLP_N_RESAMPLE
    initial_time_step_s: float = defaults.NLP_INITIAL_TIME_STEP_S
    segment_samples: int = defaults.NLP_SEGMENT_SAMPLES
    dt_min_s: float = defaults.NLP_DT_MIN_S
    dt_max_s: float = defaults.NLP_DT_MAX_S
    ipopt_max_iter: int = defaults.NLP_IPOPT_MAX_ITER

    def __post_init__(self) -> None:
        """挡住绕过 JSON 解析、直接构造参数时的越界多初值。"""
        n = int(self.rrt_initializations)
        if not 1 <= n <= defaults.RRT_INITIALIZATIONS_MAX:
            raise ValueError(
                "rrt_initializations 必须在 "
                f"1–{defaults.RRT_INITIALIZATIONS_MAX}；过大会线性放大 RRT 时延"
            )

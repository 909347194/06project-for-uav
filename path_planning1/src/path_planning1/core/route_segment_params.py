"""一次航段的**请求口径**：走廊、栅格间距、真高带、seed（算法/车辆各用契约）。

从 ``route_segment_planner.py`` 分出来，只为一件事：那里是"流程"，这里是"输入"，职责不混。
默认数值只在 ``contract/defaults.py`` 一份；这里只做引用与派生，不第二次写数字。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..contract import (
    PlannerParams,
    RegionOfInterest,
    VehicleLimits,
    defaults,
)
from .rrt_path_search import RRTConfig


@dataclass(frozen=True)
class RouteSegmentParams:
    """默认值即本轮拍板的那套：走廊 0.2、净空 5 m、真高 10–120 m。

    净空/步长/迭代**不在这里**：净空由 ``vehicle``、步长与迭代由 ``planner`` 决定；
    这里只提供"默认车辆 + 默认算法"的现成组合。改默认数值请改 ``contract/defaults.py``，
    改默认组合关系才动那两个契约——两件事别混。
    """

    roi: RegionOfInterest = field(
        default_factory=lambda: RegionOfInterest(
            width_ratio=defaults.DEFAULT_ROI_WIDTH_RATIO,
            margin_m=defaults.DEFAULT_ROI_MARGIN_M,
        )
    )
    spacing_m: tuple[float, float, float] = defaults.DEFAULT_GRID_SPACING_M
    agl_min_m: float = defaults.DEFAULT_ALT_MIN_M
    agl_max_m: float = defaults.DEFAULT_ALT_MAX_M
    planner: PlannerParams = field(default_factory=PlannerParams)
    vehicle: VehicleLimits = field(default_factory=VehicleLimits)
    seed: int = defaults.DEFAULT_SEED   # 请求级：同算法不同 seed 是不同请求

    @property
    def clearance_m(self) -> float:
        """净空只来自 ``VehicleLimits``（这里只是转发，不是第二个来源）。"""
        return float(self.vehicle.clearance_m)

    @property
    def sdf_tolerance_m(self) -> float:
        """SDF 量化容差 = **竖直格距** ``sz``，不是随手给的余量。

        SDF 量的是"到最近占据体素**中心**"的距离：屋顶若落在两个格心之间（最多差 ``sz``），
        站在屋顶上方 ``h`` 米处的读数就会比真几何乐观至多 ``sz``。栅格化已按半格对角线把
        建筑外向膨胀（见 ``core/occupancy_sdf.py``），xy 方向的误差因此变成"保守"而不是"乐观"，
        剩下唯一还会乐观的就是这个竖直量化 —— 所以容差只取 ``sz``（实测 4/4/2 下 SDF 乐观误差
        可达 2.2 m，与 ``sz=2`` 同量级；因此交付前一律再用真几何复核一次）。
        注意它不是"交付净空的余量"：交付数字由 ``core/clearance_check.py`` 真几何量，两回事。
        """
        return float(self.spacing_m[2])

    def rrt_config(self) -> RRTConfig:
        """算法参数 + **含容差**的净空阈值 → core 内部的 ``RRTConfig``（派生，不独立设默认）。

        阈值 = 车辆净空 + ``sdf_tolerance_m``：RRT 只认识 SDF，按裸净空判会放进真几何不达标的
        路径。交付出去的数字不看这里——``plan_route_segment`` 末尾用真几何复核，报的是真实净空。
        """
        return RRTConfig.from_planner(
            self.planner, clearance_m=self.clearance_m + self.sdf_tolerance_m, seed=self.seed
        )

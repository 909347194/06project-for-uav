"""世界快照与感兴趣区域（本包的两层 ROI 语义都定义在这里）。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import defaults
from .geometry_types import Building, NoFlyZone
from .local_frame import GridSpec, LocalFrame

#: 默认飞行带（真高 AGL）。数值定义在 ``contract/defaults.py``（唯一配置清单），
#: 这里只做**引用**，供 ``WorldSnapshot`` 与 ``core/route_segment_planner.RouteSegmentParams`` 共用同一个数。
DEFAULT_ALT_MIN_M = defaults.DEFAULT_ALT_MIN_M
DEFAULT_ALT_MAX_M = defaults.DEFAULT_ALT_MAX_M


@dataclass
class WorldSnapshot:
    """**预留契约**：一个城市快照，按 ``world_id`` 缓存（当前实现不消费）。

    实际的世界由 ``core/city_dataset.py::load_city`` 装载成 ``CityDataset``，全域场由
    ``core/world_field.py::WorldField`` 表示；两者都不用这个类。后端不要按它拼世界。
    下面关于 ``sdf`` 的说明也是早期"窗口重建 SDF"时期的口径，当前启动阶段建的是全域场。

    ``sdf`` 可预计算并缓存（形状必须等于 ``grid.shape``，自由空间为正、建筑内为负）；
    为 ``None`` 时由 ``core/occupancy_sdf.py`` 从 ``buildings`` 栅格化生成。

    SDF 口径：单位是**米**，正值 = 自由空间。本实现**只在走廊窗口内按 OD 重建** SDF
    （见 ``core/route_segment_planner.py``），不是全城算完再裁带；窗口 z 包络由
    ``z_envelope_from_ground`` 定，窗口外的建筑不参与净空。

    ``nofly`` 是扩展槽：本阶段引擎不消费，规划只考虑建筑。
    """

    world_id: str
    frame: LocalFrame
    buildings: list[Building]
    grid: GridSpec
    alt_min_m: float = DEFAULT_ALT_MIN_M
    alt_max_m: float = DEFAULT_ALT_MAX_M
    sdf: np.ndarray | None = None
    nofly: list[NoFlyZone] = field(default_factory=list)

    @property
    def n_buildings(self) -> int:
        return len(self.buildings)


@dataclass(frozen=True)
class RegionOfInterest:
    """当前感兴趣区域：以水平起终点线段为中心轴的走廊。

    走廊在这里是**计算域**，也因此是**解域**（两者不再是两件事）：

    * 栅格沿走廊轴向建（``GridSpec.yaw_rad``），格数 ∝ 走廊带面积，不是 ENU 外接矩形；
    * 带外没有算过 SDF，所以带外一律不可用。窄走廊会把"可解"变"无解"，
      由 ``RRT_NO_PATH`` / ``ROI_NO_PATH`` 如实报出，不做静默放行。

    原先的两个开关 ``clip_grid`` / ``hard_constraint`` 没有任何消费者（关掉也照旧裁窗口、
    打开也不改变输出），已删除——不留"看起来能配、实际不生效"的字段。

    半宽口径：``half_width_for`` 用**水平**距离；论文实现用的是三维距离
    （``d_max = 0.5·alpha·‖goal−start‖₃D``），高度差大时二者会差几个百分点。
    """

    width_ratio: float | None = None     # 走廊总宽 = ratio · 水平起终点距离
    half_width_m: float | None = None    # 或直接给半宽（优先于 ratio）
    margin_m: float = 0.0                # 走廊外扩余量（横向、纵向各加一圈）

    def half_width_for(self, horizontal_distance_m: float) -> float:
        """解析半宽（米）。两者都给时以 ``half_width_m`` 为准。"""
        if self.half_width_m is not None:
            return float(self.half_width_m)
        if self.width_ratio is not None:
            return 0.5 * float(self.width_ratio) * float(horizontal_distance_m)
        return 0.0

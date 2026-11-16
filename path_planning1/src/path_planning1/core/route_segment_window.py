"""一条航段的窗口装配：ROI 走廊 → 正式窗口 + 地形 + z 包络。

窗口要建两次：先按"绝对飞行带"量一遍地面，拿到窗口内最低地面才能定绝对 z 包络
（天花板按最低地面算，见 ``corridor_window.z_envelope_from_ground``），再按 z 包络
建正式窗口。两次的 x/y 覆盖完全相同（z 不影响水平尺寸），probe 只用于取地面高差。
"""

from __future__ import annotations

from dataclasses import replace

from ..contract import LocalFrame
from .corridor_window import CorridorWindow, corridor_window, z_envelope_from_ground
from .dem_terrain import Dem, TerrainField, terrain_field
from .route_segment_params import RouteSegmentParams


def build_segment_window(
    dem: Dem | None,
    frame: LocalFrame,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    params: RouteSegmentParams,
) -> tuple[CorridorWindow, TerrainField | None, float, float]:
    """返回 ``(window, terrain, z_lo, z_hi)``；``dem=None`` 时按平地、地面高 0。"""
    mid_z = 0.5 * (params.agl_min_m + params.agl_max_m)
    probe = corridor_window(
        (start_xy[0], start_xy[1], mid_z), (goal_xy[0], goal_xy[1], mid_z),
        params.roi, params.spacing_m, params.agl_min_m, params.agl_max_m,
    )
    probe_terrain = terrain_field(dem, frame, probe.grid) if dem is not None else None
    ground_min = 0.0 if probe_terrain is None else probe_terrain.z_range[0]
    z_lo, z_hi = z_envelope_from_ground(ground_min, params.agl_min_m, params.agl_max_m)
    window = corridor_window(
        (start_xy[0], start_xy[1], mid_z), (goal_xy[0], goal_xy[1], mid_z),
        params.roi, params.spacing_m, z_lo, z_hi,
    )
    # 两窗口 x/y 格心完全相同，地面表只依赖 x/y，再算一遍必逐元素相等：复用数组，
    # 只把 grid 换成正式窗口的，让 at / at_xy / band_ok 的索引基准仍是交付用的 grid。
    terrain = None if probe_terrain is None else replace(probe_terrain, grid=window.grid)
    return window, terrain, z_lo, z_hi

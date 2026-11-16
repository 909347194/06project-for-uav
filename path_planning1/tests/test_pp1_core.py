"""pp1 core 的回归测试：SDF 符号、走廊窗口几何、RRT 不穿楼。

这些断言存在的理由很具体：``build_sdf`` 曾把两遍 EDT 的自由/内部量接反，
结果 SDF 恒为 0、路径笔直穿楼而指标全绿。签名断言是最便宜的那道防线。
"""

from __future__ import annotations

import numpy as np
from path_planning1.contract import (
    Building,
    GridSpec,
    LocalFrame,
    PlannerParams,
    RegionOfInterest,
)
from path_planning1.core.city_loader import filter_by_bbox
from path_planning1.core.clearance_check import BuildingPrisms
from path_planning1.core.corridor_window import corridor_window
from path_planning1.core.dem_terrain import Dem, attach_base_z, terrain_field
from path_planning1.core.occupancy_sdf import SdfField, build_occupancy, build_sdf
from path_planning1.core.rrt_path_search import RRT3D, RRTConfig

SPACING = (2.0, 2.0, 2.0)


def _box(x0: float, y0: float, x1: float, y1: float, height: float) -> Building:
    return Building(
        footprint=((x0, y0), (x1, y0), (x1, y1), (x0, y1)),
        height_m=height,
    )


def _grid(shape: tuple[int, int, int] = (40, 40, 20)) -> GridSpec:
    return GridSpec(origin_enu=(0.0, 0.0, 0.0), spacing_m=SPACING, shape=shape)


# ---------- occupancy ----------


def test_occupancy_fills_prism_only() -> None:
    grid = _grid()
    occ = build_occupancy([_box(6.0, 6.0, 18.0, 18.0, 10.0)], grid)
    assert occ[5, 5, 2]        # 楼内：x=10,y=10,z=4
    assert not occ[5, 5, 6]    # 楼顶以上：z=12 > 10
    assert not occ[0, 5, 2]    # 楼外


# ---------- SDF ----------


def test_sdf_is_not_degenerate() -> None:
    """回归：整张 SDF 恒为 0 说明两遍 EDT 接反了。"""
    grid = _grid()
    occ = build_occupancy([_box(6.0, 6.0, 18.0, 18.0, 10.0)], grid)
    sdf = build_sdf(occ, grid.spacing_m)
    assert float(sdf.min()) < 0.0 < float(sdf.max())


def test_sdf_sign_matches_occupancy() -> None:
    grid = _grid()
    occ = build_occupancy([_box(6.0, 6.0, 18.0, 18.0, 10.0)], grid)
    sdf = build_sdf(occ, grid.spacing_m)
    assert np.array_equal(sdf < 0.0, occ)


def test_sdf_magnitude_is_metric_distance() -> None:
    """自由区 |SDF| 以米为单位：网格下标差 10 × 2 m/格 = 20 m。"""
    grid = _grid()
    occ = np.zeros(grid.shape, dtype=bool)
    occ[10, 10, 10] = True
    sdf = build_sdf(occ, grid.spacing_m)
    assert abs(float(sdf[20, 10, 10]) - 10 * SPACING[0]) < 0.5
    assert abs(float(sdf[10, 10, 19]) - 9 * SPACING[2]) < 0.5


def test_segment_min_detects_wall() -> None:
    grid = _grid()
    field = SdfField(
        grid, build_sdf(build_occupancy([_box(6.0, 6.0, 18.0, 18.0, 10.0)], grid), SPACING)
    )
    assert field.segment_min(np.array([2.0, 10.0, 4.0]), np.array([22.0, 10.0, 4.0]), 1.0) < 0.0
    assert field.segment_min(np.array([2.0, 10.0, 14.0]), np.array([22.0, 10.0, 14.0]), 1.0) > 0.0


# ---------- 走廊窗口 ----------


def test_corridor_window_geometry() -> None:
    win = corridor_window(
        (0.0, 0.0, 65.0), (100.0, 0.0, 65.0), RegionOfInterest(width_ratio=0.2, margin_m=5.0),
        SPACING, 10.0, 120.0,
    )
    assert win.horizontal_length_m == 100.0
    assert win.half_width_m == 10.0
    assert win.lateral_m == 15.0
    assert win.inside_grid(np.array([[0.0, 0.0, 65.0], [100.0, 0.0, 65.0]])).all()
    nx, ny, nz = win.grid.shape
    assert (nx - 1) * SPACING[0] >= 100.0
    assert (ny - 1) * SPACING[1] >= 30.0   # 2 × lateral
    assert nz == 56                       # (120 - 10) / 2 + 1


def test_margin_zero_keeps_start_inside() -> None:
    """margin=0 时起点正好落在格 0 上：斜栅格往返误差（~1e-13 格）不许把它判出窗外。"""
    win = corridor_window(
        (0.0, 0.0, 65.0), (100.0, 30.0, 65.0),   # 斜向，才能暴露旋转取整误差
        RegionOfInterest(width_ratio=0.2, margin_m=0.0), SPACING, 10.0, 120.0,
    )
    idx = win.grid.enu_to_index(np.array([[0.0, 0.0, 120.0]]))
    assert idx[0, 0] <= 0.0                 # 索引确实落在下边界上（不是 1e-13 之外）
    assert win.inside_grid(np.array([[0.0, 0.0, 120.0]])).all()


# ---------- RRT ----------


def _wall_world(alt_max_m: float) -> tuple[SdfField, object]:
    grid = _grid((60, 60, 30))
    buildings = [_box(54.0, 0.0, 62.0, 118.0, 30.0)]
    win = corridor_window(
        (10.0, 58.0, 20.0), (110.0, 58.0, 20.0),
        RegionOfInterest(width_ratio=0.2, margin_m=20.0), SPACING, 10.0, alt_max_m,
    )
    field = SdfField(grid, build_sdf(build_occupancy(buildings, grid), SPACING))
    return field, win


def test_rrt_climbs_over_wall_without_crossing() -> None:
    field, win = _wall_world(58.0)
    cfg = RRTConfig.from_planner(
        PlannerParams(rrt_step_m=20.0, rrt_min_step_m=4.0, rrt_max_iters=4000),
        clearance_m=5.0, seed=1,
    )
    result = RRT3D(field, win, (10.0, 58.0), cfg).plan(
        np.array([10.0, 58.0, 20.0]), np.array([110.0, 58.0, 20.0])
    )
    assert result is not None
    path = result.path_enu
    assert len(path) > 2
    assert float(path[:, 2].max()) > 30.0          # 必须爬过 30 m 的墙
    for i in range(len(path) - 1):
        assert field.segment_min(path[i], path[i + 1], 1.0) >= cfg.clearance_m


def test_rrt_returns_none_when_wall_is_unclimbable() -> None:
    field, win = _wall_world(25.0)                 # 飞行带顶 25 m < 墙顶 30 m
    cfg = RRTConfig.from_planner(
        PlannerParams(rrt_step_m=20.0, rrt_min_step_m=4.0, rrt_max_iters=2000),
        clearance_m=5.0, seed=2,
    )
    assert (
        RRT3D(field, win, (10.0, 25.0), cfg).plan(
            np.array([10.0, 58.0, 20.0]), np.array([110.0, 58.0, 20.0])
        )
        is None
    )


def test_rrt_plan_many_is_single_seed_when_initializations_is_one() -> None:
    """多初值的 seed+i 必须与单次 plan 的 seed 对齐；否则第一条也不是旧基线。"""
    field, win = _wall_world(58.0)
    cfg = RRTConfig.from_planner(
        PlannerParams(rrt_step_m=20.0, rrt_min_step_m=4.0, rrt_max_iters=4000),
        clearance_m=5.0, seed=1,
    )
    start = np.array([10.0, 58.0, 20.0])
    goal = np.array([110.0, 58.0, 20.0])
    single = RRT3D(field, win, (10.0, 58.0), cfg).plan(start, goal)
    batch = RRT3D(field, win, (10.0, 58.0), cfg).plan_many(
        start, goal, initializations=1
    )
    assert single is not None and len(batch) == 1
    assert np.allclose(single.path_enu, batch[0].path_enu)



def test_small_building_is_not_lost_by_cell_center_test() -> None:
    """1.2 m 的小房子夹在格心之间：靠"膨胀半格对角线"，它必须有体素被标记。

    这是净空口径的地基（旧口径下这种房子整栋消失，SDF 会报一个根本没房子的"自由空间"）。
    """
    grid = _grid()                              # 2 m 栅格，格心在偶数坐标
    b = [_box(4.4, 4.4, 5.6, 5.6, 10.0)]        # 整栋夹在 x=4 与 x=6 的格心之间
    occ = build_occupancy(b, grid)
    assert bool(occ[:, :, 0].any())             # 膨胀后至少一个格心落进膨胀多边形


# ---------- 净空（真几何）----------


def test_clearance_filter_is_padded() -> None:
    """窗外贴边的楼必须算进净空：粗筛不外扩会把"3 m 外的楼"漏掉、净空虚报成 30 m。

    路径可以贴着窗口边界走（实测横向偏移最大 109 m，半宽 142 m），所以净空度量的粗筛
    必须比窗口外接矩形再外扩一个 ``CLEARANCE_SEARCH_RADIUS_M``。
    """
    outside = _box(105.0, 0.0, 110.0, 10.0, 12.0)     # 整栋落在窗口 bbox（x≤104）之外
    bounds = (0.0, 0.0, 104.0, 10.0)
    assert filter_by_bbox([outside], bounds) == []
    assert filter_by_bbox([outside], bounds, pad_m=30.0) == [outside]
    # 路径贴着 bbox 边界走（x=104，离楼 1 m）：外扩筛进来的楼必须真的被量到
    prisms = BuildingPrisms([outside])
    assert prisms.min_clearance(np.array([[104.0, 5.0, 6.0]]), 1.0) == 1.0


# ---------- DEM / 真高 ----------


def _flat_dem(z_m: float = 10.0, shape: tuple[int, int] = (20, 20)) -> Dem:
    """造一张平 DEM：1e-4 度/格 ≈ 11 m，够覆盖测试用的小窗口。"""
    return Dem(
        z_m=np.full(shape, z_m, dtype=np.float32),
        lon_ul=113.3, lat_ul=23.12, dlon=1e-4, dlat=1e-4,
    )


def test_dem_blocks_ground_as_solid() -> None:
    grid, dem = _grid(), _flat_dem(10.0)
    frame = LocalFrame(lon0=113.3, lat0=23.12)
    terrain = terrain_field(dem, frame, grid)
    assert float(terrain.ground_m.min()) == 10.0
    occ = build_occupancy([], grid, terrain)
    assert bool(occ[0, 0, 0])           # z=0 在地面 10 m 以下 → 占据
    assert not bool(occ[0, 0, 6])       # z=12 在地面以上 → 自由


def test_agl_band_bounds() -> None:
    grid, dem = _grid(), _flat_dem(10.0)
    terrain = terrain_field(dem, LocalFrame(lon0=113.3, lat0=23.12), grid)
    pts = np.array([[10.0, 10.0, 19.0], [10.0, 10.0, 20.0], [10.0, 10.0, 130.0]])
    ok = terrain.band_ok(pts, 10.0, 120.0)      # 地面 10 → 允许 z ∈ [20, 130]
    assert list(ok) == [False, True, True]


def test_attach_base_z_lifts_buildings() -> None:
    dem = Dem(
        z_m=np.tile(np.linspace(0.0, 19.0, 20, dtype=np.float32), (20, 1)),
        lon_ul=113.3, lat_ul=23.12, dlon=1e-4, dlat=1e-4,
    )
    out = attach_base_z([_box(0.0, 0.0, 4.0, 4.0, 10.0)], dem, LocalFrame(lon0=113.3, lat0=23.12))
    assert out[0].base_z_m > 0.0
    assert out[0].z_top_m == out[0].base_z_m + out[0].height_m


def test_rrt_respects_agl_band() -> None:
    """地面抬到 10 m：墙顶 40 m，真高上限 45 m 刚好能过，压到 30 m 就过不去。"""
    grid = _grid((60, 60, 30))
    dem = _flat_dem(10.0, (60, 60))
    frame = LocalFrame(lon0=113.3, lat0=23.12)
    terrain = terrain_field(dem, frame, grid)
    buildings = [_box(54.0, 0.0, 62.0, 118.0, 40.0)]
    win = corridor_window(
        (10.0, 58.0, 30.0), (110.0, 58.0, 30.0),
        RegionOfInterest(width_ratio=0.2, margin_m=20.0), SPACING, 20.0, 55.0,
    )
    terrain = terrain_field(dem, frame, win.grid)     # 地面表必须和窗口栅格同形
    field = SdfField(win.grid, build_sdf(build_occupancy(buildings, win.grid, terrain), SPACING))
    start, goal = np.array([10.0, 58.0, 20.0]), np.array([110.0, 58.0, 20.0])

    cfg = RRTConfig.from_planner(
        PlannerParams(rrt_max_iters=4000), clearance_m=5.0, seed=3
    )
    tight = RRT3D(field, win, (20.0, 55.0), cfg, terrain=terrain, agl_range_m=(10.0, 30.0))
    assert tight.plan(start, goal) is None      # 真高 ≤30 → z ≤40，撞墙

    roomy = RRT3D(field, win, (20.0, 55.0), cfg, terrain=terrain, agl_range_m=(10.0, 45.0))
    res = roomy.plan(start, goal)
    assert res is not None
    agl = res.path_enu[:, 2] - 10.0
    assert agl.min() >= 10.0 - 1e-6 and agl.max() <= 45.0 + 1e-6

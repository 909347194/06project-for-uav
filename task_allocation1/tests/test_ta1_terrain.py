"""ta1 自带地形适配：DEM、建筑包络和垂直切面联动。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from task_allocation1.terrain import (
    DEM,
    BuildingLayer,
    LocalFrame,
    TerrainData,
    load_buildings,
    load_world_terrain,
)

import task_allocation1 as ta1

ROOT = Path(__file__).resolve().parents[2]


def _synthetic_terrain() -> TerrainData:
    dem = DEM.from_array(np.full((4, 4), 10.0), (-100, -100, 100, 100))
    building = BuildingLayer.from_polygons(
        [np.array([[-10, -10], [10, -10], [10, 10], [-10, 10]], dtype=float)],
        [20.0],
    )
    return TerrainData(LocalFrame(0, 0, 0), dem, building)


def test_terrain_samplers_return_absolute_z() -> None:
    terrain = _synthetic_terrain()
    ground, obstacle = terrain.samplers()
    assert float(ground([0.0, 50.0], [0.0, 50.0])[0]) == 10.0
    assert float(obstacle([0.0, 50.0], [0.0, 50.0])[0]) == 30.0
    assert float(obstacle([50.0], [50.0])[0]) == pytest.approx(10.0)


def test_vertical_section_uses_native_terrain() -> None:
    terrain = _synthetic_terrain()
    ground, obstacle = terrain.samplers()
    cost = ta1.VerticalSectionCost(
        ta1.CostConfig(num_samples=20), ground_z=ground, obstacle_top_z=obstacle
    )
    a, b = (0.0, -100.0, 30.0), (0.0, 100.0, 30.0)
    assert cost.estimate(a, b) > ta1.euclidean3d(a, b)


def test_load_world_terrain_lhasa() -> None:
    root = ROOT / "data" / "worlds" / "lasa_chengguan"
    if not root.exists():
        return  # 数据未随仓库分发时跳过，不把测试绑在本地数据上
    terrain = load_world_terrain(root)
    assert terrain.buildings.count > 0
    assert terrain.dem is not None
    xs = np.array([0.0, 100.0])
    ys = np.array([0.0, 100.0])
    assert np.all(np.isfinite(terrain.ground_z(xs, ys)))


def test_shapefile_height_field(tmp_path: Path) -> None:
    shapefile = pytest.importorskip("shapefile", reason="pyshp 是 [terrain] 可选依赖")
    writer = shapefile.Writer(str(tmp_path / "buildings.shp"))
    writer.field("Height", "N", 10, 2)
    writer.poly(
        [[[-0.001, -0.001], [0.001, -0.001], [0.001, 0.001],
          [-0.001, 0.001], [-0.001, -0.001]]]
    )
    writer.record(20.0)
    writer.close()

    layer = load_buildings(tmp_path, LocalFrame(0, 0, 0))
    assert layer.count == 1
    assert layer.heights_at([0.0, 500.0], [0.0, 500.0]) == pytest.approx([20.0, 0.0])

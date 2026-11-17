"""地形适配层：世界目录 → ta1 垂直切面采样器（不依赖 pp1，随包安装）。"""

from __future__ import annotations

from .buildings import BuildingLayer
from .dem import DEM, load_geotiff_dem
from .frame import LocalFrame
from .world import TerrainData, frame_from_region, load_buildings, load_world_terrain

__all__ = [
    "DEM",
    "BuildingLayer",
    "LocalFrame",
    "TerrainData",
    "frame_from_region",
    "load_buildings",
    "load_geotiff_dem",
    "load_world_terrain",
]

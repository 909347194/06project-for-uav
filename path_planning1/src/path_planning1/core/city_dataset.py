"""城市数据集：把"加载顺序"固化成一个入口，别让调用方自己拼。

顺序是**有语义的**，错了会静默出错而不是报错::

    frame_from_geojson  →  load_buildings  →  load_dem  →  attach_base_z
    (确定原点/尺度)        (多边形→ENU)        (地面)       (逐栋抬 base_z)
                       ↘  load_nofly
                          (geojson→ENU 禁飞棱柱；AGL 输入在 load_nofly 内换算，输出绝对 z)

漏掉 ``attach_base_z`` 时所有楼的 ``base_z_m`` 都是 0，楼体顶面 = 高度值本身；
在 DEM 起伏 5–75 m 的城市里，这等于把一片楼埋进地里、另一片楼飘在空中，
而栅格看起来"很正常"。所以这里不再暴露"只加载不接地形"的默认路径。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..contract import Building, LocalFrame, NoFlyZone
from ..contract.defaults import (
    DEFAULT_NOFLY_ALTITUDE_MODE,
    DEFAULT_NOFLY_PATH,
    MIN_BUILDING_AREA_M2,
    NOFLY_ALTITUDE_AGL,
    TERRAIN_MIN_Z_M,
)
from .city_loader import (
    frame_from_geojson,
    load_buildings,
    load_nofly,
    region_bounds_enu,
)
from .dem_terrain import Dem, attach_base_z, load_dem
from .nofly_altitude import validate_altitude_mode


@dataclass(frozen=True)
class CityDataset:
    """一个城市的静态数据（帧、建筑、DEM）。

    ``terrain_applied`` 为 False 表示**没有接 DEM**（``dem is None``），此时
    ``base_z_m`` 全是 0，飞行高度就是绝对高度——用它做出来的数字不能和 DEM 版
    的结果混着比（平地与接地形是两份世界，``world_id`` 必须区分）。
    """

    name: str
    root: Path
    frame: LocalFrame
    buildings: tuple[Building, ...]
    dem: Dem | None
    terrain_applied: bool
    bounds_m: tuple[float, float, float, float]
    nofly: tuple[NoFlyZone, ...] = ()

    @property
    def n_buildings(self) -> int:
        return len(self.buildings)

    @property
    def n_nofly(self) -> int:
        return len(self.nofly)

    def bbox_m(self) -> tuple[float, float, float, float]:
        """全部建筑 footprints 的包络 ``(x0, y0, x1, y1)``（米）。"""
        x0 = min(p[0] for b in self.buildings for p in b.footprint)
        x1 = max(p[0] for b in self.buildings for p in b.footprint)
        y0 = min(p[1] for b in self.buildings for p in b.footprint)
        y1 = max(p[1] for b in self.buildings for p in b.footprint)
        return x0, y0, x1, y1


def load_city(
    root: str | Path,
    *,
    with_dem: bool = True,
    height_field: str = "Height",
    min_area_m2: float = MIN_BUILDING_AREA_M2,
    terrain_min_z_m: float = TERRAIN_MIN_Z_M,
    nofly_path: str | Path | None = None,
    nofly_altitude_mode: str = DEFAULT_NOFLY_ALTITUDE_MODE,
) -> CityDataset:
    """读一个数据目录（``region.geojson`` + ``building/*.shp|*.geojson`` + ``dem.tif`` + 可选禁飞区）。

    缺文件时**明确报**，不退回默认值：``region.geojson`` 与建筑是必需的；
    建筑支持 shapefile（DBF 为 GBK）与 GeoJSON（UTF-8）两种格式，可同时存在；
    目录里一栋可用建筑都没有也直接报错，不静默建出"楼不挡路"的空世界。
    ``with_dem=True`` 时 ``dem.tif`` 也必需（要平地请显式 ``with_dem=False``）。
    禁飞区默认读数据目录下的 ``nofly.geojson``，**没有该文件就是没有禁飞区**；
    显式给了 ``nofly_path`` 却找不到文件则报错——拼错路径不能静默变成"不设防"。
    ``nofly_altitude_mode`` 只决定**输入字段口径**：``absolute`` 读 ``z_min_m/z_max_m``，
    ``agl`` 读 ``agl_min_m/agl_max_m`` 并由 DEM 换算（此时 ``with_dem`` 必须为 true）；
    ``CityDataset.nofly`` 永远存绝对 z。

    ``height_field`` / ``min_area_m2`` / ``terrain_min_z_m`` 是**数据集口径**：
    它们决定"世界长什么样"，改了等于换了世界，缓存键要用 ``world_id`` 区分。
    """
    root = Path(root)
    nofly_altitude_mode = validate_altitude_mode(nofly_altitude_mode)
    if nofly_altitude_mode == NOFLY_ALTITUDE_AGL and not with_dem:
        raise ValueError("nofly_altitude_mode='agl' 需要 DEM：with_dem 必须为 true（AGL 依赖地面高程）")
    region = root / "region.geojson"
    building_dir = root / "building"
    if not region.exists():
        raise FileNotFoundError(f"缺少区域轮廓：{region}")
    if not building_dir.is_dir():
        raise FileNotFoundError(f"缺少建筑目录：{building_dir}")

    frame = frame_from_geojson(region)
    buildings = load_buildings(
        building_dir, frame, height_field=height_field, min_area_m2=min_area_m2
    )

    dem: Dem | None = None
    if with_dem:
        dem_path = root / "dem.tif"
        if not dem_path.exists():
            raise FileNotFoundError(f"with_dem=True 但缺少 {dem_path}（要平地请显式 with_dem=False）")
        dem = load_dem(dem_path)
        buildings = attach_base_z(buildings, dem, frame, min_z_m=terrain_min_z_m)

    nofly_file = Path(nofly_path) if nofly_path is not None else Path(DEFAULT_NOFLY_PATH)
    if not nofly_file.is_absolute():
        nofly_file = root / nofly_file
    nofly: tuple[NoFlyZone, ...] = ()
    if nofly_file.is_file():
        nofly = load_nofly(
            nofly_file,
            frame,
            dem=dem,
            altitude_mode=nofly_altitude_mode,
        )
    elif nofly_path is not None:
        raise FileNotFoundError(f"world.nofly_path 指定的禁飞区文件不存在：{nofly_file}")

    return CityDataset(
        name=root.name,
        root=root,
        frame=frame,
        buildings=tuple(buildings),
        dem=dem,
        terrain_applied=dem is not None,
        bounds_m=region_bounds_enu(region, frame),
        nofly=nofly,
    )

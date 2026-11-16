"""几何实体：建筑（多边形底面 + 竖向区间）、静态禁飞棱柱。

广州数据是**多边形底面 + Height 字段**，不是论文的轴对齐体素，
所以保留真实轮廓；栅格化与距离场由 ``core/occupancy_sdf.py`` 负责。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Building:
    """建筑：底面多边形（局地 ENU 米制）+ 竖向区间 ``[base_z_m, z_top_m]``。

    ``footprint`` 为首尾不重复的顶点序列，逆时针，长度 ≥ 3。
    ``base_z_m`` 由 DEM 逐栋抬升（``core/dem_terrain.py::attach_base_z``），
    未接地形时保持 0.0——不要把"没接 DEM"和"地面是 0 m"混为一事。
    """

    footprint: tuple[tuple[float, float], ...]
    height_m: float
    base_z_m: float = 0.0
    source: str = ""          # 来源 shp 文件名（不含扩展名），不是建筑类别

    @property
    def z_top_m(self) -> float:
        return self.base_z_m + self.height_m


@dataclass(frozen=True)
class NoFlyZone:
    """静态禁飞棱柱：底面多边形 + **绝对高程**区间 ``[z_min_m, z_max_m]``。

    与 ``Building`` 同构（都是 footprint × 竖直区间），因此 SDF 与真几何终检共用一套
    障碍机制。区别是建筑从地面长上来，禁飞区可以是**悬浮层**：从其上/其下绕过都合法。
    高度口径是 ENU 的绝对 z（= 地面高程 + AGL），不是 AGL；数据方给 AGL 时必须先换算。

    动态禁飞（时间窗）不在本契约内：单段规划只有几秒，由服务层按任务时刻挑出当前
    生效的静态集合传入即可，不必把时间维塞进 NLP。
    """

    footprint: tuple[tuple[float, float], ...]
    z_min_m: float = 0.0
    z_max_m: float = 1000.0
    reason: str = ""

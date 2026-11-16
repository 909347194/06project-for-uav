"""scene_io：世界 / 任务数据适配层（**零依赖**，只用标准库）。

它解决的是"数据怎么落到磁盘"这一层，不是"算法怎么算"：

- 一个任务包只绑一个世界，靠 ``world_id + frame_hash`` 显式声明，对不上直接报错；
- 经纬度是水平坐标真源，ENU 米由 ``world.frame`` 现算，绝不落库。

**与两个算法库的关系**：互不 import。ta1 / pp1 各自独立安装即可使用；需要对接时用
``bridges`` 里的懒加载桥（``to_ta1_request`` / ``to_pp1_world_kwargs``），
底层契约是同一套 ENU 米 + 同一个 ``frame_hash``。
"""

from __future__ import annotations

from .bridges import (
    check_world_with_pp1,
    to_pp1_world_kwargs,
    to_ta1_cost_samplers,
    to_ta1_request,
)
from .mission import (
    Mission,
    SceneNest,
    SceneTask,
    SceneUav,
    load_mission,
)
from .world import (
    Frame,
    World,
    frame_from_region,
    frame_hash,
    load_world,
    region_bbox,
)

__version__ = "0.2.0"

__all__ = [
    "Frame",
    "Mission",
    "SceneNest",
    "SceneTask",
    "SceneUav",
    "World",
    "__version__",
    "check_world_with_pp1",
    "frame_from_region",
    "frame_hash",
    "load_mission",
    "load_world",
    "region_bbox",
    "to_pp1_world_kwargs",
    "to_ta1_cost_samplers",
    "to_ta1_request",
]

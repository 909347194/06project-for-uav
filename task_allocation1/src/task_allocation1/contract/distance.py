"""距离协议：本库不实现任何地形/文件 I/O，只认注入的 ``DistanceFn``。

``DistanceFn(pos_a, pos_b) -> float``：输入两个三维坐标序列，输出**有向**距离（米）。
- 同巢起降下返航腿也要走它：末任务 → 起飞机巢（=终点机巢）。
- ``None`` 时退化为三维欧氏（见 :func:`euclidean3d`），仅供无地形场景 / 测试。
- 有向性：爬升/地形口径下 d(a,b) 可能 ≠ d(b,a)；调用方若有缓存须按有序键缓存。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

# 三维坐标：长度 >= 3 的序列（取前 3 位 x/y/z）
Position = Sequence[float]
DistanceFn = Callable[[Position, Position], float]


def euclidean3d(a: Position, b: Position) -> float:
    """默认距离口径：三维直线距离。纯函数，无状态、无 I/O。"""
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    dz = float(a[2]) - float(b[2]) if len(a) > 2 and len(b) > 2 else 0.0
    return math.sqrt(dx * dx + dy * dy + dz * dz)

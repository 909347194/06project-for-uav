"""走廊窗口必须**覆盖整条飞行带**（含天花板），不能被取整截短。

背景（真实缺陷）：``corridor_window`` 的竖向层数以前用 ``round``，而水平两轴用
``ceil``。``sz`` 不整除带高时 ``round`` 向下取整，窗口顶就低于 ``agl_max_m``：
例 ``sz=6``、带 10–120 m → 110/6=18.333 → round 18 → 窗口顶 118 m < 120 m，
真高上限附近的合法点会被 ``inside_grid`` 判成窗外 / 取到边缘 SDF，与
「``agl_max_m`` 是硬天花板」的契约矛盾。默认 (4,4,2) 下 110/2 恰好整除，
所以只有换格距才会暴露——必须有专门的用例守它。
"""

from __future__ import annotations

import math

import pytest
from path_planning1.contract import RegionOfInterest
from path_planning1.core.corridor_window import corridor_window

_ROI = RegionOfInterest(width_ratio=0.2, margin_m=50.0)
_ALTS = (10.0, 120.0)


def _top_z(spacing: tuple[float, float, float]) -> float:
    w = corridor_window((0.0, 0.0, 60.0), (1000.0, 0.0, 60.0), _ROI, spacing, *_ALTS)
    nz = w.grid.shape[2]
    return float(w.grid.origin_enu[2]) + (nz - 1) * float(w.grid.spacing_m[2])


@pytest.mark.parametrize("spacing", [
    (4.0, 4.0, 2.0),      # 默认档：本来就整除
    (4.0, 4.0, 6.0),      # 110/6 = 18.333…：以前被 round 截到 118 m
    (4.0, 4.0, 7.0),      # 110/7 = 15.714…
    (4.0, 4.0, 3.0),      # 110/3 = 36.666…
    (4.0, 4.0, 11.0),     # 恰好整除
])
def test_window_covers_whole_flight_band(spacing: tuple[float, float, float]) -> None:
    top = _top_z(spacing)
    assert top >= _ALTS[1] - 1e-9, (
        f"spacing={spacing} 时窗口顶只有 {top:.2f} m，够不到 agl_max_m={_ALTS[1]}："
        "竖向层数必须向上取整（ceil），与水平轴同口径"
    )


def test_window_bottom_is_band_floor() -> None:
    w = corridor_window((0.0, 0.0, 60.0), (1000.0, 0.0, 60.0), _ROI, (4.0, 4.0, 6.0), *_ALTS)
    assert float(w.grid.origin_enu[2]) == pytest.approx(_ALTS[0])


def test_vertical_layer_count_uses_ceil() -> None:
    """层数 = ceil(带高/格距) + 1：不整除时多出一层，而不是少一层。"""
    sz = 6.0
    w = corridor_window((0.0, 0.0, 60.0), (1000.0, 0.0, 60.0), _ROI, (4.0, 4.0, sz), *_ALTS)
    assert w.grid.shape[2] == math.ceil((_ALTS[1] - _ALTS[0]) / sz) + 1

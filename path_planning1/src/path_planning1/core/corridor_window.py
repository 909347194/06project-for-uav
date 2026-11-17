"""ROI 走廊 → **走廊局部系**的 3D 栅格窗口（本工程相对论文实现的关键设计）。

论文把走廊当解空间限制、栅格仍是整块；5 km 规模必须同时把"算哪块"缩小，否则
CasADi 插值器建不起来。这里的做法是**让栅格沿走廊轴向转**（``GridSpec.yaw_rad``）：
窗口的格数正比于走廊带面积 ``L × 2(half_width + margin)``，而不是它的 ENU 外接矩形。
外接矩形在斜向走廊下会多算 1.6–3.4 倍（1200 m 级走廊里那是几千万格），这是本设计
唯一的目的——**只算走廊内**。

两点口径，别混：

* 栅格在 ENU 里是**斜的**，但坐标仍是 ENU 米制：算法只跟 ``GridSpec.index_to_enu`` /
  ``enu_to_index`` 打交道（见 ``contract/local_frame.py``），不在别处做旋转。
* 走廊 = **计算域**，也是**解域**：窗口外没有算过 SDF，所以"窗口外"一律不可用。
  这不是可选开关；窄走廊会把"可解"变"无解"，由 ``RRT_NO_PATH`` 如实报出来。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ..contract import GridSpec, RegionOfInterest

_MIN_HORIZONTAL_M = 1e-6
# 索引空间的浮点容差（单位：格）。斜栅格 index→ENU→index 往返有 ~1e-13 格误差；
# margin=0 时起点正好落在格 0 上，会被判成"窗外"（实测 -3.3e-14），整条航段白报 RRT_NO_PATH。
_EPS_INDEX = 1e-6


@dataclass(frozen=True)
class CorridorWindow:
    """绕水平起终点线段的走廊窗口（斜栅格，轴向 = 起终点方向）。"""

    grid: GridSpec
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    u: np.ndarray        # (2,) 水平单位向量 start→goal（= 栅格轴 0 方向）
    normal: np.ndarray   # (2,) 水平单位法向（= 栅格轴 1 方向）
    half_width_m: float  # 用户给的走廊半宽
    lateral_m: float     # 栅格实际半宽 = half_width + margin（≥2 格）

    @property
    def horizontal_length_m(self) -> float:
        dx = self.goal_xy[0] - self.start_xy[0]
        dy = self.goal_xy[1] - self.start_xy[1]
        return math.hypot(dx, dy)

    def lateral_of(self, pts: np.ndarray) -> np.ndarray:
        """点到走廊中心轴的**水平**侧向距离（绝对值），形状 ``(...,)``。"""
        d = np.asarray(pts, dtype=float)[..., :2] - np.asarray(self.start_xy, dtype=float)
        return np.abs(d @ self.normal)

    def inside_grid(self, pts: np.ndarray) -> np.ndarray:
        """是否落在窗口栅格内（ENU 入参，内部走 ``enu_to_index``，斜栅格也成立）。

        窗口外一律不采、不判——没算过 SDF 的地方不能当自由空间。
        """
        idx = self.grid.enu_to_index(np.asarray(pts, dtype=float))
        upper = np.asarray(self.grid.shape, dtype=float) - 1.0
        return np.all((idx >= -_EPS_INDEX) & (idx <= upper + _EPS_INDEX), axis=-1)

    @property
    def enu_bounds(self) -> tuple[float, float, float, float]:
        """窗口 4 个水平角点的 ENU 外接矩形 ``x0, y0, x1, y1``（只用于建筑粗筛）。

        斜栅格下它的面积 **大于** 栅格本身，所以按它筛出来的是**超集**：
        ``buildings_in_window`` 是"路过窗口附近"，真正挡路的由逐格栅格化决定。
        """
        corners = self.grid.index_to_enu(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [self.grid.shape[0] - 1.0, 0.0, 0.0],
                    [0.0, self.grid.shape[1] - 1.0, 0.0],
                    [self.grid.shape[0] - 1.0, self.grid.shape[1] - 1.0, 0.0],
                ]
            )
        )
        return (
            float(corners[:, 0].min()),
            float(corners[:, 1].min()),
            float(corners[:, 0].max()),
            float(corners[:, 1].max()),
        )


def corridor_window(
    start_enu: tuple[float, float, float],
    goal_enu: tuple[float, float, float],
    roi: RegionOfInterest,
    spacing_m: tuple[float, float, float],
    alt_min_m: float,
    alt_max_m: float,
) -> CorridorWindow:
    """把走廊（+margin）转成**沿走廊轴向的**窗口栅格。

    轴向：栅格轴 0 = 起终点水平方向，轴 1 = 其法向，轴 2 = 绝对 z（与 ENU 同向）。
    沿轴范围 ``[-margin, L+margin]``，横向范围 ``±lateral``（``lateral = half_width + margin``），
    竖向覆盖整条飞行带。
    """
    sx, sy, sz = (float(v) for v in spacing_m)
    dx = float(goal_enu[0]) - float(start_enu[0])
    dy = float(goal_enu[1]) - float(start_enu[1])
    length = math.hypot(dx, dy)
    if length < _MIN_HORIZONTAL_M:
        raise ValueError(
            "起终点水平重合，无法构造走廊：入口 check_request 已先行拦截，"
            "直接调用本函数时同样按入参错误报出"
        )
    u = np.array([dx / length, dy / length])
    normal = np.array([-u[1], u[0]])

    margin = float(roi.margin_m)
    half_width = roi.half_width_for(length)
    lateral = max(half_width + margin, 2.0 * max(sx, sy))
    start_xy = (float(start_enu[0]), float(start_enu[1]))
    goal_xy = (float(goal_enu[0]), float(goal_enu[1]))

    n_along = math.ceil((length + 2.0 * margin) / sx) + 1
    n_lateral = math.ceil(2.0 * lateral / sy) + 1
    # 竖向也用 ceil（不能用 round）：不整除时 round 会向下取整，窗口顶就低于飞行带
    # 天花板，AGL 上限附近的合法点会被判成窗外/取到边缘 SDF。例 sz=3.2、带 10–120：
    # 110/3.2=34.375 → round 34 → 顶 118.8 m < 120 m。默认 (4,4,2) 下 110/2 恰好整除，
    # 基线不受影响；但 spacing_m 允许任意 sz ≥ 1e-3，这里必须与水平轴同口径。
    nz = math.ceil((alt_max_m - alt_min_m) / sz) + 1

    # 斜栅格的角点 = 起点 - margin·u - lateral·normal（轴 0/1 就是 u/normal）
    corner = np.asarray(start_xy, dtype=float) - margin * u - lateral * normal
    grid = GridSpec(
        origin_enu=(float(corner[0]), float(corner[1]), float(alt_min_m)),
        spacing_m=(sx, sy, sz),
        shape=(n_along, n_lateral, nz),
        yaw_rad=math.atan2(float(u[1]), float(u[0])),
    )
    return CorridorWindow(
        grid=grid,
        start_xy=start_xy,
        goal_xy=goal_xy,
        u=u,
        normal=normal,
        half_width_m=half_width,
        lateral_m=lateral,
    )


def _aligned_step(sx: float, sy: float) -> float:
    """两条水平轴**同时**对齐所需的最小步长 = lcm(sx, sy)（按有理数算）。

    只取 ``min(sx, sy)`` 是错的：``spacing_m=(6,4,2)`` 时 min=4，pad=392 并不是 6 的倍数，
    场格网原点在 x 轴相对世界平移；只改一次 ``max_od_length_m``，所有取样位置就整体变，
    历史基线不可复现。用有理数最小公倍数才能让 pad 同时是 sx 与 sy 的倍数。
    """
    fa = Fraction(str(float(sx))).limit_denominator(10_000)
    fb = Fraction(str(float(sy))).limit_denominator(10_000)
    step = float(Fraction(math.lcm(fa.numerator, fb.numerator),
                          math.gcd(fa.denominator, fb.denominator)))
    if not math.isfinite(step) or step <= 0.0 or step > 1e4:
        raise ValueError(
            f"水平格距 ({sx}, {sy}) 无法对齐：请让 sx/sy 为可公度的小数，"
            "或把两条水平格距设成相同值（默认 4/4）"
        )
    return step


def required_pad_m(
    roi: RegionOfInterest,
    spacing_m: tuple[float, float, float],
    max_od_length_m: float,
    *,
    extra_m: float,
    min_pad_m: float = 0.0,
) -> float:
    """建场契约：任意合法 OD 的走廊窗口都不出场所需的**水平外扩**（米）。

    这是 ``build_world_field(pad_m=...)`` 与请求准入**共用**的唯一出处，不许再写第二份
    （否则又出现两套走廊定义）。三点容易算错的地方：

    * 用的是**最终 lateral**，不是 ``0.5·r·L``：``corridor_window`` 里
      ``lateral = max(half_width + margin, 2·格距)``，且 ``roi.half_width_m`` 显式给定会
      整个绕过 ``r·L``。按 ``r·L`` 直接算会漏掉 margin 与显式半宽。
    * 窗口是**斜**的：角点相对端点的单轴最大偏移是 ``hypot(轴向外扩, 横向外扩)``，
      不是两者之和，也不是单看 lateral；格数向上取整还会让两端各多出不到一格。
    * **结果必须是两条水平格距的公倍数**：``build_world_field`` 用同一个 ``pad`` 外扩
      x/y，格网原点分别是 ``bounds[0]-pad`` / ``bounds[1]-pad``；``window_sdf`` 的取样坐标是
      ``(x - origin)/spacing``。pad 只要不是某一轴的整数倍，该轴的全域格点就整体平移，
      所有窗口的三线性取样值改变、历史基线不可复现。sx≠sy 时取 lcm，不能取 min。
    """
    sx, sy, _ = (float(v) for v in spacing_m)
    lateral = max(roi.half_width_for(max_od_length_m) + roi.margin_m, 2.0 * max(sx, sy))
    reach = math.hypot(roi.margin_m + sx, lateral + sy) + float(extra_m)
    step = _aligned_step(sx, sy)
    return max(math.ceil(reach / step), math.ceil(float(min_pad_m) / step)) * step


def window_area_m2(window: CorridorWindow) -> float:
    """窗口水平投影面积（= 走廊带面积，不再是 ENU 外接矩形）。"""
    nx, ny, _ = window.grid.shape
    sx, sy, _ = window.grid.spacing_m
    return (nx - 1) * sx * (ny - 1) * sy


def z_envelope_from_ground(
    ground_min_m: float, agl_min_m: float, agl_max_m: float
) -> tuple[float, float]:
    """真高带 → 窗口的**绝对** z 包络（唯一一处口径，航段与试跑脚本共用）。

    天花板按**窗口内最低地面**算：这样任何一列都飞不到 120 m 真高以上（高地只会更
    保守），层数仍是 56；按最高地面算会撑到 195 m / 92 层，SDF 成本翻倍。
    代价是**高地上空的可用真高低于 120 m**——这是已知取舍，不是 bug
    （见 ``path_planning1/README.md``「已知限制」）。

    实现上它要求先有一个"只按绝对带"的探测窗口去量地面高差，再用本函数给出的
    ``z_lo/z_hi`` 建正式窗口；两步的 x/y 覆盖完全一致（z 不影响水平尺寸）。
    """
    return float(ground_min_m) + float(agl_min_m), float(ground_min_m) + float(agl_max_m)

"""请求入参自检：**坏数值一律 ValueError，0 秒、不建窗口**。

为什么是异常而不是 ``failure_code``：NaN 坐标、零/负间距是**调用方的编程错误**（JSON 里给了
``null``、米/度混用、间距传了 0），给它一个 ``OUT_OF_FLIGHT_BAND`` 之类的业务码只会误导——后端
会以为"换个高度就好"。而"真高越界"确实是业务输入问题，仍由 ``plan_route_segment`` 返回
``OUT_OF_FLIGHT_BAND``：两类问题的处理方不同，出口也就必须不同。

这里只放"不用数据就能判"的检查；一切需要窗口/SDF 才能判的（起终点落在楼里、带内无解）走失败码。
"""

from __future__ import annotations

import math

from ..contract import defaults
from .corridor_window import _MIN_HORIZONTAL_M


def check_request(
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    spacing_m: tuple[float, float, float],
) -> None:
    """校验坐标与栅格间距。不合法即 ``raise ValueError``（消息带字段名与收到的值）。"""
    pts: dict[str, tuple[float, float]] = {}
    for label, xy in (("start_xy", start_xy), ("goal_xy", goal_xy)):
        if len(xy) != 2 or not all(math.isfinite(float(v)) for v in xy):
            raise ValueError(f"{label} 必须是两个有限数（收到 {xy!r}）")
        pts[label] = (float(xy[0]), float(xy[1]))
    # 水平重合时走廊无法定向：以前由 corridor_window 抛英文 ValueError，和这里的入参
    # 自检不是同一个出口。统一到入口，阈值与 corridor_window._MIN_HORIZONTAL_M 同源。
    if math.dist(pts["start_xy"], pts["goal_xy"]) <= _MIN_HORIZONTAL_M:
        raise ValueError(
            "start_xy 与 goal_xy 水平重合（距离 < 1e-6 m），无法构造走廊窗口："
            "请给出两个不同的水平点；原地起飞不是一条航段"
        )
    spacing = tuple(float(v) for v in spacing_m)
    if len(spacing) != 3 or min(spacing) <= 0 or not all(math.isfinite(v) for v in spacing):
        raise ValueError(f"spacing_m 必须是 3 个正的有限数（收到 {spacing_m!r}）")


def check_vehicle(
    thrust_max_n: float,
    mass_kg: float,
    gravity_mps2: float = defaults.GRAVITY_MPS2,
) -> None:
    """机型自洽性：**悬停都要掉的配置**是填错了，不是这条航段飞不过去。

    多旋翼悬停就要求 ``T_max ≥ m·g``。低于这条线时最小推力/最大推力都不存在，任何轨迹都是
    假的（以前这类配置会一路算完、报一个"ok"出来）。所以按调用方错误处理：``ValueError``、
    0 秒、不建窗口——和后端"换个高度重试"那类业务失败分开。
    """
    t_max, mass = float(thrust_max_n), float(mass_kg)
    for label, v in (("thrust_max_n", t_max), ("mass_kg", mass)):
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError(f"{label} 必须是正的有限数（收到 {v!r}）")
    hover = mass * float(gravity_mps2)
    if t_max < hover:
        raise ValueError(
            f"VehicleLimits 不自洽：悬停就需 {hover:.2f} N（{mass:g} kg × {gravity_mps2:g} m/s²），"
            f"但 thrust_max_n={t_max:g} N（推重比 {t_max / hover:.2f} < 1）。"
            "这种机型飞不起来：请核对 mass_kg 与 thrust_max_n，别指望调高度/走廊能救"
        )

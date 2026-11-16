"""坐标与栅格规格：WGS84 ↔ 局地 ENU 米制，ENU ↔ 栅格索引。

与 ExperimentForGraduation 的 ``LocalCRS`` 同一公式，便于和 task_allocation 对接。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

M_PER_DEG_LAT = 110574.0
M_PER_DEG_LON_EQ = 111320.0


@dataclass(frozen=True)
class LocalFrame:
    """WGS84 经纬度 ↔ 局地 ENU 米制。

    ``x = (lon-lon0)·111320·cos(lat_ref)``，``y = (lat-lat0)·110574``。
    局地切平面近似，仅适用跨度 < 25 km 的小区域；再大改用 UTM，
    带号须按当地经度选（``zone = floor((lon+180)/6)+1``，北半球 EPSG:326xx），
    不能照抄广州的 49 带——北京属 50 带、乌鲁木齐属 45 带。
    """

    lon0: float
    lat0: float
    lat_ref: float | None = None   # 通常取 bbox 中心纬度；None 则用 lat0

    @property
    def lat_reference(self) -> float:
        return self.lat0 if self.lat_ref is None else float(self.lat_ref)

    @property
    def m_per_deg_lon(self) -> float:
        return M_PER_DEG_LON_EQ * math.cos(math.radians(self.lat_reference))

    def to_local(self, lon: float, lat: float) -> tuple[float, float]:
        return (
            (float(lon) - self.lon0) * self.m_per_deg_lon,
            (float(lat) - self.lat0) * M_PER_DEG_LAT,
        )

    def to_lonlat(self, x: float, y: float) -> tuple[float, float]:
        return (
            self.lon0 + float(x) / self.m_per_deg_lon,
            self.lat0 + float(y) / M_PER_DEG_LAT,
        )


@dataclass(frozen=True)
class GridSpec:
    """3D 栅格规格：``shape=(nx, ny, nz)`` + 原点 + 间距 + **xy 平面内的旋转角**。

    索引 (i, j, k) 的物理坐标::

        p = origin_enu + i·sx·**a0** + j·sy·**a1** + k·sz·**ez**

    其中 ``a0 = (cos yaw, sin yaw)``、``a1 = (-sin yaw, cos yaw)`` 是 xy 两个栅格轴在
    ENU 里的单位向量；``yaw_rad=0`` 时就是 ``ex`` / ``ey``（全域栅格与旧的轴对齐栅格）。
    ``yaw_rad≠0`` 时栅格在 ENU 里是斜的——**只有索引↔ENU 的换算受它影响**，
    ``z`` 恒是绝对 ENU 高度，所有算法看到的坐标仍是 ENU 米制（见 ``core/corridor_window.py``）。

    ``origin_enu[2]`` 通常等于飞行带下界 ``alt_min_m``。
    """

    origin_enu: tuple[float, float, float]
    spacing_m: tuple[float, float, float]
    shape: tuple[int, int, int]
    yaw_rad: float = 0.0            # xy 栅格轴相对 ENU 的旋转角（弧度，逆时针）

    @property
    def axes_xy(self) -> tuple[np.ndarray, np.ndarray]:
        """xy 两个栅格轴在 ENU 里的单位向量（``yaw_rad=0`` → ``(1,0)`` / ``(0,1)``）。"""
        c, s = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        return np.array([c, s], dtype=float), np.array([-s, c], dtype=float)

    def index_to_enu(self, idx: np.ndarray) -> np.ndarray:
        """整数/连续索引（..., 3）→ ENU 米制坐标（..., 3）。"""
        idx = np.asarray(idx, dtype=float)
        a0, a1 = self.axes_xy
        sx, sy, sz = (float(v) for v in self.spacing_m)
        ox, oy, oz = (float(v) for v in self.origin_enu)
        x = ox + idx[..., 0] * sx * a0[0] + idx[..., 1] * sy * a1[0]
        y = oy + idx[..., 0] * sx * a0[1] + idx[..., 1] * sy * a1[1]
        z = oz + idx[..., 2] * sz
        return np.stack([x, y, z], axis=-1)

    def enu_to_index(self, p: np.ndarray) -> np.ndarray:
        """ENU 米制坐标（..., 3）→ 连续索引（..., 3），不取整。

        ``a0/a1`` 是正交单位向量，所以逆变换就是**投影**（转置），不是求逆矩阵。
        """
        p = np.asarray(p, dtype=float)
        a0, a1 = self.axes_xy
        sx, sy, sz = (float(v) for v in self.spacing_m)
        ox, oy, oz = (float(v) for v in self.origin_enu)
        d = p[..., :2] - np.array([ox, oy], dtype=float)
        i = (d @ a0) / sx
        j = (d @ a1) / sy
        k = (p[..., 2] - oz) / sz
        return np.stack([i, j, k], axis=-1)

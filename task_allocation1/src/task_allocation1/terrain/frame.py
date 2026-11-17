"""WGS84 经纬度 ↔ 局地 ENU 米制（与 pp1 同公式，只用 numpy）。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

M_PER_DEG_LAT = 110574.0
M_PER_DEG_LON_EQ = 111320.0


@dataclass(frozen=True)
class LocalFrame:
    """小区域局地切平面；``lat_ref`` 通常取 bbox 中心纬度。"""

    lon0: float
    lat0: float
    lat_ref: float | None = None

    @property
    def lat_reference(self) -> float:
        return self.lat0 if self.lat_ref is None else float(self.lat_ref)

    @property
    def m_per_deg_lon(self) -> float:
        return M_PER_DEG_LON_EQ * math.cos(math.radians(self.lat_reference))

    def to_local(self, lon, lat) -> tuple[np.ndarray, np.ndarray]:
        x = (np.asarray(lon, dtype=float) - self.lon0) * self.m_per_deg_lon
        y = (np.asarray(lat, dtype=float) - self.lat0) * M_PER_DEG_LAT
        return x, y

    def to_lonlat(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        lon = self.lon0 + np.asarray(x, dtype=float) / self.m_per_deg_lon
        lat = self.lat0 + np.asarray(y, dtype=float) / M_PER_DEG_LAT
        return lon, lat

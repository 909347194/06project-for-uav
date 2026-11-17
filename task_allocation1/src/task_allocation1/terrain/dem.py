"""GeoTIFF DEM → 经纬度网格双线性采样（只用 numpy；读文件用 Pillow）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_TAG_PIXEL_SCALE = 33550
_TAG_TIEPOINT = 33922
_TAG_NODATA = 42113


@dataclass(frozen=True)
class DEM:
    """经纬度规则网格的高程（米）；``z_m`` 的行序为北→南。"""

    z_m: np.ndarray
    lon_ul: float
    lat_ul: float
    dlon: float
    dlat: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.z_m.shape

    @classmethod
    def from_array(
        cls,
        elevation: np.ndarray,
        bounds: tuple[float, float, float, float],
    ) -> DEM:
        """``bounds=(left, bottom, right, top)``，单位为度。"""
        z = np.asarray(elevation, dtype=float)
        if z.ndim != 2 or min(z.shape) < 2:
            raise ValueError(f"DEM 至少要是 2×2 的二维数组，收到 {z.shape}")
        left, bottom, right, top = (float(v) for v in bounds)
        if not (right > left and top > bottom):
            raise ValueError(f"DEM bounds 非法：{bounds}")
        h, w = z.shape
        return cls(
            z_m=z,
            lon_ul=left,
            lat_ul=top,
            dlon=(right - left) / w,
            dlat=(top - bottom) / h,
        )

    def at_lonlat(self, lon, lat) -> np.ndarray:
        """双线性采样；域外或参与加权的格点为空值时返回 NaN。"""
        lon_a = np.asarray(lon, dtype=float)
        lat_a = np.asarray(lat, dtype=float)
        lon_a, lat_a = np.broadcast_arrays(lon_a, lat_a)
        z = self.z_m
        h, w = z.shape
        col = (lon_a - self.lon_ul) / self.dlon
        row = (self.lat_ul - lat_a) / self.dlat
        outside = (col < 0) | (col > w - 1) | (row < 0) | (row > h - 1)
        col = np.clip(col, 0, w - 1)
        row = np.clip(row, 0, h - 1)
        c0 = np.clip(np.floor(col).astype(int), 0, w - 2)
        r0 = np.clip(np.floor(row).astype(int), 0, h - 2)
        dc = col - c0
        dr = row - r0
        z00 = z[r0, c0]
        z01 = z[r0, c0 + 1]
        z10 = z[r0 + 1, c0]
        z11 = z[r0 + 1, c0 + 1]
        bad = ~(np.isfinite(z00) & np.isfinite(z01) & np.isfinite(z10) & np.isfinite(z11))
        out = (
            (1 - dc) * (1 - dr) * z00
            + dc * (1 - dr) * z01
            + (1 - dc) * dr * z10
            + dc * dr * z11
        )
        return np.where(outside | bad, np.nan, out)


def load_geotiff_dem(path: str | Path, *, nodata: float | None = None) -> DEM:
    """读单波段 GeoTIFF（EPSG:4326）。依赖 ``pillow``（随包安装）。"""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - 环境问题
        raise ModuleNotFoundError(
            "读 GeoTIFF 需要 pillow：pip install 'task_allocation1[terrain]'"
        ) from exc

    path = Path(path)
    with Image.open(path) as im:
        tags = im.tag_v2
        if _TAG_PIXEL_SCALE not in tags or _TAG_TIEPOINT not in tags:
            raise ValueError(f"{path} 缺少 GeoTIFF 地理标签 33550/33922")
        scale = tags[_TAG_PIXEL_SCALE]
        tie = tags[_TAG_TIEPOINT]
        if len(tie) < 6:
            raise ValueError(f"{path} 的 ModelTiepoint 长度异常：{len(tie)}")
        dlon, dlat = float(scale[0]), float(abs(scale[1]))
        lon_ul = float(tie[3]) - float(tie[0]) * dlon
        lat_ul = float(tie[4]) + float(tie[1]) * dlat
        z = np.asarray(im, dtype=float)
        if z.ndim != 2:
            raise ValueError(f"{path} 不是单波段 DEM：shape={z.shape}")
        tag_nodata = tags.get(_TAG_NODATA)
        if nodata is None and tag_nodata is not None:
            nodata = float(tag_nodata)
        if nodata is not None:
            z = np.where(~np.isfinite(z) | (z == float(nodata)), np.nan, z)
    return DEM(z_m=z, lon_ul=lon_ul, lat_ul=lat_ul, dlon=dlon, dlat=dlat)

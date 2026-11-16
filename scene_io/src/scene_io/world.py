"""世界描述：``worlds/<world_id>/world.json`` 的加载与自洽校验。

**零依赖**：只用标准库，不 import ``path_planning1``。frame 的换算公式在这里独立实现一份
（3 行算术），与 pp1 的 ``LocalFrame`` 口径相同；两边是否真的一致，由 ``frame_hash``
和测试对拍来保证，而不是靠"共享代码"保证——两库必须能各自单独安装使用。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

WORLD_FILE = "world.json"
SCHEMA_WORLD = 1          # world.json 的 "schema" 字段；升级要写迁移，不靠人记
M_PER_DEG_LAT = 110574.0
M_PER_DEG_LON_EQ = 111320.0


@dataclass(frozen=True)
class Frame:
    """WGS84 ↔ 局地 ENU 米（与 pp1 的 ``LocalFrame`` 同一公式）。

    ``x = (lon - lon0) · 111320 · cos(lat_ref)``；``y = (lat - lat0) · 110574``。
    """

    lon0: float
    lat0: float
    lat_ref: float | None = None

    @property
    def lat_reference(self) -> float:
        return self.lat0 if self.lat_ref is None else float(self.lat_ref)

    @property
    def m_per_deg_lon(self) -> float:
        return M_PER_DEG_LON_EQ * math.cos(math.radians(self.lat_reference))

    def to_local(self, lon: float, lat: float) -> tuple[float, float]:
        x = (float(lon) - self.lon0) * self.m_per_deg_lon
        y = (float(lat) - self.lat0) * M_PER_DEG_LAT
        return (0.0 if x == 0 else x), (0.0 if y == 0 else y)

    def to_lonlat(self, x: float, y: float) -> tuple[float, float]:
        return (self.lon0 + float(x) / self.m_per_deg_lon,
                self.lat0 + float(y) / M_PER_DEG_LAT)


def frame_hash(frame: Frame) -> str:
    """frame 指纹：``sha1(f"{lon0:.9f},{lat0:.9f},{lat_ref:.9f}")[:12]``。

    任何一侧要自算 frame_hash 都必须用这一行，否则两侧永远对不上。
    """
    text = f"{frame.lon0:.9f},{frame.lat0:.9f},{frame.lat_reference:.9f}"
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def region_bbox(region_path: str | Path) -> tuple[float, float, float, float]:
    """GeoJSON 的经纬度外接矩形 ``(lon0, lat0, lon1, lat1)``；纯标准库遍历坐标。"""
    data = json.loads(Path(region_path).read_text(encoding="utf-8"))
    lons: list[float] = []
    lats: list[float] = []

    def walk(node: Any) -> None:
        if isinstance(node, (list, tuple)) and node and isinstance(node[0], (int, float)):
            lons.append(float(node[0]))
            lats.append(float(node[1]))
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                walk(child)

    for feature in data.get("features") or []:
        geom = feature.get("geometry")
        if geom:
            walk(geom.get("coordinates"))

    if not lons:
        raise ValueError(f"{region_path} 里没有可用几何（Polygon / MultiPolygon）")
    return min(lons), min(lats), max(lons), max(lats)


def frame_from_region(region_path: str | Path) -> Frame:
    """与 pp1 的 ``frame_from_geojson`` 同口径：锚点取 bbox 左下角，``lat_ref`` 取 bbox 中心纬度。"""
    lon0, lat0, _lon1, lat1 = region_bbox(region_path)
    return Frame(lon0=lon0, lat0=lat0, lat_ref=0.5 * (lat0 + lat1))


@dataclass(frozen=True)
class World:
    """一个世界（城市）的只读描述。"""

    world_id: str
    root: Path
    frame: Frame
    frame_hash: str
    dataset_digest: str | None
    elevation_basis: str
    raw: dict[str, Any]

    @property
    def region_path(self) -> Path:
        return self.root / (self.raw.get("contents", {}).get("region") or "region.geojson")

    @property
    def building_dir(self) -> Path:
        return self.root / (self.raw.get("contents", {}).get("buildings") or "building")

    @property
    def dem_path(self) -> Path | None:
        name = self.raw.get("contents", {}).get("dem")
        return (self.root / name) if name else None

    @property
    def nofly_path(self) -> Path | None:
        name = self.raw.get("contents", {}).get("nofly")
        return (self.root / name) if name else None

    def world_kwargs(self, **overrides: Any) -> dict[str, Any]:
        """给 pp1 的 ``world`` 段（纯 dict，不 import pp1）。"""
        out: dict[str, Any] = {"root": str(self.root), "with_dem": self.dem_path is not None}
        if self.nofly_path is not None:
            out["nofly_path"] = str(self.nofly_path)
        out.update(overrides)
        return out


def load_world(world_dir: str | Path, *, verify_region: bool = True) -> World:
    """读 ``worlds/<world_id>/world.json``。

    校验（失败即 ``ValueError`` / ``FileNotFoundError``）：

    1. ``frame_hash`` 必须等于按其 ``frame`` 现算的值（防止手改 frame 不重算指纹）；
    2. ``verify_region=True`` 时，frame 必须等于从 ``region.geojson`` 现算的 frame
       （region 是 frame 的唯一真源）。
    """
    root = Path(world_dir)
    meta_path = root / WORLD_FILE
    if not meta_path.is_file():
        raise FileNotFoundError(f"{root} 下缺少 {WORLD_FILE}：世界目录必须自带世界描述")

    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    schema = raw.get("schema")
    if schema is None:
        raise ValueError(
            f"{meta_path} 缺少 schema 版本锚（应为 {SCHEMA_WORLD}）：加一行 \"schema\": {SCHEMA_WORLD} 即可"
        )
    if int(schema) != SCHEMA_WORLD:
        raise ValueError(
            f"{meta_path} 的 schema={schema!r} 本包不认识（只认 {SCHEMA_WORLD}）："
            "先按新版本迁移 world.json，再升级加载代码"
        )
    fr_raw = raw.get("frame") or {}
    frame = Frame(
        lon0=float(fr_raw["lon0"]),
        lat0=float(fr_raw["lat0"]),
        lat_ref=None if fr_raw.get("lat_ref") is None else float(fr_raw["lat_ref"]),
    )
    declared = str(fr_raw.get("frame_hash") or "")
    actual = frame_hash(frame)
    if declared and declared != actual:
        raise ValueError(
            f"{meta_path} 的 frame_hash={declared!r} 与 frame 现算值 {actual!r} 不一致："
            "frame 被改过但没重算指纹"
        )

    region = root / (raw.get("contents", {}).get("region") or "region.geojson")
    if verify_region and region.is_file():
        geo = frame_from_region(region)
        same = (round(geo.lon0, 9), round(geo.lat0, 9), round(geo.lat_reference, 9)) == (
            round(frame.lon0, 9), round(frame.lat0, 9), round(frame.lat_reference, 9))
        if not same:
            raise ValueError(
                f"{meta_path} 的 frame 与 {region.name} 现算的 frame 不一致："
                f"world.json={frame} ↔ region={geo}；frame 只能由 region.geojson 决定"
            )

    return World(
        world_id=str(raw.get("world_id") or root.name),
        root=root,
        frame=frame,
        frame_hash=actual,
        dataset_digest=raw.get("dataset_digest"),
        elevation_basis=str((raw.get("elevation") or {}).get("basis") or "absolute_enu_z"),
        raw=raw,
    )

"""任务包：``missions/<mission_id>/`` 的加载与校验。

**零依赖**：只用标准库，产出本模块自己的纯数据类（不 import ta1）。
要喂给 ta1，用 ``scene_io.bridges.to_ta1_request()``（懒加载，两库互不牵连）。

一个任务包只绑一个世界。**经纬度是水平坐标真源**，ENU 由 ``world.frame`` 现算；
``z_abs`` 是绝对高程（不是 AGL，AGL 只在 pp1 请求入口出现）。
``mission.json`` 的 ``cost`` 段（可选）透传给 ta1 的距离口径配置
（``vertical_section`` 默认，需配合采样器 / ``euclid3d`` 供无地形场景显式选用，
及飞行带、净空、爬升率）。

加载期一次报出全部数据问题，不逐条抛——数据错的时候一次看全比改一条跑一次强。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .world import World, load_world

MISSION_FILE = "mission.json"
SCHEMA_MISSION = 1        # mission.json 的 "schema" 字段；升级要写迁移
SUPPORTED_TIME_BASES = ("relative_s",)
SUPPORTED_Z_BASES = ("absolute_enu_z",)


@dataclass(frozen=True)
class SceneTask:
    """任务点：``x/y`` 是派生 ENU 米，``lon/lat`` 是真源，``z_abs`` 绝对高程。"""

    id: int
    x: float
    y: float
    z: float
    lon: float
    lat: float
    reward: float = 1.0
    earliest: float | None = None
    latest: float | None = None


@dataclass(frozen=True)
class SceneUav:
    id: int
    x: float
    y: float
    z: float
    lon: float
    lat: float
    nest_id: int | None = None
    speed: float = 10.0
    remaining_range: float = 1e9
    max_time: float | None = None
    max_range: float | None = None
    status: str = "IDLE"


@dataclass(frozen=True)
class SceneNest:
    id: int
    x: float
    y: float
    z: float
    lon: float
    lat: float
    capacity: int = 10
    available: bool = True
    name: str | None = None


@dataclass(frozen=True)
class Mission:
    """一个任务包的只读视图（纯数据，不含任何库类型）。"""

    mission_id: str
    root: Path
    world: World
    current_time: float
    time_base: str
    z_basis: str
    tasks: list[SceneTask] = field(default_factory=list)
    uavs: list[SceneUav] = field(default_factory=list)
    nests: list[SceneNest] = field(default_factory=list)
    solver: dict[str, Any] = field(default_factory=dict)
    problem: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_enu(self, lon: float, lat: float) -> tuple[float, float]:
        return self.world.frame.to_local(lon, lat)

    def summary(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "world_id": self.world.world_id,
            "frame_hash": self.world.frame_hash,
            "n_tasks": len(self.tasks),
            "n_uavs": len(self.uavs),
            "n_nests": len(self.nests),
            "current_time": self.current_time,
        }


def _read_array(root: Path, name: str, problems: list[str]) -> list[dict[str, Any]]:
    path = root / name
    if not path.is_file():
        problems.append(f"缺少 {name}")
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        problems.append(f"{name} 必须是数组")
        return []
    return data


def _point(raw: dict[str, Any], where: str, world: World,
           problems: list[str]) -> tuple[float, float, float, float, float]:
    """返回 ``(x, y, z_abs, lon, lat)``；水平坐标只认 lon/lat。"""
    if raw.get("lon") is None or raw.get("lat") is None:
        if raw.get("x") is not None or raw.get("y") is not None:
            problems.append(
                f"{where} 用了 x/y（ENU 米）：任务包只收 lon/lat，ENU 是派生值。"
                "请回填经纬度，或用 world.frame.to_lonlat(x, y) 反算"
            )
        else:
            problems.append(f"{where} 缺少 lon/lat")
        return 0.0, 0.0, 0.0, 0.0, 0.0

    z = raw.get("z_abs")
    if z is None:
        problems.append(f"{where} 缺少 z_abs（绝对高程，米；不是 AGL）")
        z = 0.0
    lon, lat = float(raw["lon"]), float(raw["lat"])
    x, y = world.frame.to_local(lon, lat)
    return x, y, float(z), lon, lat


def load_mission(mission_dir: str | Path, *, world: World | None = None,
                 world_root: str | Path | None = None) -> Mission:
    """读 ``missions/<mission_id>/``。

    世界来源：显式 ``world``，或显式 ``world_root``，或默认
    ``data/worlds/<mission.world_id>``（相对当前工作目录；服务里建议显式给）。
    """
    root = Path(mission_dir)
    meta_path = root / MISSION_FILE
    if not meta_path.is_file():
        raise FileNotFoundError(f"{root} 下缺少 {MISSION_FILE}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    problems: list[str] = []
    schema = meta.get("schema")
    if schema is None:
        problems.append(f"mission.json 缺少 schema 版本锚（应为 {SCHEMA_MISSION}）")
    elif int(schema) != SCHEMA_MISSION:
        problems.append(
            f"mission.json 的 schema={schema!r} 本包不认识（只认 {SCHEMA_MISSION}）：先迁移任务包再升级代码"
        )
    world_id = meta.get("world_id")
    if not world_id:
        problems.append("mission.json 缺少 world_id")
        world_id = ""

    if world is None:
        world_root = Path(world_root) if world_root is not None else Path("data") / "worlds" / str(world_id)
        world = load_world(world_root)

    if str(world_id) != world.world_id:
        problems.append(f"world_id 不匹配：mission={world_id!r} ↔ world={world.world_id!r}")

    declared = meta.get("frame_hash")
    if not declared:
        problems.append("mission.json 缺少 frame_hash（防止换 frame 后静默跑错）")
    elif declared != world.frame_hash:
        problems.append(
            f"frame_hash 不匹配：mission={declared!r} ↔ world={world.frame_hash!r}。"
            "任务包不是用这个世界的 frame 算的，必须先重算坐标"
        )

    time_base = str(meta.get("time_base") or "relative_s")
    if time_base not in SUPPORTED_TIME_BASES:
        problems.append(f"time_base={time_base!r} 不支持，只支持 {SUPPORTED_TIME_BASES}")

    z_basis = str(meta.get("z_basis") or "absolute_enu_z")
    if z_basis not in SUPPORTED_Z_BASES:
        problems.append(f"z_basis={z_basis!r} 不支持，只支持 {SUPPORTED_Z_BASES}（AGL 不落库）")

    nests: list[SceneNest] = []
    for i, raw in enumerate(_read_array(root, "nests.json", problems)):
        x, y, z, lon, lat = _point(raw, f"nests.json[{i}]", world, problems)
        nests.append(SceneNest(
            id=int(raw["id"]), x=x, y=y, z=z, lon=lon, lat=lat,
            capacity=int(raw.get("capacity", 10)),
            available=bool(raw.get("available", True)),
            name=raw.get("name"),
        ))

    uavs: list[SceneUav] = []
    for i, raw in enumerate(_read_array(root, "uavs.json", problems)):
        x, y, z, lon, lat = _point(raw, f"uavs.json[{i}]", world, problems)
        uavs.append(SceneUav(
            id=int(raw["id"]), x=x, y=y, z=z, lon=lon, lat=lat,
            nest_id=raw.get("nest_id"),
            speed=float(raw.get("speed", 10.0)),
            remaining_range=float(raw.get("remaining_range", 1e9)),
            max_time=None if raw.get("max_time") is None else float(raw["max_time"]),
            max_range=None if raw.get("max_range") is None else float(raw["max_range"]),
            status=str(raw.get("status", "IDLE")),
        ))

    tasks: list[SceneTask] = []
    for i, raw in enumerate(_read_array(root, "tasks.json", problems)):
        x, y, z, lon, lat = _point(raw, f"tasks.json[{i}]", world, problems)
        tasks.append(SceneTask(
            id=int(raw["id"]), x=x, y=y, z=z, lon=lon, lat=lat,
            reward=float(raw.get("reward", 1.0)),
            earliest=None if raw.get("earliest") is None else float(raw["earliest"]),
            latest=None if raw.get("latest") is None else float(raw["latest"]),
        ))

    # ---- 关系校验（比 ta1 报得更早、更具体）--------------------------------
    for name, rows in (("nests", nests), ("uavs", uavs), ("tasks", tasks)):
        seen: set[int] = set()
        for r in rows:
            if r.id in seen:
                problems.append(f"{name}.json 中 id={r.id} 重复")
            seen.add(r.id)

    nest_ids = {n.id for n in nests}
    for u in uavs:
        if u.nest_id is None:
            problems.append(f"uavs.json 中 id={u.id} 缺少 nest_id（同巢起降必须有起飞巢）")
        elif u.nest_id not in nest_ids:
            problems.append(f"uavs.json 中 id={u.id} 的 nest_id={u.nest_id} 不在 nests.json 里")

    occ: dict[int, int] = {}
    for u in uavs:
        if u.nest_id is not None:
            occ[u.nest_id] = occ.get(u.nest_id, 0) + 1
    for n in nests:
        if occ.get(n.id, 0) > n.capacity:
            problems.append(
                f"巢 {n.id} 归属 {occ[n.id]} 架机，超过 capacity={n.capacity}"
                "（ta1 按归属计数判容量，不是按降落时刻）"
            )

    if problems:
        label = str(meta.get("mission_id") or root.name)
        raise ValueError(
            f"任务包 {label} 数据校验失败（{len(problems)} 项，目录 {root}）：\n- "
            + "\n- ".join(problems)
        )

    return Mission(
        mission_id=str(meta.get("mission_id") or root.name),
        root=root,
        world=world,
        current_time=float(meta.get("current_time", 0.0)),
        time_base=time_base,
        z_basis=z_basis,
        tasks=tasks,
        uavs=uavs,
        nests=nests,
        solver=dict(meta.get("solver") or {}),
        problem=dict(meta.get("problem") or {}),
        cost=dict(meta.get("cost") or {}),
        raw=meta,
    )

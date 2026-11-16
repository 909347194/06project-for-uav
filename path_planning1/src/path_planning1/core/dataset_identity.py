"""数据集身份：建筑/DEM/禁飞区文件内容指纹，给缓存键与场产物校验用。

``ctx.dataset_digest`` 是库内唯一出处，上层不要自己拼 ``world_id``：

* 只看**相对路径 + 文件内容 + 数据口径**（``height_field`` / ``min_area_m2`` /
  ``with_dem`` / ``terrain_min_z_m``，以及存在禁飞区时的 ``nofly_path`` /
  ``nofly_altitude_mode``），不含绝对路径与 mtime：产物可从构建机拷到服务机，
  checkout / 拷贝不会让指纹变化导致缓存永远不命中；
* 用内容哈希而不是 mtime：内容变了一定变，内容没变一定不变；
* 故意**不进** ``params_hash`` / ``field_signature``：那两个指纹负责参数与场身份，
  ``dataset_digest`` 是独立的一段键，由上层拼在缓存键最前面。

文件缺失时不报错（``load_city`` 已在更早处明确抛 ``FileNotFoundError``，这里也兼容
测试桩里的虚拟 root）；此时指纹只由口径参数决定。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..contract.defaults import (
    DEFAULT_NOFLY_ALTITUDE_MODE,
    DEFAULT_NOFLY_PATH,
    MIN_BUILDING_AREA_M2,
    TERRAIN_MIN_Z_M,
)


def dataset_digest(world_kwargs: Mapping[str, Any]) -> str:
    """12 位十六进制：同一份数据 + 同一套口径 ⇒ 同一个值。"""
    root = Path(world_kwargs.get("root", "."))
    with_dem = bool(world_kwargs.get("with_dem", True))
    spec = {
        "with_dem": with_dem,
        "height_field": str(world_kwargs.get("height_field", "Height")),
        "min_area_m2": float(world_kwargs.get("min_area_m2", MIN_BUILDING_AREA_M2)),
        "terrain_min_z_m": float(world_kwargs.get("terrain_min_z_m", TERRAIN_MIN_Z_M)),
    }

    # 禁飞区文件缺省可选：没有它时**不往 spec 里写键**（默认值不进指纹，避免冗余）；
    # 一旦文件存在（或显式配置），路径与内容都进指纹，禁飞区一改产物立即过期。
    nofly_raw = world_kwargs.get("nofly_path")
    nofly_rel = str(nofly_raw) if nofly_raw is not None else DEFAULT_NOFLY_PATH
    nofly_file = Path(nofly_rel)
    if not nofly_file.is_absolute():
        nofly_file = root / nofly_file
    if nofly_raw is not None or nofly_file.is_file():
        spec["nofly_path"] = nofly_rel
        mode = str(world_kwargs.get("nofly_altitude_mode", DEFAULT_NOFLY_ALTITUDE_MODE))
        if mode != DEFAULT_NOFLY_ALTITUDE_MODE:
            spec["nofly_altitude_mode"] = mode

    candidates: list[Path] = [root / "region.geojson"]
    building_dir = root / "building"
    if building_dir.is_dir():
        candidates.extend(sorted(p for p in building_dir.rglob("*") if p.is_file()))
    if with_dem:
        candidates.append(root / "dem.tif")
    if nofly_file.is_file():
        candidates.append(nofly_file)

    files: list[list[Any]] = []
    for path in candidates:
        if not path.is_file():
            continue
        stat = path.stat()
        try:
            label = path.relative_to(root).as_posix()
        except ValueError:                       # 显式给了根目录外的绝对路径
            label = path.name
        files.append([
            label,
            int(stat.st_size),
            hashlib.sha1(path.read_bytes()).hexdigest(),
        ])

    text = json.dumps({"spec": spec, "files": files}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]

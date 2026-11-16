"""全域场产物：构建期算一次落成只读文件，服务启动用 mmap 映射，不再每进程重算。

为什么需要它：``create_context()`` 默认在**进程启动**阶段建场（≈120–140 s、峰值约
4.1 GB、常驻 1.15 GB）。单机单进程没问题，但多副本 / 扩缩容 / 崩溃重启 / Windows 无
fork 的每个 worker 都要再付一遍这笔钱。产物把这一步挪到构建期：同一台机器上的多个
进程 ``mmap`` 同一份只读文件，共享同一份物理内存，启动只剩签名校验（毫秒级）。

契约比文件本身更重要——产物是一个**带身份的快照**，加载时逐项校验：

* ``schema`` 版本；
* ``signature`` = ``field_signature(field)``：格网身份 + 取样补偿量，代码版本换了 bias
  就自动判产物过期；
* ``dataset_digest``：建筑/DEM/禁飞区内容换了 ⇒ 产物过期（禁飞区不在飞行带外也要重算，
  因为产物里的 SDF 已经把它栅格化进去了）；
* ``contract``：``max_od_length_m`` / ``pad_m`` / ``spacing_m`` / 飞行带 / 城市边界，
  配置漂移 ⇒ 启动期直接报错，绝不静默拿旧场接待新契约。

meta.json 最后写（写完才算产物完成）；保存前先删掉旧 meta，任何中断都会让加载端明确
报"产物不完整"，而不是半新半旧地被当成有效场。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..contract import GridSpec
from .world_field import FieldContractError, WorldField, field_signature

FIELD_ARTIFACT_SCHEMA = 1
_SDF_FILE = "sdf.npy"
_OCCUPANCY_FILE = "occupancy.npy"
_META_FILE = "meta.json"


def _canon(value: Any) -> Any:
    """数值归一：JSON 里的 3000 与 Python 里的 3000.0 必须算同一份契约。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 6)
    if isinstance(value, (list, tuple)):
        return [_canon(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _canon(v) for k, v in sorted(value.items())}
    return value


def _same_contract(left: Any, right: Any) -> bool:
    return json.dumps(_canon(left), sort_keys=True) == json.dumps(_canon(right), sort_keys=True)


def field_contract(
    *,
    max_od_length_m: float,
    pad_m: float,
    params: Any,
    bounds_m: tuple[float, float, float, float],
) -> dict[str, Any]:
    """产物身份里的「配置」部分：保存端与加载端共用这一个出处，不写第二份。"""
    return {
        "max_od_length_m": float(max_od_length_m),
        "pad_m": float(pad_m),
        "spacing_m": tuple(float(v) for v in params.spacing_m),
        "agl_min_m": float(params.agl_min_m),
        "agl_max_m": float(params.agl_max_m),
        "bounds_m": tuple(float(v) for v in bounds_m),
    }


def save_field_artifact(
    field: WorldField,
    path: str | Path,
    *,
    dataset_digest: str,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """把场写成一个目录（``sdf.npy`` + ``occupancy.npy`` + ``meta.json``），返回 meta。

    一般由 ``PlanningContext.save_field()`` 调用；后端不需要自己拼 contract。
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    meta_path = directory / _META_FILE
    meta_path.unlink(missing_ok=True)     # 先作废旧产物：中断时加载端会明确报"不完整"

    np.save(directory / _SDF_FILE, field.sdf)
    np.save(directory / _OCCUPANCY_FILE, field.occupancy)
    meta: dict[str, Any] = {
        "schema": FIELD_ARTIFACT_SCHEMA,
        "grid": {
            "origin_enu": [float(v) for v in field.grid.origin_enu],
            "spacing_m": [float(v) for v in field.grid.spacing_m],
            "shape": [int(v) for v in field.grid.shape],
            "yaw_rad": float(field.grid.yaw_rad),
        },
        "signature": field_signature(field),
        "dataset_digest": dataset_digest,
        "contract": _canon(dict(contract)),
        "nbytes": int(field.nbytes),
    }
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8"
    )
    return meta


def load_field_artifact(
    path: str | Path,
    *,
    dataset_digest: str = "",
    contract: Mapping[str, Any] | None = None,
) -> tuple[WorldField, dict[str, Any]]:
    """只读 mmap 载入产物，并在返回前把上面四项契约全部验完。

    任何不一致都抛 ``FieldContractError``（部署事故，别重试请求）。返回数组的
    ``writeable`` 为 False；在线路径只做三线性取样，不会写场。
    """
    directory = Path(path)
    meta_path = directory / _META_FILE
    if not meta_path.is_file():
        raise FieldContractError(
            f"场产物不完整或不存在：{directory / _META_FILE} 缺失；"
            "请先在构建期跑 `plan_route_segment.py --save-field <目录>`"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if int(meta.get("schema", -1)) != FIELD_ARTIFACT_SCHEMA:
        raise FieldContractError(
            f"场产物 schema={meta.get('schema')} 与本库要求的 "
            f"{FIELD_ARTIFACT_SCHEMA} 不符：请用当前版本重新生成产物"
        )
    expected_digest = str(meta.get("dataset_digest", ""))
    if dataset_digest and expected_digest and dataset_digest != expected_digest:
        raise FieldContractError(
            f"建筑/DEM/禁飞区数据版本变了（产物 {expected_digest} / 当前 {dataset_digest}）："
            "场已过期，必须重新建场，不能复用旧产物"
        )
    if contract is not None and not _same_contract(meta.get("contract", {}), dict(contract)):
        raise FieldContractError(
            "建场契约变了（max_od / pad / spacing / 飞行带 / 城市边界之一），产物与当前配置不匹配："
            f"产物 {meta.get('contract')} / 当前 {_canon(dict(contract))}；请重新建场"
        )

    for name in (_SDF_FILE, _OCCUPANCY_FILE):
        if not (directory / name).is_file():
            raise FieldContractError(f"场产物缺文件 {directory / name}：请重新生成产物")
    g = meta["grid"]
    grid = GridSpec(
        origin_enu=tuple(float(v) for v in g["origin_enu"]),
        spacing_m=tuple(float(v) for v in g["spacing_m"]),
        shape=tuple(int(v) for v in g["shape"]),
        yaw_rad=float(g.get("yaw_rad", 0.0)),
    )
    sdf = np.load(directory / _SDF_FILE, mmap_mode="r", allow_pickle=False)
    occupancy = np.load(directory / _OCCUPANCY_FILE, mmap_mode="r", allow_pickle=False)
    try:
        field = WorldField(grid=grid, sdf=sdf, occupancy=occupancy)
    except ValueError as exc:
        raise FieldContractError(f"场产物数组形状与 meta 不一致：{exc}") from exc
    if field_signature(field) != meta.get("signature"):
        raise FieldContractError(
            "场产物签名与内容不一致（文件被替换/损坏，或取样补偿量已改版）：请重新生成产物"
        )
    return field, meta

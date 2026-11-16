"""参数指纹：缓存键的**唯一来源**（``Diagnostics.params_hash`` 由它填）。

为什么单独一个文件：参数散在 ``RouteSegmentParams`` / ``PlannerParams`` / ``VehicleLimits`` /
``RegionOfInterest`` 四处，缓存键必须把它们**全部**纳进来。少纳一项的后果是
"改了净空却命中旧 SDF"，比慢更糟——错得静默。

同理，``defaults.py`` 里的**实现细节常数**（段内采样步长、shortcut 次数、净空采样步长）也会
改变输出，所以它们在 ``PlannerParams`` 里也各有一个字段、随 ``planner`` 一起进指纹——
这样"能从外面配的"和"能进指纹的"是同一批，不会漏。
**数据集级常量**（``MIN_BUILDING_AREA_M2`` / ``TERRAIN_MIN_Z_M``）不在这里：
它们决定的是"世界长什么样"，属于 ``world_id`` / 数据集身份，调用方要一并纳入缓存键。

指纹只保证"同参同值"，不保证"同参同解"（解还依赖 world_id 与 seed；缓存键怎么拼见
``docs/后端怎么用-pp1.md`` §8）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any

from ..contract import PlannerParams, RegionOfInterest, VehicleLimits
from ..contract.defaults import CLEARANCE_SEARCH_RADIUS_M, DEFAULT_SEED

if TYPE_CHECKING:                       # 只为注解；不在运行时把 core 内部依赖绕成环
    from .route_segment_params import RouteSegmentParams


def _canonical(value: Any) -> Any:
    """dataclass / 元组 / 路径 → 可稳定序列化的结构（键序由 ``sort_keys`` 兜住）。"""
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _canonical(v) for k, v in asdict(value).items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, bool):          # bool 是 int 的子类，先摘出去
        return value
    if isinstance(value, (int, float)):
        # 一律按 float 归一：从 JSON 读进来的 `[4,4,2]`（int）与 `[4.0,4.0,2.0]`（默认）必须是
        # 同一个指纹，否则同一份配置算出两个 hash，缓存永远不命中、日志里两个键各说各话。
        return round(float(value), 9)
    return value


def params_hash(
    *,
    planner: PlannerParams,
    vehicle: VehicleLimits,
    roi: RegionOfInterest,
    spacing_m: tuple[float, float, float],
    agl_range_m: tuple[float, float],
    seed: int = DEFAULT_SEED,
    clearance_search_radius_m: float = CLEARANCE_SEARCH_RADIUS_M,
    extra: dict[str, Any] | None = None,
) -> str:
    """12 位十六进制指纹。``extra`` 留给 world_id / frame_id 这类外部键。

    RRT 的实现细节常数（段内采样步长、shortcut 次数、净空采样步长）现在已经是
    ``PlannerParams`` 的字段，随 ``planner`` 一起进指纹，不再单独列一栏——
    这样"能从外面配的"和"能进指纹的"是同一批，不会漏。

    ``clearance_search_radius_m`` 单独列一栏：它不属于算法旋钮（不改路径），但它决定
    交付的 ``min_clearance_m`` 报多少，改了就是另一个结果，缓存不能命中旧值。
    """
    payload = {
        "planner": planner,
        "vehicle": vehicle,
        "roi": roi,
        "spacing_m": spacing_m,
        "agl_range_m": agl_range_m,
        "seed": int(seed),
        "clearance_search_radius_m": float(clearance_search_radius_m),
        "extra": extra or {},
    }
    text = json.dumps(_canonical(payload), ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def digest_for(params: RouteSegmentParams, extra: dict[str, Any] | None = None) -> str:
    """``params_hash`` 的单一调用点：规划器与装配层准入共用，避免两处拼参数拼出不同 hash。

    ``extra`` 传 ``field_signature(world_field)``：场版与窗口现算版的 SDF 基准不同，
    指纹必须能区分，否则缓存会静默命中另一口径的结果；``on_demand`` 传空字典。
    """
    return params_hash(
        planner=params.planner,
        vehicle=params.vehicle,
        roi=params.roi,
        spacing_m=params.spacing_m,
        agl_range_m=(params.agl_min_m, params.agl_max_m),
        seed=params.seed,
        clearance_search_radius_m=CLEARANCE_SEARCH_RADIUS_M,
        extra=extra,
    )

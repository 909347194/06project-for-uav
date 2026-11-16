"""建场契约的派生：pad 与 ``build_world_field`` 入参的唯一出处。

从 ``planning_context`` 拆出来是因为它是**纯派生函数**：给定世界、参数与 OD 上限就能算出
格网口径，不装载文件、不建场、不持有状态。装配层与测试都从这里取同一个答案，
不会出现"装配层算一套、测试算另一套"的漂移。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..contract import defaults
from .corridor_window import required_pad_m

if TYPE_CHECKING:
    from .city_dataset import CityDataset
    from .route_segment_params import RouteSegmentParams


def world_field_kwargs(
    ds: CityDataset,
    params: RouteSegmentParams,
    max_od_length_m: float | None = None,
) -> dict[str, Any]:
    """``build_world_field`` 的入参：外扩按契约派生，不是写死的常量。"""
    length = (
        defaults.WORLD_FIELD_MAX_OD_LENGTH_M if max_od_length_m is None
        else float(max_od_length_m)
    )
    pad_m = required_pad_m(
        params.roi,
        params.spacing_m,
        length,
        extra_m=defaults.WORLD_FIELD_PAD_EXTRA_M,
        min_pad_m=defaults.WORLD_FIELD_PAD_MIN_M,
    )
    return {
        "buildings": ds.buildings,
        "frame": ds.frame,
        "dem": ds.dem,
        "nofly": getattr(ds, "nofly", ()),
        "spacing_m": params.spacing_m,
        "agl_min_m": params.agl_min_m,
        "agl_max_m": params.agl_max_m,
        "bounds_m": ds.bounds_m,
        "pad_m": pad_m,
    }

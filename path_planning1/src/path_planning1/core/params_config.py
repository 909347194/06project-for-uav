"""一份 JSON 配齐全部参数：**缺省即默认、拼错就报错、量值有上下界**。

三件事，别的都不做（不做 HTTP、不做缓存、不做结果序列化）：

1. ``params_from_config``：JSON/字典 → ``RouteSegmentParams``。**不写就是默认值**，所以 ``{}``
   与 ``RouteSegmentParams()`` 等价；只写你要改的键。
2. ``world_kwargs_from_config`` / ``request_from_config``：同一份 JSON 里的 ``world`` /
   ``request`` 段 → ``load_city(**...)`` 与 ``plan_route_segment(...)`` 的入参。
3. ``params_to_config`` / ``default_config``：把对象**连同默认值**全部摊平回 JSON，
   用于日志、前端表单回填、以及"默认到底是多少"这件事只有一个出处。

为什么要有这层：参数散在 4 个 dataclass（``RouteSegmentParams`` / ``RegionOfInterest`` /
``VehicleLimits`` / ``PlannerParams``），后端不想知道它们怎么嵌套；而"少写一个键就用默认"
和"拼错一个键必须报错"是两件相反的要求，只有集中解析才能同时满足。

字段清单**从 dataclass 本身读**（``dataclasses.fields``），不在这里抄第二份——
新增字段不会漏进 JSON，删字段也不会留下幽灵键。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import fields, replace
from typing import Any

from ..contract import PlannerParams, RegionOfInterest, VehicleLimits, defaults
from .route_segment_params import RouteSegmentParams

#: 顶层三段：世界（数据口径）/ 请求（哪一段）/ 参数（怎么算）
_TOP_KEYS = ("world", "request", "params")
_WORLD_KEYS = ("root", "with_dem", "height_field", "min_area_m2", "terrain_min_z_m",
               "field_mode", "max_od_length_m", "nofly_path", "nofly_altitude_mode")
_REQUEST_KEYS = ("start_xy", "goal_xy", "start_z_m", "goal_z_m")
_FIELD_MODES = (defaults.WORLD_FIELD_MODE_STARTUP, defaults.WORLD_FIELD_MODE_ON_DEMAND)


def _names(cls: type) -> set[str]:
    return {f.name for f in fields(cls)}


def _section(cfg: Mapping[str, Any], name: str, allowed: set[str] | tuple[str, ...]) -> dict[str, Any]:
    """取出子段并挡住未知键（拼错一个键就静默用默认值是最坏的配置事故）。"""
    if cfg is None or name not in cfg:
        return {}   # 整段省略 = 全默认
    raw = cfg[name]
    if raw is None:
        raise ValueError(f"配置段 `{name}` 不能为 null（省略整段才表示用默认值）")
    if not isinstance(raw, Mapping):
        raise TypeError(f"配置段 `{name}` 必须是对象，收到 {type(raw).__name__}")
    unknown = set(raw) - set(allowed)
    if unknown:
        raise ValueError(
            f"配置段 `{name}` 有未知键 {sorted(unknown)}；可用键：{sorted(allowed)}"
        )
    return dict(raw)


def _top(cfg: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if cfg is None:
        return {}
    if not isinstance(cfg, Mapping):
        raise TypeError("配置必须是 JSON 对象，收到 " + type(cfg).__name__)
    unknown = set(cfg) - set(_TOP_KEYS)
    if unknown:
        raise ValueError(f"未知顶层键 {sorted(unknown)}；只允许 {list(_TOP_KEYS)}")
    return cfg


def _num(value: Any, key: str, *, lo: float | None = None, hi: float | None = None,
         integer: bool = False) -> float | int:
    """类型与量值的唯一关口：JSON 里 ``4`` 与 ``4.0`` 都收，``\"4\"`` 也收（表单常见）。"""
    try:
        out = int(value) if integer else float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"`{key}` 需要{'整数' if integer else '数值'}，收到 {value!r}") from exc
    # NaN 与任何 lo/hi 比较都为假，Inf 在没有 hi 的字段上也会一路放行，最后流进求解器 /
    # required_pad_m 变成天文数字的格网。有限性必须在这里单独挡一道。
    try:
        finite = math.isfinite(float(out))
    except OverflowError:                       # 超大 int：float() 溢出
        finite = False
    if not finite:
        raise ValueError(f"`{key}` 必须是有限数，收到 {value!r}")
    if lo is not None and out < lo:
        raise ValueError(f"`{key}` 必须 ≥ {lo}，收到 {out}")
    if hi is not None and out > hi:
        raise ValueError(f"`{key}` 必须 ≤ {hi}，收到 {out}")
    return out


def params_from_config(cfg: Mapping[str, Any] | None = None) -> RouteSegmentParams:
    """JSON → ``RouteSegmentParams``。``None`` / ``{}`` 都等于"全默认"。

    例：``{"params": {"vehicle": {"v_max_mps": 15}}}`` 只改巡航速度，其余全部默认。
    """
    base = RouteSegmentParams()
    top = _top(cfg)
    sec = _section(top, "params", _names(RouteSegmentParams))

    roi_raw = _section(sec, "roi", _names(RegionOfInterest))
    ratio = roi_raw.get("width_ratio", base.roi.width_ratio)
    half = roi_raw.get("half_width_m", base.roi.half_width_m)
    roi = replace(
        base.roi,
        width_ratio=None if ratio is None else _num(ratio, "roi.width_ratio", lo=1e-6, hi=1.0),
        half_width_m=None if half is None else _num(half, "roi.half_width_m", lo=1e-6),
        margin_m=_num(roi_raw.get("margin_m", base.roi.margin_m), "roi.margin_m", lo=0.0),
    )
    if roi.width_ratio is None and roi.half_width_m is None:
        raise ValueError(
            "`roi.width_ratio` 与 `roi.half_width_m` 不能同时为 null："
            "那等于零宽走廊（只沿基线算，没有横向腾挪空间）。留空即 0.2（默认）。"
        )

    spacing_raw = sec.get("spacing_m", base.spacing_m)
    if not isinstance(spacing_raw, (list, tuple)) or len(spacing_raw) != 3:
        raise ValueError(f"`spacing_m` 需要 3 个数 (sx, sy, sz)，收到 {spacing_raw!r}")
    spacing = tuple(_num(v, "spacing_m[i]", lo=1e-3) for v in spacing_raw)

    agl_min = _num(sec.get("agl_min_m", base.agl_min_m), "agl_min_m", lo=0.0)
    agl_max = _num(sec.get("agl_max_m", base.agl_max_m), "agl_max_m", lo=0.0)
    if agl_max <= agl_min:
        raise ValueError(f"`agl_max_m`({agl_max}) 必须大于 `agl_min_m`({agl_min})")

    veh_raw = _section(sec, "vehicle", _names(VehicleLimits))
    veh = replace(base.vehicle,
                  v_max_mps=_num(veh_raw.get("v_max_mps", base.vehicle.v_max_mps), "vehicle.v_max_mps", lo=1e-3),
                  a_max_mps2=_num(veh_raw.get("a_max_mps2", base.vehicle.a_max_mps2), "vehicle.a_max_mps2", lo=1e-3),
                  thrust_max_n=_num(veh_raw.get("thrust_max_n", base.vehicle.thrust_max_n), "vehicle.thrust_max_n", lo=1e-3),
                  mass_kg=_num(veh_raw.get("mass_kg", base.vehicle.mass_kg), "vehicle.mass_kg", lo=1e-3),
                  clearance_m=_num(veh_raw.get("clearance_m", base.vehicle.clearance_m), "vehicle.clearance_m", lo=0.0))

    st = sec.get("seed", base.seed)
    plc_raw = _section(sec, "planner", _names(PlannerParams))
    planner = _planner(base.planner, plc_raw)
    return replace(base, roi=roi, spacing_m=spacing, agl_min_m=agl_min, agl_max_m=agl_max,
                   vehicle=veh, seed=int(_num(st, "seed", lo=0, integer=True)), planner=planner)


def _planner(base: PlannerParams, raw: dict[str, Any]) -> PlannerParams:
    """算法那 14 个旋钮的边界（只有会让求解必然失败或算错的才设界）。"""
    d = {k: raw.get(k, getattr(base, k)) for k in _names(PlannerParams)}
    step = _num(d["rrt_step_m"], "planner.rrt_step_m", lo=1e-3)
    min_step = _num(d["rrt_min_step_m"], "planner.rrt_min_step_m", lo=1e-3)
    if min_step > step:
        raise ValueError(f"`rrt_min_step_m`({min_step}) 不能大于 `rrt_step_m`({step})")
    dt_min = _num(d["dt_min_s"], "planner.dt_min_s", lo=1e-9)
    dt_max = _num(d["dt_max_s"], "planner.dt_max_s", lo=1e-9)
    if dt_max <= dt_min:
        raise ValueError(f"`dt_max_s`({dt_max}) 必须大于 `dt_min_s`({dt_min})")
    return replace(
        base,
        rrt_step_m=step,
        rrt_min_step_m=min_step,
        rrt_sdf_step_scale=_num(d["rrt_sdf_step_scale"], "planner.rrt_sdf_step_scale", lo=1e-6),
        rrt_max_iters=int(_num(d["rrt_max_iters"], "planner.rrt_max_iters", lo=1, integer=True)),
        rrt_goal_sample_rate=_num(d["rrt_goal_sample_rate"], "planner.rrt_goal_sample_rate", lo=0.0, hi=1.0),
        rrt_shortcut_iters=int(_num(d["rrt_shortcut_iters"], "planner.rrt_shortcut_iters", lo=0, integer=True)),
        clearance_sample_step_m=_num(d["clearance_sample_step_m"], "planner.clearance_sample_step_m", lo=1e-3),
        rrt_initializations=int(_num(
            d["rrt_initializations"], "planner.rrt_initializations",
            lo=1, hi=defaults.RRT_INITIALIZATIONS_MAX, integer=True,
        )),
        n_resample=int(_num(d["n_resample"], "planner.n_resample", lo=3, integer=True)),
        initial_time_step_s=_num(d["initial_time_step_s"], "planner.initial_time_step_s", lo=1e-6),
        segment_samples=int(_num(d["segment_samples"], "planner.segment_samples", lo=0, integer=True)),
        dt_min_s=dt_min,
        dt_max_s=dt_max,
        ipopt_max_iter=int(_num(d["ipopt_max_iter"], "planner.ipopt_max_iter", lo=1, integer=True)),
    )


def world_kwargs_from_config(cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``world`` 段 → ``load_city(**...)`` 的入参。``root`` 必填（没有默认数据目录）。"""
    raw = _section(_top(cfg), "world", _WORLD_KEYS)
    root = raw.get("root")
    if not root:
        raise ValueError("`world.root` 必填（数据目录，含 region.geojson / building / dem.tif）")
    kwargs: dict[str, Any] = {"root": root}
    if "with_dem" in raw:
        if not isinstance(raw["with_dem"], bool):
            raise ValueError("`world.with_dem` 必须是 true/false")
        kwargs["with_dem"] = raw["with_dem"]
    if "height_field" in raw:
        if not isinstance(raw["height_field"], str):
            raise ValueError("`world.height_field` 必须是字段名字符串（不能是 null）")
        kwargs["height_field"] = raw["height_field"]
    if raw.get("min_area_m2") is not None:
        kwargs["min_area_m2"] = _num(raw["min_area_m2"], "world.min_area_m2", lo=0.0)
    if raw.get("terrain_min_z_m") is not None:
        kwargs["terrain_min_z_m"] = _num(raw["terrain_min_z_m"], "world.terrain_min_z_m")
    # 省略 / null = 读数据目录下默认的 nofly.geojson，文件可没有；显式给了就必须存在。
    nofly_path = raw.get("nofly_path")
    if nofly_path is not None:
        if not isinstance(nofly_path, str) or not nofly_path.strip():
            raise ValueError("`world.nofly_path` 必须是非空字符串（相对数据目录或绝对路径）")
        kwargs["nofly_path"] = nofly_path
    mode = raw.get("nofly_altitude_mode")
    if mode is not None:
        allowed = (defaults.NOFLY_ALTITUDE_ABSOLUTE, defaults.NOFLY_ALTITUDE_AGL)
        if mode not in allowed:
            raise ValueError(f"`world.nofly_altitude_mode` 必须是 {allowed} 之一，收到 {mode!r}")
        kwargs["nofly_altitude_mode"] = mode
    return kwargs


def field_mode_from_config(cfg: Mapping[str, Any] | None = None) -> str:
    """``world.field_mode`` → 全域场建场时机；省略 / 空配置都是 ``startup``。

    这是**服务启动行为**而不是 ``load_city`` 的入参，所以单独解析，不混进
    ``world_kwargs_from_config`` 的返回值。取值只有两个：``startup``（默认，启动阶段
    建一次、在线只取样）与 ``on_demand``（不预建，每条请求窗口现算）。
    """
    raw = _section(_top(cfg), "world", _WORLD_KEYS)
    mode = raw.get("field_mode", defaults.DEFAULT_WORLD_FIELD_MODE)
    if mode is None:
        raise ValueError(f"`world.field_mode` 不能为 null；可用值：{list(_FIELD_MODES)}")
    if mode not in _FIELD_MODES:
        raise ValueError(f"`world.field_mode` 必须是 {list(_FIELD_MODES)} 之一，收到 {mode!r}")
    return mode


def max_od_length_from_config(cfg: Mapping[str, Any] | None = None) -> float:
    """``world.max_od_length_m`` → 建场契约支持的最大 OD 水平长度（米）。

    它同时决定 ``required_pad_m`` 派生的外扩量与请求期准入上限，所以**不是**
    ``load_city`` 的入参（和 ``field_mode`` 一样单独解析）。调大 ⇒ 场更大、更慢、更占内存，
    启动自检会替你验证新 pad 是否真的罩得住；省略 / 空配置都是 3000 m。
    """
    raw = _section(_top(cfg), "world", _WORLD_KEYS)
    value = raw.get("max_od_length_m", defaults.WORLD_FIELD_MAX_OD_LENGTH_M)
    if value is None:
        raise ValueError("`world.max_od_length_m` 不能为 null（省略整段才表示用默认值）")
    return float(_num(value, "world.max_od_length_m", lo=1.0))


def request_from_config(cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``request`` 段 → ``plan_route_segment(...)`` 的位置/kwargs 入参。"""
    raw = _section(_top(cfg), "request", _REQUEST_KEYS)
    for key in ("start_xy", "goal_xy", "start_z_m"):
        if key not in raw:
            raise ValueError(f"`request.{key}` 必填（起终点水平 ENU 米 / 起点绝对 z）")
    xy: dict[str, tuple[float, float]] = {}
    for key in ("start_xy", "goal_xy"):
        val = raw[key]
        if not isinstance(val, (list, tuple)) or len(val) != 2:
            raise ValueError(f"`request.{key}` 需要 [x, y] 两个数，收到 {val!r}")
        xy[key] = (float(_num(val[0], f"request.{key}[0]")), float(_num(val[1], f"request.{key}[1]")))
    out = dict(xy)
    out["start_z_m"] = float(_num(raw["start_z_m"], "request.start_z_m"))
    out["goal_z_m"] = (None if raw.get("goal_z_m") is None
                       else float(_num(raw["goal_z_m"], "request.goal_z_m")))
    return out


#: 摊平 JSON 时的键顺序（"生效的在前"，便于人看；未列出的新字段按字母序补在后面）
_ORDER = {
    "vehicle": ("v_max_mps", "a_max_mps2", "clearance_m", "thrust_max_n", "mass_kg"),
    "planner": ("n_resample", "segment_samples", "dt_min_s", "dt_max_s", "initial_time_step_s",
                "ipopt_max_iter", "rrt_step_m", "rrt_min_step_m", "rrt_sdf_step_scale",
                "rrt_max_iters", "rrt_goal_sample_rate", "rrt_shortcut_iters",
                "clearance_sample_step_m", "rrt_initializations"),
}


def _dump(obj: Any, section: str) -> dict[str, Any]:
    """按既定顺序输出字段（从 dataclass 读，字段集合永远不落后于代码）。"""
    order = [k for k in _ORDER.get(section, ()) if hasattr(obj, k)]
    rest = sorted(k for k in _names(type(obj)) if k not in order)
    return {k: getattr(obj, k) for k in order + rest}


def params_to_config(params: RouteSegmentParams | None = None) -> dict[str, Any]:
    """``RouteSegmentParams`` → **展开全部字段**的 JSON（含默认值）。"""
    p = params or RouteSegmentParams()
    return {
        "roi": {"width_ratio": p.roi.width_ratio, "half_width_m": p.roi.half_width_m,
                "margin_m": p.roi.margin_m},
        "spacing_m": [float(v) for v in p.spacing_m],
        "agl_min_m": p.agl_min_m, "agl_max_m": p.agl_max_m, "seed": int(p.seed),
        "vehicle": _dump(p.vehicle, "vehicle"),
        "planner": _dump(p.planner, "planner"),
    }


def default_config() -> dict[str, Any]:
    """整份 JSON 模板：``world``/``request`` 留空待填，``params`` 全是默认值。

    ``params`` 段可以直接删掉——删掉就是全默认（这是"缺省即默认"的规则）。
    """
    return {
        "world": {"root": None, "with_dem": True, "height_field": "Height",
                  "min_area_m2": defaults.MIN_BUILDING_AREA_M2,
                  "terrain_min_z_m": defaults.TERRAIN_MIN_Z_M,
                  "field_mode": defaults.DEFAULT_WORLD_FIELD_MODE, "max_od_length_m": defaults.WORLD_FIELD_MAX_OD_LENGTH_M,
                  "nofly_path": None, "nofly_altitude_mode": defaults.DEFAULT_NOFLY_ALTITUDE_MODE},
        "request": {"start_xy": None, "goal_xy": None, "start_z_m": None, "goal_z_m": None},
        "params": params_to_config(),
    }

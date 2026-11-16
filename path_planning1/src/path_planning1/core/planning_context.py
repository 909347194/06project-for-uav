"""服务启动上下文：一次装载世界 + 建好**满足契约**的全域场，在线请求只做走廊取样。

上层服务只持有一个 ``PlanningContext``：进程启动时 ``create_context()`` 一次，
之后每条请求调 ``ctx.plan(...)``。"离线一次、在线只取"因此是**结构保证**——
在线路径拿不到"不传场"的入口，不依赖调用方记得把 ``world_field`` 传下去。

这里同时是**建场契约**的唯一实现者：

* 外扩 ``pad_m`` 由 ``required_pad_m(roi, spacing, max_od_length_m)`` 派生并对齐格距；
* ``max_od_length_m`` 按城市对角线封顶（端点都在 ``bounds_m`` 内，更长 OD 不存在），
  防止配一个天文数字把 pad / 内存无界放大；
* 建完场立刻用契约允许的最坏窗口做覆盖自检，不够就**在启动阶段炸**，不留到请求期；
* 请求期只做两件事：端点出城市边界 → ``OUTSIDE_WORLD``；OD 超过契约长度 →
  ``OD_TOO_LONG``。两者都是结果码，不是异常，也不进核心规划。

核心 ``plan_route_segment`` 仍是纯函数（同样入参同样输出）；本模块只是有状态的
装配层：不含 HTTP、缓存淘汰、全局单例，多个上下文可并存（不同世界 / 不同口径）。
``field_mode=on_demand`` 时不建场、不做场覆盖准入，每条请求窗口现算（低内存 / 冷启动调试）。

**部署形态可选**：默认 ``create_context()`` 现场建场；给 ``field_artifact=<目录>`` 则
改为 mmap 只读映射构建期产物（毫秒级启动，同机多进程共享物理内存）。两条路都跑同一份
覆盖自检与契约校验，场身份签名也完全一致，因此 ``params_hash`` 与规划结果不变。
``ctx.save_field(path)`` 用于构建期把刚算好的场落成产物。
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..contract import FailureCode, defaults
from .city_dataset import CityDataset, load_city
from .dataset_identity import dataset_digest
from .field_artifact import field_contract, load_field_artifact, save_field_artifact
from .field_contract import world_field_kwargs
from .params_config import (
    field_mode_from_config,
    max_od_length_from_config,
    params_from_config,
    world_kwargs_from_config,
)
from .params_fingerprint import digest_for
from .route_segment_outcome import RouteSegmentPlan, rejected_plan
from .route_segment_params import RouteSegmentParams
from .route_segment_planner import plan_route_segment
from .world_field import (
    FieldContractError,
    WorldField,
    build_world_field,
    check_contract_coverage,
    field_signature,
)

_FIELD_MODES = (defaults.WORLD_FIELD_MODE_STARTUP, defaults.WORLD_FIELD_MODE_ON_DEMAND)


@dataclass(frozen=True)
class PlanningContext:
    """启动阶段装配好的只读上下文；``plan`` 是唯一的在线入口。

    ``field`` 为 ``None`` 表示 ``on_demand``：此时 ``plan`` 退回窗口现算，且不做
    ``OD_TOO_LONG`` 准入（现算模式不依赖场，任意长度的走廊都能现算）。
    ``load_s`` 是数据装载耗时；``build_s`` 是"取得场"的耗时（现算 ≈120 s / 读产物
    ≈毫秒级），都供服务打点，不进规划结果的 ``timings``。
    ``dataset.nofly`` 是启动期装载的静态禁飞棱柱，场与每条请求都固定取它；
    ``dataset_digest`` 是建筑/DEM/禁飞区内容指纹，上层把它拼在缓存键最前面；
    ``field_source`` 取 ``built`` / ``artifact`` / ``none``，用于区分启动形态。
    """

    dataset: CityDataset
    params: RouteSegmentParams
    field: WorldField | None
    field_mode: str
    bounds_m: tuple[float, float, float, float]
    max_od_length_m: float
    field_pad_m: float
    digest: str = ""
    dataset_digest: str = ""
    field_source: str = "built"
    load_s: float = 0.0
    build_s: float = 0.0

    @property
    def field_ready(self) -> bool:
        return self.field is not None

    def save_field(self, path: str | Path) -> dict[str, Any]:
        """构建期用：把当前场落成只读产物，供其它进程/机器 ``field_artifact=`` 复用。

        后端不需要知道产物里有哪些字段；契约、数据版本、场身份都由上下文自己带上。
        启动时读产物的路径是 ``create_context(cfg, field_artifact=path)``。
        """
        if self.field is None:
            raise FieldContractError(
                "on_demand 模式没有全域场可保存；改用 field_mode=startup 建场后再存"
            )
        return save_field_artifact(
            self.field,
            path,
            dataset_digest=self.dataset_digest,
            contract=field_contract(max_od_length_m=self.max_od_length_m,
                                    pad_m=self.field_pad_m, params=self.params,
                                    bounds_m=self.bounds_m),
        )

    def plan(
        self,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        start_z_m: float,
        *,
        goal_z_m: float | None = None,
    ) -> RouteSegmentPlan:
        """一条航段。``start_z_m`` / ``goal_z_m`` 是**绝对 z**（与 xy 同一高程基准）；
        真高 = z − 地面，由库内部按 DEM 算，调用方不要先减。"""
        rejected = self._admission(start_xy, goal_xy)
        if rejected is not None:
            return rejected
        return plan_route_segment(
            self.dataset.buildings,
            self.dataset.frame,
            self.dataset.dem,
            start_xy,
            goal_xy,
            start_z_m,
            self.params,
            buildings_total=self.dataset.n_buildings,
            goal_z_m=goal_z_m,
            world_field=self.field,
            nofly=getattr(self.dataset, "nofly", ()),
        )

    def _admission(
        self, start_xy: tuple[float, float], goal_xy: tuple[float, float]
    ) -> RouteSegmentPlan | None:
        """准入：只按城市边界与建场契约判，**不判断"场够不够"**（那是启动自检的事）。"""
        x0, y0, x1, y1 = self.bounds_m
        for label, xy in (("起点", start_xy), ("终点", goal_xy)):
            if not (x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1):
                return rejected_plan(
                    FailureCode.OUTSIDE_WORLD,
                    f"{label} ({xy[0]:.1f}, {xy[1]:.1f}) 超出城市边界 "
                    f"[{x0:.1f}, {y0:.1f}]–[{x1:.1f}, {y1:.1f}]（ENU 米）："
                    "城市外没有建筑/DEM 数据，规划无意义",
                    self.digest,
                )
        if self.field_ready:
            length = math.dist(start_xy, goal_xy)
            if length > self.max_od_length_m + 1e-6:
                return rejected_plan(
                    FailureCode.OD_TOO_LONG,
                    f"OD 水平长度 {length:.1f} m 超过建场契约 max_od_length_m="
                    f"{self.max_od_length_m:.1f} m：走廊窗口会超出全域场覆盖。"
                    f"调大 world.max_od_length_m 会同步放大 pad（当前 {self.field_pad_m:.0f} m）"
                    "并增加建场内存/耗时，需在启动阶段重新建场",
                    self.digest,
                )
        return None


def create_context(
    cfg: Mapping[str, Any] | None = None,
    *,
    world_kwargs: Mapping[str, Any] | None = None,
    params: RouteSegmentParams | None = None,
    field_mode: str | None = None,
    max_od_length_m: float | None = None,
    field_artifact: str | Path | None = None,
) -> PlanningContext:
    """启动阶段一次性装配。两种传法二选一：

    * ``create_context(cfg)``：一份 JSON 同时给出 ``world`` / ``params`` / ``field_mode``
      与 ``world.max_od_length_m``；
    * ``create_context(world_kwargs=..., params=..., field_mode=..., max_od_length_m=...)``：
      调用方已自己解析配置。

    ``field_mode`` / ``max_od_length_m`` 显式传入时覆盖配置；不传则读 ``world`` 段。
    ``max_od_length_m`` 会被收敛到城市对角线（见上），返回的 ``ctx.max_od_length_m``
    是**实际生效的契约值**，不是配置里的原始数字。

    ``field_artifact=<目录>`` 只在 ``startup`` 下有效：不现场建场，改为 mmap 只读映射
    ``ctx.save_field()`` 生成的产物。产物的数据版本 / 建场契约 / 场身份都会被校验，
    不一致就在启动期抛 ``FieldContractError``。不给它时行为与以前完全一致。
    """
    if cfg is not None:
        if world_kwargs is not None or params is not None:
            raise ValueError("create_context: 给了 cfg 就不要再给 world_kwargs / params")
        world_kwargs = world_kwargs_from_config(cfg)
        params = params_from_config(cfg)
        mode = field_mode if field_mode is not None else field_mode_from_config(cfg)
        max_od = (
            max_od_length_m if max_od_length_m is not None
            else max_od_length_from_config(cfg)
        )
    else:
        if not world_kwargs:
            raise ValueError("create_context: 没给 cfg 时必须给 world_kwargs（至少 root）")
        params = params if params is not None else RouteSegmentParams()
        mode = field_mode if field_mode is not None else defaults.DEFAULT_WORLD_FIELD_MODE
        max_od = (
            max_od_length_m if max_od_length_m is not None
            else defaults.WORLD_FIELD_MAX_OD_LENGTH_M
        )

    if mode not in _FIELD_MODES:
        raise ValueError(f"`field_mode` 必须是 {list(_FIELD_MODES)} 之一，收到 {mode!r}")
    max_od = float(max_od)
    if not math.isfinite(max_od) or max_od <= 0.0:
        raise ValueError(f"`max_od_length_m` 必须是正的有限数，收到 {max_od!r}")
    if field_artifact is not None and mode == defaults.WORLD_FIELD_MODE_ON_DEMAND:
        raise ValueError("field_artifact 只在 field_mode=startup 下有效；on_demand 不建场、不读产物")

    t0 = time.perf_counter()
    ds = load_city(**world_kwargs)
    load_s = time.perf_counter() - t0
    ds_digest = dataset_digest(world_kwargs)

    # 端点必须在 bounds_m 内 ⇒ OD 水平长度天然 ≤ 城市对角线。配置给更大的值没有意义，
    # 却会让 pad 无界放大（配 100000 m 曾推出 ~23 GB 格网，启动即 OOM）。收敛到对角线，
    # 让「准入上限 / 派生 pad / 启动自检」用同一个数，不出现两套口径。
    bx0, by0, bx1, by1 = (float(v) for v in ds.bounds_m)
    diag = math.hypot(bx1 - bx0, by1 - by0)
    if diag <= 0.0:
        raise ValueError(f"城市 bounds_m 退化（对角线 {diag:.1f} m），无法建立建场契约")
    max_od = min(max_od, diag)

    field: WorldField | None = None
    build_s = 0.0
    pad_m = 0.0
    field_source = "none"
    signature: dict[str, Any] = {}
    if mode == defaults.WORLD_FIELD_MODE_STARTUP:
        kw = world_field_kwargs(ds, params, max_od)
        pad_m = float(kw["pad_m"])
        contract = field_contract(max_od_length_m=max_od, pad_m=pad_m,
                                  params=params, bounds_m=ds.bounds_m)
        t0 = time.perf_counter()
        if field_artifact is not None:
            field, _meta = load_field_artifact(
                field_artifact, dataset_digest=ds_digest, contract=contract
            )
            field_source = "artifact"
        else:
            field = build_world_field(**kw)
            field_source = "built"
        build_s = time.perf_counter() - t0
        # 自检在建场耗时之外：这是配置校验，不是建场成本。两条取场路径都跑，
        # 读产物也不允许跳过——产物可能来自另一套参数。失败必须发生在启动阶段。
        check_contract_coverage(field, ds.bounds_m, params.roi, params.spacing_m, max_od)
        signature = field_signature(field)

    return PlanningContext(
        dataset=ds,
        params=params,
        field=field,
        field_mode=mode,
        bounds_m=tuple(float(v) for v in ds.bounds_m),
        max_od_length_m=max_od,
        field_pad_m=pad_m,
        digest=digest_for(params, signature),
        dataset_digest=ds_digest,
        field_source=field_source,
        load_s=load_s,
        build_s=build_s,
    )

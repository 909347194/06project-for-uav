"""距离代价口径配置：把「用什么尺子估计航程」变成请求级配置，而不是写死的实现。

两档 ``model``：

- ``vertical_section``（**默认**）：垂直切面估计——沿 S-T 连线采样地面与建筑屋顶包络，
  按飞行带与爬升率做地形跟随，得到「可飞航程」的近似（比直线长，比真规划便宜）。
- ``euclid3d``：三维直线，零依赖、零开销；无地形数据或只想要粗略排序时显式选用。

``vertical_section`` 需要地面与建筑包络采样器；核心不读文件，采样器可由调用方注入，
也可用可选 ``task_allocation1.terrain`` 从世界目录现造（不依赖 pp1）：

    ground_z(xs, ys)        -> 每点地面高程（绝对 z）；None = 平地 0 m
    obstacle_top_z(xs, ys)  -> 每点障碍包络顶（绝对 z，建筑屋顶）；None = 无建筑

两个采样器至少给一个，两个都不给直接报错（不偷偷退回欧氏）。

口径参数（语义固定在这里，数值可配）：

- ``agl_min_m`` / ``agl_max_m``：飞行带（**真高**，相对该点地面）。下界同时是巡航
  基准；上界在 ``over_band_mode="infeasible"`` 时成为硬约束（带内飞不过去则该腿不可飞）。
- ``clearance_m``：高于障碍包络的最小竖直净空（楼顶 + 5 m 这类）。
- ``climb_rate``：爬升/下降斜率上限（米 / 米水平）。越小，绕楼越远。
- ``num_samples``：切面采样点数（≥2）。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from . import defaults as D

# 采样器协议：一批水平坐标 → 一批高程（米）。允许返回 list / numpy 数组。
SurfaceSampler = Callable[[Sequence[float], Sequence[float]], Sequence[float]]

_MODELS = ("euclid3d", "vertical_section")
_OVER_BAND_MODES = ("allow", "infeasible")


@dataclass
class CostConfig:
    """距离代价口径（分配阶段的代理尺子）。缺省即默认，未知键忽略。"""

    model: str = D.DEFAULT_COST_MODEL
    agl_min_m: float = D.DEFAULT_COST_AGL_MIN_M
    agl_max_m: float = D.DEFAULT_COST_AGL_MAX_M
    clearance_m: float = D.DEFAULT_COST_CLEARANCE_M
    climb_rate: float = D.DEFAULT_COST_CLIMB_RATE
    num_samples: int = D.DEFAULT_COST_NUM_SAMPLES
    over_band_mode: str = D.DEFAULT_COST_OVER_BAND_MODE

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> CostConfig:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in dict(d or {}).items() if k in known})

    def validate(self) -> None:
        """配置自检：非法值直接报错，不静默钳制（口径错误必须显式暴露）。"""
        if self.model not in _MODELS:
            raise ValueError(f"cost.model={self.model!r} 不认识（只认 {_MODELS}）")
        if self.over_band_mode not in _OVER_BAND_MODES:
            raise ValueError(
                f"cost.over_band_mode={self.over_band_mode!r} 不认识（只认 {_OVER_BAND_MODES}）"
            )
        if not (self.agl_max_m > self.agl_min_m):
            raise ValueError(
                f"cost.agl_max_m={self.agl_max_m} 必须大于 agl_min_m={self.agl_min_m}"
            )
        if self.clearance_m < 0:
            raise ValueError(f"cost.clearance_m={self.clearance_m} 不能为负")
        if not (self.climb_rate > 0):
            raise ValueError(f"cost.climb_rate={self.climb_rate} 必须为正")
        if int(self.num_samples) < 2:
            raise ValueError(f"cost.num_samples={self.num_samples} 必须 ≥ 2")

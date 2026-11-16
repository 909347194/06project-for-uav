"""垂直切面距离尺子（可配置版）：沿 S-T 连线做障碍包络跟随，估计可飞航程。

算法（四步，全部由 :class:`CostConfig` 决定，无写死数值）：

1. 沿 S→T 等距采样 ``num_samples`` 点，累计水平距离 ``xs``；
2. 每点下限 ``floor = max(地面 + agl_min, 包络顶 + clearance)``，
   包络顶 = ``max(地面, 建筑屋顶)``；巡航上限 ``ceil = 地面 + agl_max``；
3. 以下限 floor 为地板、只限制**下降**斜率 ``climb_rate``，分别从起点、终点
   各递推一趟，取两者逐点较大值 —— 得到贴包络走、且两端接得上的剖面；
4. 航程 = ``Σ sqrt(ds² + dz²)``。

与「三维直线」的差别只在有地形/建筑时出现：楼越高、带越低，爬升与下降吃掉的
水平距离越多，估计越长。与真规划器相比它仍只是代理：**只沿连线竖直翻越，
不横向绕行**，所以楼密集时偏低（这也是它只作分配代理、不作交付口径的原因）。

本模块零 I/O：地面与障碍包络一律由调用方注入采样器（``SurfaceSampler``）。
"""

from __future__ import annotations

import math

import numpy as np

from ..contract.cost_config import CostConfig, SurfaceSampler
from ..contract.distance import DistanceFn, euclidean3d


class VerticalSectionCost:
    """垂直切面航程估算器：``estimate(a, b) -> 米``（有向）。"""

    def __init__(
        self,
        config: CostConfig,
        *,
        ground_z: SurfaceSampler | None = None,
        obstacle_top_z: SurfaceSampler | None = None,
    ) -> None:
        config.validate()
        if ground_z is None and obstacle_top_z is None:
            raise ValueError(
                "vertical_section 至少需要一个采样器："
                "ground_z(xs, ys) 给地面高程，obstacle_top_z(xs, ys) 给建筑屋顶包络顶"
            )
        self._cfg = config
        self._ground = ground_z
        self._obstacle = obstacle_top_z

    @property
    def config(self) -> CostConfig:
        return self._cfg

    def profile(
        self, a, b
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """切面剖面：``(xs 累计水平距离, floor 飞行下限, ceil 飞行带上限)``。"""
        n = int(self._cfg.num_samples)
        sx, sy = float(a[0]), float(a[1])
        ex, ey = float(b[0]), float(b[1])
        ts = np.linspace(0.0, 1.0, n)
        px = sx + (ex - sx) * ts
        py = sy + (ey - sy) * ts
        xs = np.concatenate(
            [[0.0], np.cumsum(np.hypot(np.diff(px), np.diff(py)))]
        )
        ground = self._sample(self._ground, px, py, default=0.0)
        if self._obstacle is None:
            envelope = ground
        else:
            envelope = np.maximum(ground, self._sample(self._obstacle, px, py, default=None))
        cfg = self._cfg
        floor = np.maximum(ground + float(cfg.agl_min_m), envelope + float(cfg.clearance_m))
        ceil = ground + float(cfg.agl_max_m)
        return xs, floor, ceil

    def estimate(self, a, b) -> float:
        """有向航程（米）；``over_band_mode="infeasible"`` 且必须超出带时返回 ``inf``。"""
        xs, floor, ceil = self.profile(a, b)
        if self._cfg.over_band_mode == "infeasible" and np.any(floor > ceil + 1e-9):
            return math.inf
        z0 = float(a[2]) if len(a) > 2 else float(floor[0])
        z1 = float(b[2]) if len(b) > 2 else float(floor[-1])
        zs = self._follow(xs, floor, z0, z1)
        return float(np.sum(np.hypot(np.diff(xs), np.diff(zs))))

    def _follow(
        self, xs: np.ndarray, floor: np.ndarray, z0: float, z1: float
    ) -> np.ndarray:
        """贴包络的地形跟随剖面。

        基线 = 两端高度之间的直线；只有被 floor 顶起来才抬升。抬升/回落都用
        ``climb_rate`` 展开（越平缓，同样的升高吃掉的水平距离越多、航程越短），
        所以剖面 = 前向递推与后向递推的逐点较大值：没有障碍的地方**保持基线**，
        不出现无意义的下降-再爬升抖动。
        """
        rate = float(self._cfg.climb_rate)
        ds = np.diff(xs)
        n = len(xs)
        if n == 1:
            return np.array([z0], dtype=np.float64)
        total = float(xs[-1]) if xs[-1] > 0 else 1.0
        base = z0 + (z1 - z0) * (xs / total)
        raw = np.maximum(base, floor)
        # 端点高度按调用方给的值：低于飞行带时由两趟递推把爬升摊开，不在这里硬抬
        raw[0] = z0
        raw[-1] = z1
        fwd = raw.copy()
        for i in range(1, n):
            fwd[i] = max(raw[i], fwd[i - 1] - rate * ds[i - 1])
        bwd = raw.copy()
        for i in range(n - 2, -1, -1):
            bwd[i] = max(raw[i], bwd[i + 1] - rate * ds[i])
        zs = np.maximum(fwd, bwd)
        # 两端高度是调用方给的硬事实：floor 高于端点时按"从这里爬上去"计长度
        zs[0] = z0
        zs[-1] = z1
        return zs

    @staticmethod
    def _sample(
        fn: SurfaceSampler | None,
        xs: np.ndarray,
        ys: np.ndarray,
        *,
        default: float | None,
    ) -> np.ndarray:
        """调采样器并把缺口填平（NaN 线性插值；全 NaN 用 ``default``）。"""
        if fn is None:
            return np.zeros_like(xs) if default is None else np.full_like(xs, float(default))
        vals = np.asarray(fn(xs.tolist(), ys.tolist()), dtype=np.float64).reshape(-1)
        if vals.size != xs.size:
            raise ValueError(
                f"采样器返回 {vals.size} 个数，但请求了 {xs.size} 个点：必须逐点一一对应"
            )
        nan = ~np.isfinite(vals)
        if nan.any():
            good = np.where(~nan)[0]
            if good.size == 0:
                return np.zeros_like(xs) if default is None else np.full_like(xs, float(default))
            vals[nan] = np.interp(np.where(nan)[0], good, vals[good])
        return vals


def build_distance_fn(
    config: CostConfig,
    *,
    ground_z: SurfaceSampler | None = None,
    obstacle_top_z: SurfaceSampler | None = None,
) -> DistanceFn:
    """按配置生成**带记忆化**的距离函数（有序键：切面航程方向相关，禁对称化）。

    ``model="euclid3d"`` 直接返回 :func:`euclidean3d`（无状态、无需缓存）。
    """
    config.validate()
    if config.model == "euclid3d":
        return euclidean3d
    est = VerticalSectionCost(config, ground_z=ground_z, obstacle_top_z=obstacle_top_z)
    memo: dict[tuple[tuple[float, ...], tuple[float, ...]], float] = {}

    def _key(p) -> tuple[float, float, float]:
        return (float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0)

    def distance_fn(a, b) -> float:
        ka, kb = _key(a), _key(b)
        key = (ka, kb)
        hit = memo.get(key)
        if hit is None:
            hit = est.estimate(ka, kb)
            memo[key] = hit
        return hit

    distance_fn.cache = memo  # type: ignore[attr-defined]
    distance_fn.estimator = est  # type: ignore[attr-defined]
    return distance_fn

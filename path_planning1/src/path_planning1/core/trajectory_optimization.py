"""时间域轨迹优化（NLP）：几何路径 → **带时间的可行轨迹**（论文里的 trajectory）。

口径（照 Lin et al. 2021 的 2.4 节，不是参考实现的能耗目标）：

* **目标是"最短飞行时间"**：``T = (N-1)·Δt``，决策变量里只有一个标量 ``Δt``，所以"最小化 Δt"
  就是"最小化总飞行时间"。论文连阻力都忽略（"drag force ... much smaller than lift"），
  所以"能耗最优"不是它的命题；我们没接风场，硬做能耗目标只会得到一个与风无关、说不清物理意义的数。
* **决策变量是时空量**：``x / v / a`` 各 ``N×3`` 加一个 ``Δt``（``n_resample=60`` 时 541 个）。
  节点速度是连续变量 ⇒ 速度矢量连续 ⇒ 空间曲线的切线连续（C¹）——"折线生硬"在这一层被约束消掉，
  不是靠事后套样条。
* **可行性在这里第一次出现**：``‖v‖≤v_max``、``‖a‖≤a_max``、``‖a+g·ê_z‖≤T_max/m``
  （推力要托住重力，见 ``_solve_once``）、``dt∈[dt_min,dt_max]``，
  再加**净空**（SDF 在每个节点、每段内插 ``segment_samples`` 个点上都 ≥ 净空+容差）、
  **走廊横向半宽**（``|n·(x-start)|≤half_width``，交付物口径）和**窗口硬边界**（索引必须落在窗口内）。
* **两端悬停**：``v_0 = v_N = 0``。航段端点是空中航段端点/交接点，"以 20 m/s 冲过终点"当解没有意义。

为什么不用论文的"沿固定路径做前进式速度剖面"：那条路要求路径本身先被 Dubins 平滑成 C¹，
否则每个 90° 折角处速度必须瞬间归零，解退化成"每个拐角停一下"。我们让 ``x`` 也在走廊里自由，
等于把"平滑"与"时间最优"合成一个问题解，省掉 Dubins 这一层，也不再有"平滑后是否还避障"的两套口径。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

try:  # casadi 是**可选依赖**（``pip install "path_planning1[nlp]"``）：没装时几何层照常用
    import casadi as ca
except ImportError:  # pragma: no cover - 只在瘦装环境触发
    ca = None

import numpy as np

from ..contract import Trajectory, defaults
from .occupancy_sdf import SdfField
from .sdf_casadi import SdfOnGrid
from .trajectory_sampling import initial_guess, resample_path

_MAX_ITER_STATUS = "Maximum_Iterations_Exceeded"   # IPOPT 唯一的"再给点预算就能解"的失败态


@dataclass(frozen=True)
class NlpConfig:
    """NLP 的全部旋钮（数值来自 ``contract/defaults.py``，这里只搬运）。"""

    n_nodes: int
    segment_samples: int
    dt_min_s: float
    dt_max_s: float
    initial_dt_s: float
    max_iter: int
    v_max_mps: float
    a_max_mps2: float
    clearance_m: float          # 已含 SDF 量化容差（调用方算好，不在这里二次加）
    thrust_max_n: float = defaults.DEFAULT_THRUST_MAX_N
    mass_kg: float = defaults.DEFAULT_MASS_KG
    gravity_mps2: float = defaults.GRAVITY_MPS2
    # 走廊横向硬约束（交付物口径）：|normal·(x - pivot)| ≤ half_width_m。
    # None = 不约束（只用作计算域），只有显式给半宽时才约束轨迹。
    half_width_m: float | None = None
    corridor_normal_xy: tuple[float, float] | None = None
    corridor_pivot_xy: tuple[float, float] | None = None

    @property
    def specific_thrust_mps2(self) -> float:
        """单位质量可用推力 ``T_max/m``（N/kg = m/s²）——推力约束在这里换算成加速度尺度。"""
        return float(self.thrust_max_n) / float(self.mass_kg)

    @classmethod
    def from_planner(
        cls, planner, vehicle, *, clearance_m: float,
        half_width_m: float | None = None,
        corridor_normal_xy: tuple[float, float] | None = None,
        corridor_pivot_xy: tuple[float, float] | None = None,
    ) -> NlpConfig:
        """从 ``PlannerParams`` + ``VehicleLimits`` 派生（不重复定义默认值）。"""
        return cls(
            n_nodes=int(planner.n_resample),
            segment_samples=int(planner.segment_samples),
            dt_min_s=float(planner.dt_min_s),
            dt_max_s=float(planner.dt_max_s),
            initial_dt_s=float(planner.initial_time_step_s),
            max_iter=int(planner.ipopt_max_iter),
            v_max_mps=float(vehicle.v_max_mps),
            a_max_mps2=float(vehicle.a_max_mps2),
            clearance_m=float(clearance_m),
            thrust_max_n=float(vehicle.thrust_max_n),
            mass_kg=float(vehicle.mass_kg),
            half_width_m=None if half_width_m is None else float(half_width_m),
            corridor_normal_xy=corridor_normal_xy,
            corridor_pivot_xy=corridor_pivot_xy,
        )


@dataclass
class NlpResult:
    """求解结果。``trajectory is None`` 时其余字段说明**为什么没有**（不抛异常）。"""

    trajectory: Trajectory | None
    solver_status: str
    iterations: int | None
    elapsed_s: float
    message: str = ""


def optimize_trajectory(
    path_enu: np.ndarray, field: SdfField, cfg: NlpConfig, *,
    warm: Trajectory | None = None,
) -> NlpResult:
    """在走廊内解"最短时间"轨迹。失败（无解/迭代上限/求解器异常）返回 ``trajectory=None``。

    这里**不**把整条航段判成不可行：几何层已经成功时，NLP 失败只降级（没有时长与速度剖面），
    因为"几何可飞、时间剖面没解出来"和"这条航段飞不过去"是两件事。

    **三级退让**（只换初值/换离散/换预算，不换模型）：

    1. **走廊冷启动失败 → 无走廊版当种子**：带半宽约束冷启动时 IPOPT 可能退化（约束违反停在
       1e-3 量级，预算加到 800 次也不动），而无走廊版 72 次就收敛。只在冷启动失败时做，正常航段
       不多花一次求解。
    2. **节点数 60→40→30**：窄走廊实测 60 节点 200 次不收敛，40 节点 41 次就收敛，时长只差 0.2%
       （149.4 vs 149.2 s）；代价是转弯粒度变粗（节点间距 = 航段/(n-1)）。
    3. **迭代预算 ×3（只在"撞迭代上限"时）**：慢机会吃满预算（``v_max=12`` 在 60 节点要 573 次、
       v=10 要 230 次、v=20 只要 86 次：越慢时间尺度越大、条件数越差）。失败原因是"真的不可行"时
       这一级直接跳过，不白烧时间。

    退让过程写进 ``message``（进 ``warnings``），不做对调用方不可见的静默降级。
    """
    if ca is None:
        return NlpResult(None, "casadi_missing", None, 0.0,
                         "未安装 casadi：pip install \"path_planning1[nlp]\" 后重跑")
    sdf = SdfOnGrid(field)
    n0 = int(cfg.n_nodes)
    elapsed = 0.0
    base = resample_path(path_enu, n0)

    # 第一手：冷启动。带走廊时它可能把 IPOPT 拖进退化（实测约束违反停在 1e-3 量级、
    # 预算抬到 800 次也不收敛，去掉走廊 72 次就收敛）——这时先解无走廊版，再用它热启动。
    r = _solve_once(base, sdf, cfg, n0, cfg.max_iter, warm)
    elapsed += r.elapsed_s
    if r.trajectory is None and cfg.half_width_m is not None:
        plain = replace(cfg, half_width_m=None, corridor_normal_xy=None, corridor_pivot_xy=None)
        seed = _solve_once(base, sdf, plain, n0, cfg.max_iter)
        elapsed += seed.elapsed_s
        if seed.trajectory is not None:
            warm_r = _solve_once(base, sdf, cfg, n0, cfg.max_iter, seed.trajectory)
            elapsed += warm_r.elapsed_s
            if warm_r.trajectory is not None:
                warm_r.message = "走廊约束冷启动不收敛（IPOPT 退化），用无走廊解热启动才收敛"
                warm_r.elapsed_s = elapsed
                return warm_r
    if r.trajectory is not None:
        r.elapsed_s = elapsed
        return r

    hit_iter_limit = r.solver_status == _MAX_ITER_STATUS
    last = r
    for n in _ladder(n0):
        r = _solve_once(resample_path(path_enu, n), sdf, cfg, n, cfg.max_iter, warm)
        elapsed += r.elapsed_s
        r.elapsed_s = elapsed
        if r.trajectory is not None:
            r.message = (f"节点数从 {n0} 退到 {n} 才在 {cfg.max_iter} 次迭代内收敛"
                         f"（粗离散换收敛；时长几乎不变，转弯粒度变粗）")
            return r
        hit_iter_limit = hit_iter_limit or r.solver_status == _MAX_ITER_STATUS
        last = r
    if not hit_iter_limit:
        return last

    r = _solve_once(base, sdf, cfg, n0, 3 * cfg.max_iter, warm)
    elapsed += r.elapsed_s
    r.elapsed_s = elapsed
    if r.trajectory is not None:
        r.message = (f"把 ipopt_max_iter 从 {cfg.max_iter} 放大到 {3 * cfg.max_iter} 才收敛"
                     f"（慢机/病态尺度；想省掉试错就直接把该参数调大）")
    return r


def _ladder(n0: int) -> tuple[int, ...]:
    """节点阶梯：60 → 40 → 30（等比缩，且**只能比 n0 小**）。

    ``n_resample`` 允许小到 3；旧写法用常数下限 12/8 会把 3 个节点"退让"成 12 个，
    越退越细，与"粗离散换收敛"相反。小 n0 时阶梯为空，直接走第三级加预算。
    """
    return tuple(
        n for n in (max(round(n0 * 2 / 3), 12), max(round(n0 / 2), 8))
        if 3 <= n < n0
    )


def _solve_once(
    path: np.ndarray, sdf: SdfOnGrid, cfg: NlpConfig, n: int, max_iter: int,
    warm: Trajectory | None = None,
) -> NlpResult:
    """固定节点数 ``n`` 与迭代上限 ``max_iter`` 解一次（``path`` 已重采样到 ``n`` 点）。

    ``warm`` 给了就按它的位姿/时间步做初值，否则由折线猜（见 ``trajectory_sampling``）。
    """
    t0 = time.perf_counter()
    vmax2, amax2 = cfg.v_max_mps ** 2, cfg.a_max_mps2 ** 2

    opti = ca.Opti()
    x = opti.variable(3, n)
    v = opti.variable(3, n)
    a = opti.variable(3, n)
    dt = opti.variable()

    opti.subject_to(opti.bounded(cfg.dt_min_s, dt, cfg.dt_max_s))
    for i in range(n):
        opti.subject_to(ca.dot(v[:, i], v[:, i]) <= vmax2)
        opti.subject_to(ca.dot(a[:, i], a[:, i]) <= amax2)

    # ---- 推力：托住重力之后再谈加速度，所以约束的是 ‖a + g·ê_z‖（力平衡 ‖T‖ = m‖a+g‖）。
    # a_z ≥ -g：推力只能朝上。没有这两条，mass_kg / thrust_max_n 就是"配了不生效"的死参数。
    specific2 = cfg.specific_thrust_mps2 ** 2
    g = float(cfg.gravity_mps2)
    for i in range(n):
        opti.subject_to(ca.dot(a[:, i], a[:, i]) + 2.0 * g * a[2, i] + g * g <= specific2)
        opti.subject_to(a[2, i] >= -g)

    # ---- 走廊横向：半宽是**交付物**的约束，用 ENU 线性式而不是绑栅格索引（换间距/外扩就不漂）----
    if cfg.half_width_m is not None and cfg.corridor_normal_xy and cfg.corridor_pivot_xy:
        cn_x, cn_y = (float(v) for v in cfg.corridor_normal_xy)
        cp_x, cp_y = (float(v) for v in cfg.corridor_pivot_xy)
        lateral = cn_x * (x[0, :] - cp_x) + cn_y * (x[1, :] - cp_y)
        opti.subject_to(opti.bounded(-cfg.half_width_m, lateral, cfg.half_width_m))

    for i in range(n - 1):
        opti.subject_to(x[:, i + 1] == x[:, i] + 0.5 * (v[:, i] + v[:, i + 1]) * dt)
        opti.subject_to(v[:, i + 1] == v[:, i] + 0.5 * (a[:, i] + a[:, i + 1]) * dt)
    opti.subject_to(x[:, 0] == ca.DM(path[0]))
    opti.subject_to(x[:, n - 1] == ca.DM(path[-1]))
    opti.subject_to(v[:, 0] == 0)
    opti.subject_to(v[:, n - 1] == 0)

    ii, jj, kk = sdf.index_of(x)
    nx, ny, nz = sdf.shape
    opti.subject_to(opti.bounded(0.0, ii, nx - 1.0))
    opti.subject_to(opti.bounded(0.0, jj, ny - 1.0))
    opti.subject_to(opti.bounded(0.0, kk, nz - 1.0))
    for i in range(n):
        opti.subject_to(sdf.at(x[:, i]) >= cfg.clearance_m)
    for k in range(1, cfg.segment_samples + 1):
        alpha = float(k) / float(cfg.segment_samples + 1)
        for i in range(n - 1):
            xm = (1.0 - alpha) * x[:, i] + alpha * x[:, i + 1]
            opti.subject_to(sdf.at(xm) >= cfg.clearance_m)

    opti.minimize(dt)
    if warm is not None and len(warm.position_enu) == n:
        ig_x, ig_v = warm.position_enu, warm.velocity_enu
        ig_a, ig_dt = warm.acceleration_enu, warm.dt_uniform_s
    else:
        ig_v, ig_a, ig_dt = initial_guess(path, cfg)
        ig_x = path
    opti.set_initial(x, ig_x.T)
    opti.set_initial(v, ig_v.T)
    opti.set_initial(a, ig_a.T)
    opti.set_initial(dt, ig_dt)
    opti.solver(
        "ipopt",
        {"print_time": False,
         "ipopt": {"max_iter": max_iter, "print_level": 0, "sb": "yes", "tol": 1e-4}},
    )

    try:
        opti.solve()
    except Exception as exc:  # noqa: BLE001 — 求解器不只抛 RuntimeError，一律降级
        try:
            stats = opti.stats()  # 求解中途失败时 stats() 本身也可能不可用
            status = str(stats.get("return_status", "solve_failed"))
            iters = int(stats.get("iter_count", 0) or 0)
        except Exception:  # noqa: BLE001 — 同上，拿不到统计就记 solve_failed
            status, iters = "solve_failed", 0
        return NlpResult(
            None, status, iters, time.perf_counter() - t0,
            f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
        )

    stats = opti.stats()
    dt_val = float(opti.value(dt))
    return NlpResult(
        Trajectory(
            t_s=np.arange(n, dtype=float) * dt_val,
            position_enu=np.asarray(opti.value(x), dtype=float).T,
            velocity_enu=np.asarray(opti.value(v), dtype=float).T,
            acceleration_enu=np.asarray(opti.value(a), dtype=float).T,
            dt_uniform_s=dt_val,
        ),
        str(stats.get("return_status", "Solve_Succeeded")),
        int(stats.get("iter_count", 0) or 0),
        time.perf_counter() - t0,
    )

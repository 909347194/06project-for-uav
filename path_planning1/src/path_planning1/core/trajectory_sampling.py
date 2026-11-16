"""轨迹离散与初值：把几何折线变成"定维数的初值"（NLP 之前的最后一道预处理）。

两件事都不属于 NLP 模型本身：

* ``resample_path``：按弧长等距重采样到固定节点数。RRT 原始节点数随环境漂移（实测 3–40 点），
  决策变量维度不定就没法比多初值；论文 F.3 的"固定 N=60"就是这一步。
* ``initial_guess``：由折线猜 ``v`` / ``a`` / ``Δt``——先按"别超速"定 dt，再核对加速度上限。

单独成文件是为了让 ``trajectory_optimization`` 只装模型与求解策略（行数上限 300）。
"""

from __future__ import annotations

import numpy as np


def resample_path(path_enu: np.ndarray, n_nodes: int) -> np.ndarray:
    """按**弧长**等距重采样到 ``n_nodes`` 个点（含两端）。

    一维线性插值够用——这里只需要一个落在走廊里的**初值**，形状由 NLP 决定。
    """
    path = np.asarray(path_enu, dtype=float)
    if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2:
        raise ValueError(f"path_enu 必须是 (K≥2, 3)，收到 {path.shape}")
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total <= 0.0:
        raise ValueError("path_enu 长度为 0，无法重采样")
    target = np.linspace(0.0, total, int(n_nodes))
    return np.stack([np.interp(target, s, path[:, d]) for d in range(3)], axis=1)


def initial_guess(path: np.ndarray, cfg) -> tuple[np.ndarray, np.ndarray, float]:
    """由重采样后的折线猜 ``v`` / ``a`` / ``Δt``（中心差分；dt 不够就按 1.5 倍放大，最多 8 轮）。"""
    dt = max(
        cfg.initial_dt_s,
        float(np.max(np.linalg.norm(np.diff(path, axis=0), axis=1)) / cfg.v_max_mps),
    )
    for _ in range(8):
        v = np.zeros_like(path)
        v[1:-1] = (path[2:] - path[:-2]) / (2.0 * dt)
        a = np.zeros_like(path)
        a[1:-1] = (v[2:] - v[:-2]) / (2.0 * dt)
        if (np.max(np.linalg.norm(v, axis=1)) <= cfg.v_max_mps
                and np.max(np.linalg.norm(a, axis=1)) <= cfg.a_max_mps2):
            break
        dt *= 1.5
    return v, a, dt

"""RRT 多初值候选：真几何评估 + 选一条 NLP 初值。

多初值不是替 NLP 找最终最优，而是给它一个更稳的起点。硬过滤是
``clearance_m``（交付净空），排序按“NLP 好不好解”优先于“几何更短”：

1. 折线在走廊半宽内（NLP 的半宽是硬约束）；
2. 真几何净空达到 NLP 实际阈值（避免初值一开始就在约束外）；
3. 几何长度更短（最短时间目标的下界代理）；
4. 真净空更大、路径点更少（同长度时更稳、更好解）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .clearance_check import BuildingPrisms
from .corridor_window import CorridorWindow
from .rrt_path_search import RRTResult

_EPS = 1e-9


@dataclass(frozen=True)
class RRTCandidate:
    """一条 RRT 候选，带上选优需要的真几何量。"""

    result: RRTResult
    length_m: float
    clearance_m: float
    worst_pt: np.ndarray
    max_lateral_m: float
    inside_corridor: bool
    meets_nlp_clearance: bool


def evaluate_candidates(
    results: list[RRTResult],
    prisms: BuildingPrisms,
    window: CorridorWindow,
    *,
    sample_step_m: float,
    nlp_need_m: float,
) -> list[RRTCandidate]:
    """逐条量真几何净空；RRT 只保证 SDF，这里是交付口径。"""
    out: list[RRTCandidate] = []
    for result in results:
        path = result.path_enu
        length_m = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        clearance_m, worst_pt = prisms.worst(path, sample_step_m)
        max_lateral_m = float(window.lateral_of(path).max())
        out.append(
            RRTCandidate(
                result=result,
                length_m=length_m,
                clearance_m=float(clearance_m),
                worst_pt=worst_pt,
                max_lateral_m=max_lateral_m,
                inside_corridor=bool(max_lateral_m <= window.half_width_m + 1e-6),
                meets_nlp_clearance=bool(clearance_m >= nlp_need_m - _EPS),
            )
        )
    return out


def select_candidate(
    candidates: list[RRTCandidate],
    *,
    clearance_m: float,
) -> tuple[RRTCandidate | None, int]:
    """先按交付净空硬过滤，再按 NLP 初值优先级选一条；返回 (选中, 可行数)。"""
    feasible = [c for c in candidates if c.clearance_m >= clearance_m - _EPS]
    if not feasible:
        return None, 0
    return min(feasible, key=_rank), len(feasible)


def _rank(candidate: RRTCandidate) -> tuple[int, int, float, float, int]:
    return (
        0 if candidate.inside_corridor else 1,
        0 if candidate.meets_nlp_clearance else 1,
        candidate.length_m,
        -candidate.clearance_m,
        len(candidate.result.path_enu),
    )

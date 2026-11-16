"""多初值选优的排序契约：先硬过滤，再按走廊/净空/长度选一条。"""

from __future__ import annotations

import numpy as np
from path_planning1.core.rrt_candidates import RRTCandidate, select_candidate
from path_planning1.core.rrt_path_search import RRTResult


def _candidate(
    length_m: float,
    clearance_m: float,
    *,
    inside_corridor: bool = True,
    meets_nlp_clearance: bool = True,
) -> RRTCandidate:
    path = np.array([[0.0, 0.0, 0.0], [length_m, 0.0, 0.0]])
    result = RRTResult(
        path_enu=path,
        iterations=0,
        nodes=2,
        path_points_before_shortcut=2,
    )
    return RRTCandidate(
        result=result,
        length_m=length_m,
        clearance_m=clearance_m,
        worst_pt=path[0],
        max_lateral_m=0.0 if inside_corridor else 999.0,
        inside_corridor=inside_corridor,
        meets_nlp_clearance=meets_nlp_clearance,
    )


def test_select_candidate_uses_hard_clearance_filter() -> None:
    selected, feasible_count = select_candidate(
        [_candidate(100.0, 4.9)], clearance_m=5.0
    )
    assert selected is None
    assert feasible_count == 0


def test_select_candidate_prefers_rank_order_over_shortest_length() -> None:
    candidates = [
        _candidate(100.0, 9.0, inside_corridor=False),          # 出走廊
        _candidate(120.0, 6.0, meets_nlp_clearance=False),      # 在走廊、NLP 净空不够
        _candidate(300.0, 8.0),                                  # 在走廊、NLP 净空够
        _candidate(250.0, 6.0),                                  # 更短，且 NLP 净空够
    ]
    selected, feasible_count = select_candidate(candidates, clearance_m=5.0)
    assert feasible_count == 4
    assert selected is candidates[3]


def test_select_candidate_uses_clearance_when_length_ties() -> None:
    shorter = _candidate(200.0, 6.0)
    safer = _candidate(200.0, 9.0)
    selected, _ = select_candidate([shorter, safer], clearance_m=5.0)
    assert selected is safer

"""默认值唯一来源守卫：契约 dataclass 的默认值 == defaults.py 常量。"""

from __future__ import annotations

from task_allocation1.contract import defaults as D
from task_allocation1.contract.config import ProblemConfig, SolverConfig
from task_allocation1.contract.cost_config import CostConfig


def test_cost_config_defaults_match_constants() -> None:
    c = CostConfig()
    assert c.model == D.DEFAULT_COST_MODEL
    assert c.agl_min_m == D.DEFAULT_COST_AGL_MIN_M
    assert c.agl_max_m == D.DEFAULT_COST_AGL_MAX_M
    assert c.clearance_m == D.DEFAULT_COST_CLEARANCE_M
    assert c.climb_rate == D.DEFAULT_COST_CLIMB_RATE
    assert c.num_samples == D.DEFAULT_COST_NUM_SAMPLES
    assert c.over_band_mode == D.DEFAULT_COST_OVER_BAND_MODE


def test_cost_config_from_dict_ignores_unknown_keys() -> None:
    c = CostConfig.from_dict({"agl_min_m": 30.0, "bogus": 1})
    assert c.agl_min_m == 30.0
    assert c.model == D.DEFAULT_COST_MODEL


def test_problem_config_defaults_match_constants() -> None:
    p = ProblemConfig()
    assert p.w_penalty == D.DEFAULT_W_PENALTY
    assert p.w_violation == D.DEFAULT_W_VIOLATION
    assert p.lambda_cost == D.DEFAULT_LAMBDA_COST
    assert p.hard_violation_kappa == D.DEFAULT_HARD_VIOLATION_KAPPA
    assert p.default_service_time == D.DEFAULT_DEFAULT_SERVICE_TIME
    assert p.current_time == D.DEFAULT_CURRENT_TIME


def test_solver_config_defaults_match_constants() -> None:
    s = SolverConfig()
    assert s.pop_size == D.DEFAULT_POP_SIZE
    assert s.max_generations == D.DEFAULT_MAX_GENERATIONS
    assert s.zeta == D.DEFAULT_ZETA
    assert s.seed == D.DEFAULT_SEED
    assert s.participating_policy == D.DEFAULT_PARTICIPATING_POLICY
    assert s.repair_max_passes == D.DEFAULT_REPAIR_MAX_PASSES
    assert s.cost_polish_eps == D.DEFAULT_COST_POLISH_EPS
    assert s.assign_perturb_factor == D.DEFAULT_ASSIGN_PERTURB_FACTOR
    assert s.tour_perturb_factor == D.DEFAULT_TOUR_PERTURB_FACTOR


def test_from_dict_ignores_unknown_keys() -> None:
    p = ProblemConfig.from_dict({"lambda_cost": 0.5, "unknown": 1})
    assert p.lambda_cost == 0.5
    s = SolverConfig.from_dict({"seed": 9, "bogus": True})
    assert s.seed == 9

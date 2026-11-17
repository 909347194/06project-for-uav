"""配置：约束开关（ConstraintConfig）、问题/评价权重（ProblemConfig）、求解预算（SolverConfig）。

只放数据类与 ``from_dict``（缺省即默认、未知键忽略）；不实现任何算法、不读文件。
默认值唯一来源见 ``defaults.py``，本文件引用而不复制。

**契约边界**：本库是单周期、同巢起降、单航段分配内核，只判 7 类硬约束
（+ 恒开的同巢约束），不存在时序 / 同步 / 覆盖开关类配置。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from . import defaults as D


@dataclass
class ConstraintConfig:
    """7 类硬约束的独立开关。

    ``home_return`` 是恒开硬约束：终点机巢恒 = 起飞巢，只读恒 True、无关闭入口。
    """

    duplicate_task: bool = True
    range: bool = True
    endurance: bool = True
    time_window: bool = True
    nest_capacity: bool = True
    nest_availability: bool = True
    end_nest_reachability: bool = True

    @property
    def home_return(self) -> bool:
        """同巢起降硬约束（只可读；恒 True，不提供关闭入口）。"""
        return True

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> ConstraintConfig:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in dict(d or {}).items() if k in known})


@dataclass
class ProblemConfig:
    """评价权重与惩罚参数（无覆盖 / 选择性开关）。"""

    w_penalty: float = D.DEFAULT_W_PENALTY
    w_violation: float | None = D.DEFAULT_W_VIOLATION
    w_hard_violation: float | None = None
    w_range_excess: float | None = None
    hard_violation_kappa: float = D.DEFAULT_HARD_VIOLATION_KAPPA
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    lambda_cost: float = D.DEFAULT_LAMBDA_COST
    big_m: float | None = None
    default_service_time: float = D.DEFAULT_DEFAULT_SERVICE_TIME
    w_skip_violation: float | None = None
    current_time: float = D.DEFAULT_CURRENT_TIME

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> ProblemConfig:
        src = dict(d or {})
        if isinstance(src.get("constraints"), Mapping):
            src["constraints"] = ConstraintConfig.from_dict(src["constraints"])
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in src.items() if k in known})


@dataclass
class SolverConfig:
    """求解预算与开关：选机 / DMDE / 修复链 / 交付后 polish。"""

    pop_size: int = D.DEFAULT_POP_SIZE
    max_generations: int = D.DEFAULT_MAX_GENERATIONS
    zeta: int = D.DEFAULT_ZETA
    extinction_delta: float | None = D.DEFAULT_EXTINCTION_DELTA
    seed: int | None = D.DEFAULT_SEED
    participating_policy: str = D.DEFAULT_PARTICIPATING_POLICY
    demand_range_factor: float = D.DEFAULT_DEMAND_RANGE_FACTOR
    max_participating_uavs: int | None = None
    repair_max_passes: int = D.DEFAULT_REPAIR_MAX_PASSES
    polish_enabled: bool = True
    polish_rounds: int = D.DEFAULT_POLISH_ROUNDS
    cost_polish_eps: float = D.DEFAULT_COST_POLISH_EPS
    assign_perturb_factor: float = D.DEFAULT_ASSIGN_PERTURB_FACTOR
    tour_perturb_factor: float = D.DEFAULT_TOUR_PERTURB_FACTOR
    perturb_temp_gate: float = D.DEFAULT_PERTURB_TEMP_GATE

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> SolverConfig:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in dict(d or {}).items() if k in known})

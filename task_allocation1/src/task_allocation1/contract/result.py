"""结果：单周期分配输出 + 诊断量。纯数据，可安全序列化（无函数字段）。

同巢起降：``home_return_ok`` 直接来自 solution；``return_leg_m`` / ``path_m``
是事后核算的诊断量，不参与搜索。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .enums import ModelType
from .solution import AllocationSolution


@dataclass
class AllocateResult:
    solution: AllocationSolution
    participating_uav_ids: list[int] = field(default_factory=list)
    model_type: ModelType = ModelType.EMPTY
    n_evals: int = 0
    cost_history: list[float] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def home_return_ok(self) -> bool:
        return self.solution.home_return_ok()

    def metrics_dict(self) -> dict[str, Any]:
        """扁平指标字典，供上层直接序列化。"""
        return {
            "n_participating": len(self.participating_uav_ids),
            "model_type": self.model_type.value,
            "n_evals": self.n_evals,
            "n_executed": len(self.solution.executed_task_ids),
            "path_m": float(self.solution.meta.get("path_m", self.solution.f2_cost)),
            "return_leg_m": float(self.solution.meta.get("return_leg_m", 0.0)),
            "f1_reward": float(self.solution.f1_reward),
            "f2_cost": float(self.solution.f2_cost),
            "feasible": bool(self.solution.feasible),
            "home_return_ok": self.home_return_ok,
        }

"""`docs/ta1_full_config.example.json` 与代码默认值不许漂移。

示例文件是"默认口径清单"：任何一段的键名写错、或默认值改了没同步，
本测试都会红。JSONC 阅读版不参与校验（给人看的）。
"""

from __future__ import annotations

import json
from pathlib import Path

from task_allocation1 import CostConfig, ProblemConfig, SolverConfig

REPO = Path(__file__).resolve().parents[2]
CFG_PATH = REPO / "docs" / "ta1_full_config.example.json"


def _load() -> dict:
    return json.loads(CFG_PATH.read_text(encoding="utf-8"))


def test_example_exists_and_parses() -> None:
    cfg = _load()
    assert set(cfg) == {"solver", "problem", "cost"}


def test_sections_only_use_known_keys() -> None:
    cfg = _load()
    assert set(cfg["solver"]) <= set(SolverConfig.__dataclass_fields__)
    assert set(cfg["problem"]) <= set(ProblemConfig.__dataclass_fields__)
    assert set(cfg["cost"]) <= set(CostConfig.__dataclass_fields__)
    assert set(cfg["problem"]["constraints"]) <= set(
        type(ProblemConfig().constraints).__dataclass_fields__
    )


def test_example_values_are_the_defaults() -> None:
    cfg = _load()
    assert SolverConfig.from_dict(cfg["solver"]) == SolverConfig()
    assert ProblemConfig.from_dict(cfg["problem"]) == ProblemConfig()
    assert CostConfig.from_dict(cfg["cost"]) == CostConfig()

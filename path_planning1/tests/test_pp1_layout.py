"""布局守卫：契约层不许反向依赖，单文件不许过长。

"分层清晰、单文件不过长"写成文档容易被忘，用测试守住才有约束力。
"""

from __future__ import annotations

from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "src" / "path_planning1"
CONTRACT_MAX_LINES = 120
CORE_MAX_LINES = 300

FORBIDDEN_IN_CONTRACT = ("from ..core", "from ..api", "path_planning1.core", "path_planning1.api")


def _n_lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def test_contract_modules_are_short() -> None:
    files = sorted((PKG / "contract").glob("*.py"))
    assert files, "contract 子包为空"
    for path in files:
        assert _n_lines(path) <= CONTRACT_MAX_LINES, f"{path.name} {_n_lines(path)} 行，超过上限"


def test_core_modules_are_short() -> None:
    core = PKG / "core"
    if not core.exists():
        return  # M0 阶段 core 还没建
    for path in sorted(core.rglob("*.py")):
        assert _n_lines(path) <= CORE_MAX_LINES, f"{path.name} {_n_lines(path)} 行，超过上限"


def test_contract_does_not_depend_on_core_or_api() -> None:
    for path in sorted((PKG / "contract").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_IN_CONTRACT:
            assert token not in text, f"{path.name} 反向依赖 {token}"


def test_core_does_not_depend_on_api() -> None:
    core = PKG / "core"
    if not core.exists():
        return
    for path in sorted(core.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "path_planning1.api" not in text, f"{path.name} 反向依赖 api"

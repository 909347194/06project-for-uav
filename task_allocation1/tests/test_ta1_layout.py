"""布局守卫：契约层不反向依赖、单文件不过长、默认值唯一来源。"""

from __future__ import annotations

from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "src" / "task_allocation1"
CONTRACT_MAX_LINES = 120
CORE_MAX_LINES = 300
FORBIDDEN_IN_CONTRACT = ("from ..core", "from ..api", "task_allocation1.core", "task_allocation1.api")


def _n_lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def test_contract_modules_are_short() -> None:
    files = sorted((PKG / "contract").glob("*.py"))
    assert files
    for path in files:
        assert _n_lines(path) <= CONTRACT_MAX_LINES, f"{path.name} {_n_lines(path)} 行超限"


def test_core_modules_are_short() -> None:
    for path in sorted((PKG / "core").rglob("*.py")):
        assert _n_lines(path) <= CORE_MAX_LINES, f"{path.name} {_n_lines(path)} 行超限"


def test_contract_does_not_depend_on_core() -> None:
    for path in sorted((PKG / "contract").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_IN_CONTRACT:
            assert token not in text, f"{path.name} 反向依赖 {token}"


def test_no_file_io_in_core() -> None:
    """纯计算库：core 不得出现文件/网络 I/O 调用。"""
    banned = ("open(", "read_csv", "read_json", "requests.", "urlopen", "Path(")
    for path in sorted((PKG / "core").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{path.name} 出现 I/O {token}"

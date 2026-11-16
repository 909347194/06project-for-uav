"""真数据端到端回归（默认跳过，显式开启才跑）：建场 → 规划 → 产物 → 产物启动。

为什么单独一组：默认快测全是合成小网格，最危险的"最坏窗口覆盖 / 契约漂移 /
1.15 GB 场能否无损往返"没有端到端证据。但真建场要 120–140 s，塞进 PR CI 会毁掉
反馈循环，所以按 nightly 处理：

    PP1_SLOW_TESTS=1 python -m pytest -q path_planning1/tests/test_pp1_slow_integration.py

产物落在 pytest 的 tmp 目录（约 1.15 GB），跑完由 pytest 的临时目录策略回收；
本机磁盘紧张时别开。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from path_planning1.contract import FailureCode
from path_planning1.core.field_artifact import FIELD_ARTIFACT_SCHEMA

import path_planning1 as pp1

pytestmark = pytest.mark.skipif(
    os.environ.get("PP1_SLOW_TESTS") != "1",
    reason="真数据建场约 120–140 s；设置 PP1_SLOW_TESTS=1 才跑",
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "worlds" / "guangzhou"
START = (1636.8, 4198.2)
GOAL = (2040.6, 1393.6)
ALT_Z_M = 37.5   # 绝对 z；该点地面约 12.5 m，真高约 25 m


def test_real_startup_and_artifact_roundtrip(tmp_path: Path) -> None:
    if not (DATA / "region.geojson").is_file():
        pytest.skip(f"缺真实数据集：{DATA}")
    cfg = {"world": {"root": str(DATA)}}

    ctx = pp1.create_context(cfg)
    assert ctx.field_ready and ctx.field_source == "built"
    assert len(ctx.dataset_digest) == 12
    assert ctx.field_pad_m > 0 and ctx.max_od_length_m > 0

    first = ctx.plan(START, GOAL, ALT_Z_M)
    assert first.status.value == "ok"
    assert first.params_hash == ctx.digest

    # 域外 OD 在装配层被拒，不进核心规划
    rejected = ctx.plan((-100.0, -100.0), GOAL, ALT_Z_M)
    assert rejected.failure_code is FailureCode.OUTSIDE_WORLD

    artifact = tmp_path / "field"
    meta = ctx.save_field(artifact)
    assert meta["schema"] == FIELD_ARTIFACT_SCHEMA
    assert meta["dataset_digest"] == ctx.dataset_digest

    loaded = pp1.create_context(cfg, field_artifact=artifact)
    assert loaded.field_source == "artifact"
    assert loaded.dataset_digest == ctx.dataset_digest
    assert loaded.digest == ctx.digest               # 两条路径场身份完全一致
    assert loaded.field_pad_m == ctx.field_pad_m
    assert loaded.build_s < ctx.build_s              # 读产物是毫秒级
    assert np.array_equal(loaded.field.sdf[0, :8, :8], ctx.field.sdf[0, :8, :8])

    second = loaded.plan(START, GOAL, ALT_Z_M)
    a, b = first.metrics_dict(), second.metrics_dict()
    assert b["params_hash"] == a["params_hash"]
    assert b["length_m"] == a["length_m"]
    assert b["min_clearance_m"] == a["min_clearance_m"]

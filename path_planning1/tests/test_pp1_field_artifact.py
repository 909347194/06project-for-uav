"""场产物：往返无损、只读 mmap、四类契约漂移都在加载期炸掉。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from path_planning1.contract import GridSpec
from path_planning1.core.field_artifact import (
    FIELD_ARTIFACT_SCHEMA,
    field_contract,
    load_field_artifact,
    save_field_artifact,
)
from path_planning1.core.planning_context import PlanningContext
from path_planning1.core.route_segment_params import RouteSegmentParams
from path_planning1.core.world_field import FieldContractError, WorldField, field_signature

CONTRACT = {
    "max_od_length_m": 3000.0,
    "pad_m": 392.0,
    "spacing_m": (4.0, 4.0, 2.0),
    "agl_min_m": 10.0,
    "agl_max_m": 120.0,
    "bounds_m": (0.0, 0.0, 4846.0, 4794.0),
}
DIGEST = "d60b50223a62"


def _field() -> WorldField:
    grid = GridSpec(
        origin_enu=(0.0, 0.0, 10.0), spacing_m=(4.0, 4.0, 2.0),
        shape=(5, 4, 3), yaw_rad=0.0,
    )
    sdf = (np.arange(3 * 5 * 4, dtype=np.float32) + 1.0).reshape(3, 5, 4)
    occupancy = (np.arange(5 * 4 * 3) % 3 == 0).reshape(5, 4, 3)
    return WorldField(grid=grid, sdf=sdf, occupancy=occupancy)


def _saved(tmp_path: Path) -> Path:
    directory = tmp_path / "field"
    save_field_artifact(_field(), directory, dataset_digest=DIGEST, contract=CONTRACT)
    return directory


def test_roundtrip_is_lossless_readonly_and_sampleable(tmp_path: Path) -> None:
    original = _field()
    directory = _saved(tmp_path)
    assert (directory / "meta.json").is_file()

    field, meta = load_field_artifact(directory, dataset_digest=DIGEST, contract=CONTRACT)
    assert meta["schema"] == FIELD_ARTIFACT_SCHEMA
    assert meta["dataset_digest"] == DIGEST
    assert meta["signature"] == field_signature(original)
    assert field.grid == original.grid
    assert np.array_equal(field.sdf, original.sdf)
    assert np.array_equal(field.occupancy, original.occupancy)
    assert not field.sdf.flags.writeable and not field.occupancy.flags.writeable

    # mmap 数组必须能直接喂给在线取样（scipy 只读插值），不是只能看不能用
    window = GridSpec(origin_enu=(2.0, 2.0, 10.0), spacing_m=(4.0, 4.0, 2.0), shape=(2, 2, 2))
    assert field.window_sdf(window).shape == (2, 2, 2)


def test_missing_meta_is_reported_as_deployment_failure(tmp_path: Path) -> None:
    with pytest.raises(FieldContractError, match="产物不完整或不存在"):
        load_field_artifact(tmp_path / "never-built", dataset_digest=DIGEST, contract=CONTRACT)


def test_dataset_version_mismatch_rejects_stale_artifact(tmp_path: Path) -> None:
    with pytest.raises(FieldContractError, match="数据版本"):
        load_field_artifact(_saved(tmp_path), dataset_digest="other-data", contract=CONTRACT)


@pytest.mark.parametrize("key,value", [
    ("max_od_length_m", 6000.0), ("pad_m", 400.0),
    ("spacing_m", (4.0, 4.0, 1.0)), ("agl_min_m", 12.0),
])
def test_contract_drift_rejects_stale_artifact(tmp_path: Path, key: str, value) -> None:
    drifted = dict(CONTRACT, **{key: value})
    with pytest.raises(FieldContractError, match="建场契约"):
        load_field_artifact(_saved(tmp_path), dataset_digest=DIGEST, contract=drifted)


def test_schema_and_signature_tampering_are_detected(tmp_path: Path) -> None:
    directory = _saved(tmp_path)
    meta_path = directory / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    meta["schema"] = FIELD_ARTIFACT_SCHEMA + 1
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(FieldContractError, match="schema"):
        load_field_artifact(directory, dataset_digest=DIGEST, contract=CONTRACT)

    meta["schema"] = FIELD_ARTIFACT_SCHEMA
    meta["signature"] = {"world_field": {"shape": [1, 1, 1]}}
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(FieldContractError, match="签名"):
        load_field_artifact(directory, dataset_digest=DIGEST, contract=CONTRACT)


def test_context_save_field_carries_its_own_contract(tmp_path: Path) -> None:
    """后端只调 ctx.save_field(path)，不需要知道产物里该写哪些字段。"""
    params = RouteSegmentParams()
    ctx = PlanningContext(
        dataset=object(), params=params, field=_field(),
        field_mode="startup", bounds_m=CONTRACT["bounds_m"],
        max_od_length_m=CONTRACT["max_od_length_m"], field_pad_m=CONTRACT["pad_m"],
        dataset_digest=DIGEST,
    )
    directory = tmp_path / "from-context"
    meta = ctx.save_field(directory)
    assert meta["dataset_digest"] == DIGEST
    assert meta["contract"]["pad_m"] == CONTRACT["pad_m"]

    expected = field_contract(
        max_od_length_m=ctx.max_od_length_m, pad_m=ctx.field_pad_m,
        params=params, bounds_m=ctx.bounds_m,
    )
    loaded, _ = load_field_artifact(directory, dataset_digest=DIGEST, contract=expected)
    assert field_signature(loaded) == field_signature(_field())

    on_demand = PlanningContext(
        dataset=object(), params=params, field=None,
        field_mode="on_demand", bounds_m=CONTRACT["bounds_m"],
        max_od_length_m=CONTRACT["max_od_length_m"], field_pad_m=0.0,
    )
    with pytest.raises(FieldContractError, match="没有全域场"):
        on_demand.save_field(tmp_path / "should-not-exist")

"""数据集身份：内容变了一定要换缓存键，路径/时间变了不要换。"""

from __future__ import annotations

from pathlib import Path

from path_planning1.core.dataset_identity import dataset_digest


def _world(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "region.geojson").write_text('{"type":"FeatureCollection"}', encoding="utf-8")
    (root / "building").mkdir(exist_ok=True)
    (root / "building" / "a.shp").write_bytes(b"shp-bytes")
    (root / "building" / "a.dbf").write_bytes(b"dbf-bytes")
    (root / "dem.tif").write_bytes(b"tif-bytes")
    return root


def test_digest_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    root = _world(tmp_path / "city")
    kw = {"root": root}
    first = dataset_digest(kw)
    assert len(first) == 12
    assert dataset_digest(kw) == first
    assert dataset_digest({"root": str(root)}) == first       # Path 与 str 等价

    (root / "building" / "a.dbf").write_bytes(b"dbf-CHANGED")
    changed = dataset_digest(kw)
    assert changed != first
    (root / "building" / "a.dbf").write_bytes(b"dbf-bytes")
    assert dataset_digest(kw) == first                       # 改回去 = 同一个世界


def test_digest_tracks_every_data_scope_choice(tmp_path: Path) -> None:
    root = _world(tmp_path / "city")
    base = dataset_digest({"root": root})
    # with_dem=False 时 DEM 根本不参与世界，指纹也不该算它
    assert dataset_digest({"root": root, "with_dem": False}) != base
    assert dataset_digest({"root": root, "height_field": "H"}) != base
    assert dataset_digest({"root": root, "min_area_m2": 5.0}) != base
    assert dataset_digest({"root": root, "terrain_min_z_m": 2.0}) != base


def test_digest_ignores_absolute_path_and_mtime(tmp_path: Path) -> None:
    """产物要能从构建机拷到服务机：绝对路径 / 拷贝时间都不能进指纹。"""
    a, b = _world(tmp_path / "a"), _world(tmp_path / "b")
    assert dataset_digest({"root": a}) == dataset_digest({"root": b})
    import os
    import time
    stamp = time.time() + 3600
    os.utime(b / "building" / "a.shp", (stamp, stamp))
    assert dataset_digest({"root": b}) == dataset_digest({"root": a})


def test_missing_files_do_not_break_stub_worlds() -> None:
    """测试桩里 root 不存在也要给稳定的键（load_city 会在更早处明确报错）。"""
    kw = {"root": "definitely-not-a-real-directory"}
    assert dataset_digest(kw) == dataset_digest(kw)
    assert len(dataset_digest(kw)) == 12

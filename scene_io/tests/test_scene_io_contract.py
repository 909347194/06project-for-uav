"""scene_io 契约测试：零依赖、frame 校验、经纬度→ENU、两库对接。"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import scene_io

REPO = Path(__file__).resolve().parents[2]
WORLD_DIR = REPO / "data" / "worlds" / "guangzhou"
MISSION_DIR = REPO / "data" / "missions" / "demo_001"


@pytest.fixture(scope="module")
def world():
    return scene_io.load_world(WORLD_DIR)


@pytest.fixture()
def mission_copy(tmp_path):
    dst = tmp_path / "m"
    shutil.copytree(MISSION_DIR, dst)
    return dst


def _patch(path: Path, fn) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    fn(data)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# ---- 解耦：两库各自独立，本包不绑架任何一方 ---------------------------------
def test_package_has_no_runtime_dependencies():
    text = (REPO / "scene_io" / "pyproject.toml").read_text(encoding="utf-8")
    head = text.split("[project.optional-dependencies]")[0]
    assert "dependencies = []" in head, "scene_io 不能声明运行时依赖（会绑架两库）"


def test_import_scene_io_does_not_pull_algorithm_libs():
    code = (
        "import sys, scene_io;"
        "print('task_allocation1' in sys.modules, 'path_planning1' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False False", f"import scene_io 竟拉起了算法库：{out.stdout}"


def test_algorithm_libs_do_not_import_scene_io():
    for pkg in ("task_allocation1", "path_planning1"):
        hits = [p for p in (REPO / pkg / "src").rglob("*.py")
                if "scene_io" in p.read_text(encoding="utf-8")]
        assert not hits, f"{pkg} 不能依赖 scene_io：{hits}"


# ---- 世界 -------------------------------------------------------------------
def test_world_frame_hash_matches_recorded(world):
    assert world.frame_hash == scene_io.frame_hash(world.frame) == "59ccbad38a06"


def test_scene_io_frame_matches_pp1(world):
    """零依赖实现与 pp1 的 LocalFrame 必须同口径（对拍，不共享代码）。"""
    report = scene_io.check_world_with_pp1(world)
    assert report["frame_ok"] is True, report
    assert report["digest_ok"] is True, report


def test_world_kwargs_is_pp1_contract(world):
    kw = scene_io.to_pp1_world_kwargs(world, max_od_length_m=3000.0)
    assert kw["root"] == str(WORLD_DIR)
    assert kw["with_dem"] is True
    assert kw["max_od_length_m"] == 3000.0


# ---- 任务包 -----------------------------------------------------------------
def test_mission_points_are_derived_from_lonlat(world, mission_copy):
    m = scene_io.load_mission(mission_copy, world=world)
    raw = json.loads((mission_copy / "tasks.json").read_text(encoding="utf-8"))
    for t, r in zip(m.tasks, raw, strict=True):
        assert (t.x, t.y) == pytest.approx(world.frame.to_local(r["lon"], r["lat"]))
        assert t.z == pytest.approx(r["z_abs"])


def test_mission_bridges_to_ta1_and_matches_baseline(world, mission_copy):
    import task_allocation1 as ta1

    m = scene_io.load_mission(mission_copy, world=world)
    res = ta1.allocate(scene_io.to_ta1_request(m))
    base = json.loads((mission_copy / "baseline.json").read_text(encoding="utf-8"))
    tol = base["tolerance"]["abs"]

    assert res.solution.feasible == base["expect"]["feasible"]
    assert res.home_return_ok == base["expect"]["home_return_ok"]
    assert len(res.solution.executed_task_ids) == base["expect"]["n_executed"]
    assert res.metrics_dict()["path_m"] == pytest.approx(base["expect"]["path_m"], abs=tol)
    for tour in res.solution.tours:
        assert tour.is_home_return(), f"机 {tour.uav_id} 没回起飞巢"


def test_mission_bridges_to_ta1_with_vertical_section(world, mission_copy):
    """切面尺子的两座桥：采样器现造 + cost_config 透传（缺 pp1/shapely 就跳过）。"""
    pytest.importorskip("path_planning1")
    pytest.importorskip("shapely")
    import task_allocation1 as ta1

    ground_z, obstacle_top_z = scene_io.to_ta1_cost_samplers(world)
    xs = [t.x for t in scene_io.load_mission(mission_copy, world=world).tasks]
    ys = [t.y for t in scene_io.load_mission(mission_copy, world=world).tasks]
    g, o = ground_z(xs, ys), obstacle_top_z(xs, ys)
    assert len(g) == len(o) == len(xs)
    for gi, oi in zip(g, o, strict=True):      # 包络顶不低于地面
        assert float(oi) >= float(gi) - 1e-6

    cost = ta1.CostConfig(model="vertical_section", agl_min_m=10.0, agl_max_m=120.0)
    req = scene_io.to_ta1_request(
        scene_io.load_mission(mission_copy, world=world),
        cost_config=cost, ground_z=ground_z, obstacle_top_z=obstacle_top_z,
    )
    assert req.cost_config.model == "vertical_section"
    assert req.ground_z is ground_z and req.obstacle_top_z is obstacle_top_z


def test_mission_rejects_frame_hash_mismatch(world, mission_copy):
    _patch(mission_copy / "mission.json", lambda d: d.update({"frame_hash": "deadbeef0000"}))
    with pytest.raises(ValueError, match="frame_hash 不匹配"):
        scene_io.load_mission(mission_copy, world=world)


def test_mission_rejects_enu_points(world, mission_copy):
    def add_xy(d):
        d[0].pop("lon")
        d[0].pop("lat")
        d[0].update({"x": 100.0, "y": 200.0})

    _patch(mission_copy / "tasks.json", add_xy)
    with pytest.raises(ValueError, match="只收 lon/lat"):
        scene_io.load_mission(mission_copy, world=world)


def test_mission_rejects_missing_nest_id(world, mission_copy):
    _patch(mission_copy / "uavs.json", lambda d: d[0].pop("nest_id"))
    with pytest.raises(ValueError, match="缺少 nest_id"):
        scene_io.load_mission(mission_copy, world=world)


def test_mission_rejects_over_capacity(world, mission_copy):
    def clone_uavs(d):
        base = dict(d[0])
        for extra in range(1, 6):
            row = dict(base)
            row["id"] = 100 + extra
            d.append(row)

    _patch(mission_copy / "uavs.json", clone_uavs)
    with pytest.raises(ValueError, match="超过 capacity"):
        scene_io.load_mission(mission_copy, world=world)


def test_mission_reports_all_problems_at_once(world, mission_copy):
    _patch(mission_copy / "mission.json", lambda d: d.pop("frame_hash"))
    _patch(mission_copy / "nests.json", lambda d: d[0].pop("z_abs"))
    with pytest.raises(ValueError) as exc:
        scene_io.load_mission(mission_copy, world=world)
    msg = str(exc.value)
    assert "frame_hash" in msg and "z_abs" in msg

# ---- schema 版本锚 ----------------------------------------------------------
def test_world_rejects_unknown_schema(tmp_path):
    dst = tmp_path / "w"
    shutil.copytree(WORLD_DIR, dst)

    def bump(d):
        d["schema"] = 9

    _patch(dst / "world.json", bump)
    with pytest.raises(ValueError, match="schema"):
        scene_io.load_world(dst)


def test_mission_rejects_unknown_schema(world, mission_copy):
    _patch(mission_copy / "mission.json", lambda d: d.update({"schema": 2}))
    with pytest.raises(ValueError, match="schema"):
        scene_io.load_mission(mission_copy, world=world)


def test_mission_error_header_names_the_package(world, mission_copy):
    """报错抬头要说清是哪个任务包，而不是把所有跨文件问题记在 mission.json 头上。"""
    _patch(mission_copy / "tasks.json", lambda d: d[0].pop("lon"))
    with pytest.raises(ValueError) as exc:
        scene_io.load_mission(mission_copy, world=world)
    msg = str(exc.value)
    assert msg.startswith("任务包 demo_001"), msg
    assert "缺少 lon/lat" in msg, msg

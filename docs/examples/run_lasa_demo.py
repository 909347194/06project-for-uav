#!/usr/bin/env python3
"""拉萨市城关区 · 多机协同「分配 → 规划」完整流程演示（教程 4 的可执行配套）。

    python3 docs/examples/run_lasa_demo.py            # 跑全程
    python3 docs/examples/run_lasa_demo.py --quick    # 只跑 ta1（跳过 pp1，秒级）

流程（对应 `docs/4完整流程Tutorial.md` §6–§8）：

    ① scene_io 装载世界 + 任务包（经纬度 → ENU，校验 frame / frame_hash）
    ② ta1.allocate()            —— 谁飞、什么顺序、回哪个巢（同巢起降）
    ③ 展开 tours → 航段          —— 只把**空中航段**交给 pp1，起降另算
    ④ pp1.create_context()      —— 建全域场（一次），保存产物供复用
    ⑤ 逐段 ctx.plan()           —— 真航迹 + 时长 + 净空
    ⑥ 用真 duration_s 复核时间窗与航程，写 results/

零第三方依赖（只用标准库 + 三个 workspace 成员包）；不改任何库内代码。
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # 仓库根 = 运行根（教程假设项目根目录为 /）

import scene_io
import task_allocation1 as ta1

WORLD_DIR = ROOT / "data/worlds/lasa_chengguan"
MISSION_DIR = ROOT / "data/missions/lasa_demo_001"
ARTIFACT_DIR = ROOT / "artifacts/lasa_chengguan/default"
RESULT_DIR = ROOT / "results/lasa_chengguan/lasa_demo_001"

# 起降点不在 pp1 的飞行带内（贴地），所以每个航路点都按“巡航真高”抬到绝对 z 再规划；
# 垂直起降段由上层自己管，pp1 只负责水平位移的那几段。
CRUISE_AGL_M = 40.0


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------- ① 装载
def load_scene():
    section("① 装载世界 + 任务包（scene_io，零依赖）")
    world = scene_io.load_world(WORLD_DIR)
    mission = scene_io.load_mission(MISSION_DIR, world=world)
    print(f"world_id      : {world.world_id}")
    print(f"frame         : lon0={world.frame.lon0} lat0={world.frame.lat0} "
          f"lat_ref={world.frame.lat_reference}")
    print(f"frame_hash    : {world.frame_hash}")
    print(f"dataset_digest: {world.dataset_digest}")
    print(f"任务包        : {mission.mission_id}  "
          f"tasks={len(mission.tasks)} uavs={len(mission.uavs)} nests={len(mission.nests)}")
    print("\n前 3 个任务的经纬度 → ENU（水平真源是经纬度，ENU 是派生值）：")
    for t in mission.tasks[:3]:
        print(f"  task {t.id}: ({t.lon:.6f}, {t.lat:.6f}) -> "
              f"({t.x:8.1f}, {t.y:8.1f})  z_abs={t.z:.2f}")
    return world, mission


# ---------------------------------------------------------------- ② ta1
def allocate(mission):
    section("② ta1.allocate() —— 谁飞、什么顺序、回哪个巢")
    req = scene_io.to_ta1_request(mission)          # 距离口径走 mission.json 的 cost 段
    t0 = time.perf_counter()
    res = ta1.allocate(req)
    dt = time.perf_counter() - t0
    print(f"耗时 {dt:.2f}s  model={res.model_type.value}  n_evals={res.n_evals}")
    print(f"home_return_ok = {res.home_return_ok}   feasible = {res.solution.feasible}")
    for tour in res.solution.tours:
        if not tour.task_ids:
            print(f"  UAV {tour.uav_id}: (闲置)")
            continue
        print(f"  UAV {tour.uav_id}: nest {tour.start_nest_id} -> "
              f"{tour.task_ids} -> nest {tour.end_nest_id}   "
              f"est_cost={tour.est_cost:.1f} m  est_reward={tour.est_reward}")
    print("metrics:", json.dumps(res.metrics_dict(), ensure_ascii=False))
    return res


# ---------------------------------------------------------------- ③ 展开
def expand_tours(res, mission):
    """tours → 航段（只含空中航段；起降垂直段不进 pp1）。

    每条 tour 的形状是  巢 → t1 → … → tn → 巢（同巢起降）。
    这里把“巢”也抬到巡航真高，于是  巢@巡航高 → t1 → … → tn → 巢@巡航高
    每一对相邻点都是一条合法的空中航段，可直接喂 pp1；真正的垂直起降另算。
    """
    by_task = {t.id: t for t in mission.tasks}
    by_nest = {n.id: n for n in mission.nests}
    segs = []
    for tour in res.solution.tours:
        if not tour.task_ids:
            continue
        pts = [by_nest[tour.start_nest_id]]
        pts += [by_task[i] for i in tour.task_ids]
        pts.append(by_nest[tour.end_nest_id])
        for a, b in itertools.pairwise(pts):
            segs.append({
                "uav_id": tour.uav_id,
                "from": {"id": a.id, "x": a.x, "y": a.y, "z": a.z},
                "to": {"id": b.id, "x": b.x, "y": b.y, "z": b.z},
            })
    return segs


# ------------------------------------------------------- ④⑤ pp1 规划
def plan_segments(world, segs):
    section("④⑤ pp1：建全域场（一次）→ 逐段规划")
    import path_planning1 as pp1

    cfg = {
        "world": scene_io.to_pp1_world_kwargs(
            world, max_od_length_m=1500.0, nofly_altitude_mode="absolute"),
        "params": {},
    }
    t0 = time.perf_counter()
    ctx = pp1.create_context(cfg)
    print(f"建场耗时 {ctx.load_s:.1f}s(装载) + {ctx.build_s:.1f}s(建场)"
          f"  field_source={ctx.field_source}")
    print(f"dataset_digest = {ctx.dataset_digest}")
    print(f"bounds_m = {[round(v, 1) for v in ctx.bounds_m]}  "
          f"max_od_length_m = {ctx.max_od_length_m:.0f}  pad_m = {ctx.field_pad_m:.0f}")
    if ctx.field is not None:
        print(f"全域场 shape={ctx.field.grid.shape}  "
              f"内存 {ctx.field.nbytes / 1e6:.0f} MB")

    # 构建期把场落成产物：下次启动 mmap 只读映射，毫秒级
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    meta = ctx.save_field(ARTIFACT_DIR)
    print(f"场产物已存 {ARTIFACT_DIR.relative_to(ROOT)}  keys={sorted(meta)}")

    out = []
    for i, seg in enumerate(segs, 1):
        a, b = seg["from"], seg["to"]
        t0 = time.perf_counter()
        plan = ctx.plan((a["x"], a["y"]), (b["x"], b["y"]), a["z"], goal_z_m=b["z"])
        seg_time = time.perf_counter() - t0
        rec = {
            "seg": i, "uav_id": seg["uav_id"],
            "from_id": a["id"], "to_id": b["id"],
            "status": plan.status.value, "failure_code": plan.failure_code.value,
            "elapsed_s": round(seg_time, 2),
            "length_m": plan.metrics.length_m,
            "min_clearance_m": plan.metrics.min_clearance_m,
            "duration_s": plan.metrics.duration_s,
            "max_airspeed_mps": plan.metrics.max_airspeed_mps,
            "has_trajectory": plan.trajectory is not None,
        }
        out.append(rec)
        print(f"  [{i}/{len(segs)}] UAV{seg['uav_id']} {a['id']}→{b['id']}  "
              f"{plan.status.value:9s} len={plan.metrics.length_m} m  "
              f"clear={plan.metrics.min_clearance_m} m  "
              f"dur={plan.metrics.duration_s} s  {seg_time:.1f}s")
        if plan.status.value != "ok":
            print(f"        失败码={plan.failure_code.value}  {plan.message[:120]}")
    return out, ctx


# ---------------------------------------------------------------- ⑥ 复核
def review(res, mission, seg_recs):
    section("⑥ 用真 duration_s 复核时间窗与航程")
    by_task = {t.id: t for t in mission.tasks}
    by_uav = {u.id: u for u in mission.uavs}
    service = 30.0            # mission.json 的 problem.default_service_time
    rows = []
    for tour in res.solution.tours:
        if not tour.task_ids:
            continue
        u = by_uav[tour.uav_id]
        recs = [r for r in seg_recs if r["uav_id"] == tour.uav_id and r["status"] == "ok"]
        cursor = mission.current_time
        plan_m = 0.0
        for tid in tour.task_ids:
            t = by_task[tid]
            leg = next((r for r in recs if r["to_id"] == tid), None)
            cursor += (leg["duration_s"] if leg else 0.0)
            arrive = max(cursor, float(t.earliest or 0.0))
            late = max(0.0, arrive - float(t.latest)) if t.latest is not None else 0.0
            plan_m += (leg["length_m"] if leg else 0.0)
            rows.append({"uav_id": tour.uav_id, "task_id": tid,
                         "arrive_s": round(arrive, 1), "latest": t.latest,
                         "late_s": round(late, 1)})
            cursor = arrive + service
        print(f"  UAV {tour.uav_id} (v={u.speed:g} m/s, 航程预算 {u.remaining_range:.0f} m): "
              f"真航程 {plan_m:.0f} m  交付 {len(tour.task_ids)} 个任务")
    print(f"\n{'UAV':>4} {'task':>6} {'到达(s)':>9} {'最晚':>8} {'迟到(s)':>8}")
    for r in rows:
        print(f"{r['uav_id']:>4} {r['task_id']:>6} {r['arrive_s']:>9} "
              f"{r['latest']!s:>8} {r['late_s']:>8}")
    n_late = sum(1 for r in rows if r["late_s"] > 0)
    print(f"\n迟到任务数 = {n_late} / {len(rows)}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="只跑 ta1，跳过 pp1")
    args = ap.parse_args()

    world, mission = load_scene()
    res = allocate(mission)
    segs = expand_tours(res, mission)
    print(f"\n展开为 {len(segs)} 条空中航段（起降垂直段不在其中）")
    for s in segs:
        print(f"  UAV{s['uav_id']}  {s['from']['id']} -> {s['to']['id']}")

    if args.quick:
        print("\n(--quick：跳过 pp1)")
        return 0

    seg_recs, ctx = plan_segments(world, segs)
    rows = review(res, mission, seg_recs)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "mission_id": mission.mission_id,
        "world_id": world.world_id,
        "frame_hash": world.frame_hash,
        "dataset_digest": ctx.dataset_digest,
        "ta1": res.metrics_dict(),
        "tours": [{"uav_id": t.uav_id, "task_ids": t.task_ids,
                   "start_nest_id": t.start_nest_id, "end_nest_id": t.end_nest_id}
                  for t in res.solution.tours],
        "segments": seg_recs,
        "review": rows,
    }
    out = RESULT_DIR / "run_001.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n结果已写 {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

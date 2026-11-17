# 4. 完整流程 Tutorial：第一次就跑通一次多机任务

用一个现成例子（拉萨城关区 `lasa_chengguan`）从头到尾跑一遍：安装 → 校验 → 分配 → 展开 → 规划 → 复核 → 落盘。**数据已随仓库提供，命令可直接复制。**

> 本文用 `/` 表示项目根目录（`06project-for-uav/`），相对路径都从它算起。数据格式和换城市见 [文档 0](0数据准备与处理.md)，单个库的细节见 [文档 1](1怎么用-ta1算法.md) / [文档 2](2怎么用-pp1算法.md)，对接口径见 [文档 3](3任务分配-路径规划对接.md)。

## 想先看到结果？

```bash
python3 docs/examples/run_lasa_demo.py            # 全程，约 1 分钟
python3 docs/examples/run_lasa_demo.py --quick    # 只跑 ta1，约 2 秒
```

这一个脚本做六件事，对应下面 §4–§6：

```text
① scene_io 装载世界 + 任务包（经纬度 → ENU，校验 frame / frame_hash）
② ta1.allocate()        —— 谁飞、什么顺序、回哪个巢（同巢起降）
③ 展开 tours → 航段      —— 只把空中航段交给 pp1，起降另算
④ pp1.create_context()  —— 建全域场（一次）
⑤ 逐段 ctx.plan()       —— 真航迹 + 时长 + 净空
⑥ 用真 duration_s 复核时间窗与航程（只报告差距），写 results/
```

输出落在 `/results/lasa_chengguan/lasa_demo_001/run_001.json`，含 `ta1` 指标、`tours`、每段 `segments`、逐任务 `review`（迟到对比）。

## 1. 先记住三件事

三个包：`task_allocation1`（ta1，分配）、`path_planning1`（pp1，单航段规划）、`scene_io`（数据适配，零依赖）。

1. **两库互不 import，各自可单独安装上线**；只用分配就只装 ta1，只用规划就只装 pp1。
2. **对接只走数据契约**：同一套 ENU 米、同一个 `frame_hash`、同一份 `world.json`。
3. **高度从头到尾只有一个口径：绝对 z**。`AGL`（真高）是 pp1 库内部按 DEM 派生的校验量，**不是接口字段、也不落库**。拉萨地面 ~3650 m，"离地 40 m"就是 `z_abs ≈ 3690`，**不要传 40**。

## 2. 安装

环境：Python **3.11 / 3.12**、内存 ≥ 4 GB、磁盘 ≥ 1 GB、首次装依赖需联网。

```bash
cd /
uv sync --all-packages        # 推荐；这个 flag 不能省——根是 virtual workspace，不带就装不到成员包依赖
uv run pytest                 # 269 个用例
uv run ruff check .
```

没有 uv 就用 venv + pip，按需装：

```bash
cd /
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install -e ./task_allocation1                    # 只要"谁飞"
pip install -e "./path_planning1[nlp]"               # 要时长就必须带 [nlp]，否则 trajectory 恒为 None
pip install -e ./scene_io
```

验证：

```bash
python -c "import task_allocation1 as ta1; print(ta1.__version__, ta1.ALGORITHM_ID)"   # 0.2.0 ta1-dmde-home-nest
python -c "import path_planning1 as pp1;  print(pp1.__version__, pp1.ALGORITHM_ID)"    # 0.4.0 pp1-rrt-nlp-time
python -c "import scene_io;               print(scene_io.__version__)"                 # 0.2.0
python -m pytest -q                        # 266 passed + 3 skipped
```

> 新 clone 跳过 3 个测试：`guangzhou` 的 5 个 shp 夹具（约 26 MB）按 `.gitignore` 排除需自备；本教程用的 `lasa_chengguan` 建筑是 GeoJSON（211 KB），**已随仓库提供**，clone 下来就能跑。

## 3. 配置

三层配置别混：

| 层 | 在哪 | 谁消费 | 什么时候定 |
|---|---|---|---|
| 世界 | `data/worlds/<id>/world.json` + `world` 段 | pp1 / scene_io | 建世界时，季度级 |
| 参数 | pp1 的 `params` 段（走廊、栅格、飞行带、机型） | pp1 | `create_context` 一次固定 |
| 请求 | ta1 的 `solver / problem / cost` 段 + 每条 OD | ta1 / pp1 | 每个任务包 / 每条请求 |

本教程只用这份配置（`max_od_length_m=1500` 派生 `pad=244`：场约 255 MB、建场约 18 s，够用即可）：

```json
{
  "world": {
    "root": "data/worlds/lasa_chengguan", "with_dem": true, "field_mode": "startup",
    "max_od_length_m": 1500.0, "nofly_path": "nofly.geojson", "nofly_altitude_mode": "absolute"
  },
  "params": {
    "roi": {"width_ratio": 0.2, "margin_m": 50.0}, "spacing_m": [4.0, 4.0, 2.0],
    "agl_min_m": 10.0, "agl_max_m": 120.0, "seed": 0,
    "vehicle": {"v_max_mps": 20.0, "a_max_mps2": 3.0, "clearance_m": 5.0,
                "thrust_max_n": 100.0, "mass_kg": 5.0}
  }
}
```

完整字段与默认值见 [文档 1](1怎么用-ta1算法.md) / [文档 2](2怎么用-pp1算法.md)。

## 4. 数据校验

数据已随仓库提供，`data/worlds/lasa_chengguan/` + `data/missions/lasa_demo_001/`，不用重新生成。只需两行验过：

```python
import scene_io
world = scene_io.load_world("data/worlds/lasa_chengguan")          # 校验 frame
m = scene_io.load_mission("data/missions/lasa_demo_001", world=world)  # 一次报全数据问题
print(m.summary())
# {'mission_id': 'lasa_demo_001', 'world_id': 'lasa_chengguan', 'frame_hash': '0af4e4fb592e',
#  'n_tasks': 6, 'n_uavs': 4, 'n_nests': 2, 'current_time': 0.0}
```

这两行过了就能往下跑。> 这批建筑与 DEM 是合成示意数据：坐标、`frame_hash`、`dataset_digest`、`z_abs` 口径是真的，输出数字只证明流程通，不代表真实可飞性。

## 5. 核心四步

### 5.1 ta1：任务分配

```python
import task_allocation1 as ta1
from task_allocation1.terrain import load_world_terrain

ground_z, obstacle_top_z = load_world_terrain("data/worlds/lasa_chengguan").samplers()
res = ta1.allocate(scene_io.to_ta1_request(m, ground_z=ground_z, obstacle_top_z=obstacle_top_z))
print(res.home_return_ok, res.solution.feasible)
for t in res.solution.tours:
    print(f"UAV {t.uav_id}: nest {t.start_nest_id} -> {t.task_ids} -> nest {t.end_nest_id}")
```

实测（`seed=0`，同种子逐位可复现）：耗时 1.6 s，`home_return_ok = True`、`feasible = True`，6 个任务分给 2 架机（另 2 架进场但闲置——`n_participating=4`，`n_active_uavs=2`）：

```text
UAV 0: nest 0 -> [102, 103, 101] -> nest 0   est_cost=3239.5 m
UAV 2: nest 1 -> [106, 105, 104] -> nest 1   est_cost=3171.6 m
path_m=6411.15  return_leg_m=1035.87  f1_reward=6.8
```

同巢不变量：每条非空航次 `end_nest_id == start_nest_id == uav.nest_id`。`path_m` 只是代理航程，**不是交付值**。

### 5.2 展开为航段（最容易踩的坑）

ta1 的一条 tour 是 `巢 → t1 → t2 → t3 → 巢`，但 pp1 的飞行带是**离地 10–120 m**，巢在地面。直接把 `巢 → t1` 喂给 pp1 会遇到起降垂直段的问题——那不是 pp1 的事。

正确做法：每个航路点都带绝对 z，只把相邻点之间的空中位移交给 pp1；垂直起降段由上层自己管。

```python
import itertools
def expand_tours(res, mission):
    by_task = {t.id: t for t in mission.tasks}
    by_nest = {n.id: n for n in mission.nests}
    segs = []
    for tour in res.solution.tours:
        if not tour.task_ids:
            continue
        pts = [by_nest[tour.start_nest_id]] + [by_task[i] for i in tour.task_ids] + [by_nest[tour.end_nest_id]]
        for a, b in itertools.pairwise(pts):
            segs.append({"uav_id": tour.uav_id,
                         "from": {"x": a.x, "y": a.y, "z": a.z},
                         "to":   {"x": b.x, "y": b.y, "z": b.z}})
    return segs
```

本例 2 条 tour → **8 条空中航段**。要求"离地 25 m 巡航"就在这步把 z 抬到 `地面 + 25`，交给 pp1 后没有再换算的环节。

### 5.3 pp1：逐段规划

```python
import path_planning1 as pp1
ctx = pp1.create_context({"world": scene_io.to_pp1_world_kwargs(world, max_od_length_m=1500.0), "params": {}})
res = ctx.plan((x0, y0), (x1, y1), z0, goal_z_m=z1)
print(res.status.value, res.failure_code.value, res.metrics_dict())
```

建场实测：装载 + 建场 ≈ 17.7 s，`field_source=built`，全域场 `shape=(994, 732, 70)`、内存 255 MB。8 段全绿，摘录三段：

| 航段 | status | length_m | min_clearance_m | duration_s |
|---|---|---:|---:|---:|
| nest0 → 102 | ok | 822.2 | 17.03 | 47.83 |
| 102 → 103 | ok | 894.2 | 24.60 | 51.44 |
| 104 → nest1 | ok | 591.7 | 8.43 | 35.72 |

`min_clearance_m=30.0` 表示"至少 30 m"（量程上限），不是精确值。`duration_s` 只来自 pp1 的时间域轨迹；`trajectory=None` 时是 `DEGRADED`，不能拿长度除速度凑时长。生产上把场落成产物复用见 [文档 2](2怎么用-pp1算法.md) §部署。

### 5.4 复核：用真时长看差距（只报告）

把 ta1 估算和 pp1 真值摆在一起，列出哪些任务真值下会迟到、真航程差多少。**不改派、不重排**，要不要调整由上层服务决定。本例：

```text
 UAV   task     到达(s)     最晚    迟到(s)
   0    102      47.8   1500.0      0.0
   0    103     200.0   1800.0      0.0
   2    106      36.8   1700.0      0.0
迟到任务数 = 0 / 6
```

三种回执只进报告、不改派：`ok + trajectory` 用真值算差距；`ok + trajectory=None` 标 `DEGRADED`（时长不可信，不进对比）；`infeasible` 记失败码 + 端点。

## 6. 落盘与缓存

结果写 `results/<world_id>/<mission_id>/run_001.json`，含 `mission_id`、`world_id`、`frame_hash`、`dataset_digest`、`ta1`、`tours`、`segments`、`review`。

缓存键 = `dataset_digest + params_hash + 两端 xy/z`（**不用 `world_id`**——目录名改了数据未必变，只有内容指纹可靠）；任一项变了结果就不可比、缓存必须失效。

## 7. 换城市

按 [文档 0](0数据准备与处理.md) §12 的九步做：定运营区 → 建筑 → DEM → 禁飞区 → `world.json` → 预检建场 → 任务包 → ta1 冒烟 → 基线。改完数据必须同时做三件事：重算 `dataset_digest` 写回 `world.json`、删掉 `artifacts/` 重建场、重算 `baseline.json`。

## 8. 常见报错（按在哪一步炸的找）

| 现象 / 报错 | 多半是 | 怎么修 |
|---|---|---|
| `import task_allocation1` 失败 | 没带 `--all-packages`，根是 virtual workspace | `uv sync --all-packages`，或 `pip install -e ./task_allocation1` |
| `ModuleNotFoundError: casadi` / `trajectory` 全 `None` | 装了 pp1 没装 NLP 依赖 | `pip install -e "./path_planning1[nlp]"` |
| `frame_hash` / `world_id` 不匹配 | 任务包不是用这个世界的 frame 算的 | 重算经纬度或换 `world_id` |
| `任务包只收 lon/lat` | 任务包里写了 ENU 的 `x/y` | 用 `frame.to_lonlat(x, y)` 反算 |
| `缺少 z_abs` | 只给了 AGL / 相对高度 | 补 `z_abs = ground_z + AGL`（唯一加法点在适配层） |
| 大量 `out_of_flight_band` | 把"离地 40 m"当绝对 z 传了 40 | 拉萨地面 ~3650 m，应传 `z_abs ≈ 3690` |
| `od_too_long` | OD 超过 `max_od_length_m`（本教程 1500 m） | 拆航段，或调大契约并重建场 |
| 建场慢 / 内存爆 | `max_od_length_m` 或 `spacing_m` 调太大 | 够用即可；生产改 `field_artifact` mmap |
| `rrt_no_path` / `roi_no_path` | 走廊太窄 / 净空太大 / 分辨率不匹配 | 先调 `roi.width_ratio`、净空、数据，别只加迭代 |
| `FieldContractError` | 产物过期、数据改了没重建场、窗口越界 | **部署事故**：重建产物，别重试、别映射成 400 |
| `trajectory=None` 但 `status=ok` | NLP 没解出时间剖面 | 标 `DEGRADED`，只给折线；**不是失败** |
| `path_clearance_violated` | 真几何净空终检没过 | 不重试；带 OD + `params_hash` + 最差点上报人工 |

更全的清单：[文档 0](0数据准备与处理.md) §13（数据）、[文档 2](2怎么用-pp1算法.md)（pp1 失败码）、[文档 3](3任务分配-路径规划对接.md)（对接）。

---

**文档地图**：`0` 数据准备 → `1` ta1 → `2` pp1 → `3` 两库对接 → `4` 本文（完整跑通）。

# 06project-for-uav 快速上手

按使用视角组织：先看结构，再备数据/配置/环境，然后分别单用 ta1、单用 pp1，最后串起两个算法。库内部设计与实测基线不在这里，见文末文档地图。

## 1. 项目结构

```
06project-for-uav/
├─ task_allocation1/    ta1 分配层：谁飞、什么顺序、回哪个巢（同巢起降）
├─ path_planning1/      pp1 落地层：每条航段怎么飞（航迹＋时长＋净空）
├─ scene_io/            数据适配：世界/任务包 → 纯数据，零依赖，不 import 两库
├─ data/
│  ├─ worlds/<id>/      world.json + region/building/dem/nofly
│  ├─ missions/<id>/    mission/nests/uavs/tasks.json
│  └─ contracts/        pp1 契约测试夹具
├─ docs/                用法文档 + 全量配置示例 + examples/run_lasa_demo.py
├─ artifacts/   (可选)  建场产物，跟世界走，不进版本库
└─ results/     (可选)  运行结果，不进版本库
```

两库互不 import，可单独安装、单独上线；对接只靠数据契约（同一套 ENU 米、同一个 `frame_hash`、`world.json`）。高度口径全程只有**绝对 z**，"离地 N 米"要自己换算。

## 2. 快速上手

### 2.1 数据

仓库自带拉萨样例（`lasa_chengguan` + `lasa_demo_001`，建筑为 GeoJSON），clone 下来即可跑。广州样例的建筑 shp（约 26 MB）按 `.gitignore` 排除、需自备；缺它只影响广州，不影响拉萨。

校验数据是否就绪：

```python
import scene_io
world = scene_io.load_world("data/worlds/lasa_chengguan")
m = scene_io.load_mission("data/missions/lasa_demo_001", world=world)
print(m.summary())   # frame_hash / 任务数 / 机数 / 巢数
```

### 2.2 配置

三处配置，别混：

| 配置 | 位置 | 谁读 |
|---|---|---|
| 世界 | `data/worlds/<id>/world.json` | pp1 / scene_io |
| ta1 | `solver` / `problem` / `cost` 三段 | ta1 |
| pp1 | `world` / `request` / `params` 三段 | pp1 |

全量示例：`docs/ta1_full_config.example.json`、`docs/pp1_full_config.example.json`（`.jsonc` 是带注释版）。不写即默认；换城市改 `root`，换航段改 `request`。走任务包时，`mission.json` 里的同名段由 `scene_io.to_ta1_request()` 自动装配。

### 2.3 安装

Python 3.11。用 uv（推荐）：

```bash
uv sync --all-packages      # 根是 virtual workspace，这个 flag 不能省
```

或用 venv + pip，按需装：

```bash
pip install -e ./task_allocation1
pip install -e "./path_planning1[nlp]"   # 要时长就必须带 [nlp]（casadi）
pip install -e ./scene_io
```

只用分配就只装 ta1；pp1 不装 `[nlp]` 也能跑几何层，但拿不到 `duration_s`。验证：

```bash
python -c "import task_allocation1 as ta1, path_planning1 as pp1, scene_io; print(ta1.__version__, pp1.__version__, scene_io.__version__)"
```

## 3. 单独使用 ta1

内存里给任务/无人机/机巢，返回"谁飞、按什么顺序、回哪个巢"。核心不读文件、不联网；无地形数据时用欧氏尺子：

```python
import task_allocation1 as ta1

req = ta1.AllocateRequest(
    tasks=[ta1.Task(id=0, x=0, y=0, reward=1.0, latest=3600),
           ta1.Task(id=1, x=800, y=0, reward=1.0, latest=3600)],
    uavs=[ta1.UAV(id=0, x=0, y=0, nest_id=0, speed=15.0, remaining_range=1e5)],
    nests=[ta1.Nest(id=0, x=0, y=0, capacity=4)],
    distance_fn=ta1.euclidean3d,
    solver_config=ta1.SolverConfig(pop_size=30, max_generations=100, seed=42),
)
res = ta1.allocate(req)
res.home_return_ok              # 每条非空航次都回到起飞巢
res.solution.tours              # 每条航次：task_ids + 终点巢
res.metrics_dict()              # 扁平指标，可直接序列化
```

有世界目录时用默认的垂直切面尺子（更贴近真实可飞航程）；两个采样器由 ta1 自己造，不依赖 pp1：

```python
from task_allocation1.terrain import load_world_terrain   # 需 task_allocation1[terrain]
g, o = load_world_terrain("data/worlds/lasa_chengguan").samplers()
req = ta1.AllocateRequest(tasks=..., uavs=..., nests=...,
                          ground_z=g, obstacle_top_z=o)    # 去掉 distance_fn
```

硬不变量：每条非空航次 `end_nest_id == start_nest_id == uav.nest_id`。`path_m` 只是排序代理，不是真航程。

## 4. 单独使用 pp1

给一个世界 + 一条空中航段，返回航迹、时长、净空。端点用 ENU 米 + 绝对 z；只有 `trajectory` 是要执行的交付物。

```python
import json, path_planning1 as pp1

cfg = json.load(open("docs/pp1_full_config.example.json", encoding="utf-8"))
cfg["world"]["root"] = "data/worlds/guangzhou"     # 广州样例；建筑 shp 需自备

ctx = pp1.create_context(cfg)                       # 进程启动一次，建全域场
req = pp1.request_from_config(cfg)
res = ctx.plan(req["start_xy"], req["goal_xy"],
               req["start_z_m"], goal_z_m=req["goal_z_m"])

print(res.status.value, res.failure_code.value)     # ok / infeasible
print(res.metrics_dict())                           # length_m / duration_s / 净空 ...
```

- 生产上复用场：`ctx.save_field(dir)` → `create_context(cfg, field_artifact=dir)`，启动从分钟级降到毫秒级。
- `trajectory=None` 不是失败，只是没解出时间剖面；`duration_s` 只来自轨迹，不能用长度除速度凑。

## 5. 完整使用两个算法

链路：ta1 定"谁飞、什么顺序"→ 展开航段 → pp1 逐段出真时长 → 用真时长复核时间窗。直接跑随仓库的示例（拉萨数据，全程约 1 分钟；`--quick` 只跑 ta1，约 2 秒）：

```bash
python docs/examples/run_lasa_demo.py
python docs/examples/run_lasa_demo.py --quick
```

展开航段时两个容易踩的坑：

1. **起降不进 pp1**。ta1 的一条 tour 是"巢 → t1 → … → 巢"，巢贴地不在飞行带内；把每个航路点抬到巡航绝对 z，只把水平空中段交给 `ctx.plan`，垂直起降由上层管。
2. **高度只有绝对 z**。`AGL = z − 地面高程` 是 pp1 内部按 DEM 派生的校验量，不是接口字段；传"离地 25"会直接 `out_of_flight_band`。

结果落在 `results/lasa_chengguan/lasa_demo_001/run_001.json`，含 ta1 指标、tours、每段航迹指标、逐任务到达时间复核。

## 文档地图

- `docs/0数据准备与处理.md` — 数据格式 / 换城市
- `docs/1怎么用-ta1算法.md`、`docs/2怎么用-pp1算法.md` — 各库字段、失败码、部署
- `docs/3任务分配-路径规划对接.md` — 两库对接契约、失败回环、延迟预算
- `docs/4完整流程Tutorial.md` — 第 5 节的逐步展开
- `docs/5交付说明-给后端.md` — 能力边界与已知问题
- `path_planning1/README.md` — pp1 算法设计与实测基线

# scene_io

**数据适配层**。只做一件事：把磁盘上的「世界 + 任务包」读成纯数据，并守住坐标系一致性。

**零运行时依赖**（只用标准库），**不 import `task_allocation1` / `path_planning1`**。

```text
scene_io          ← 谁都不依赖（标准库）
ta1 / pp1         ← 各自独立安装、单独可用，互不 import
对接点            ← 同一套 ENU 米 + 同一个 frame_hash + bridges 里的懒加载桥
```

## 两条不变量

| # | 不变量 | 违反时 |
|---|---|---|
| 1 | 一个任务包只绑一个世界，`world_id + frame_hash` 显式声明 | `ValueError`，不静默跑错 |
| 2 | 经纬度是水平坐标真源，ENU 米由 `world.frame` 现算、不落库 | 写了 `x/y` 直接报错，提示回填经纬度 |
| 3 | `world.json` / `mission.json` 必带 `"schema": 1`（版本锚） | 缺或不为 1 → `ValueError`，提示先迁移再升级代码 |

## 目录约定

```text
data/
├─ worlds/<world_id>/               # 大、慢变
│  ├─ world.json                    # world_id、frame、高程基准、数据集指纹
│  ├─ region.geojson                # frame 的唯一真源
│  ├─ building/                     # 顶层 *.shp | *.geojson
│  ├─ dem.tif
│  └─ nofly.geojson                 # 可选
└─ missions/<mission_id>/           # 小、快变
   ├─ mission.json                  # world_id + frame_hash + current_time
   ├─ nests.json / uavs.json / tasks.json
   └─ baseline.json                 # 可选：回归基线
```

## 用法

```python
import scene_io

world = scene_io.load_world("data/worlds/guangzhou")          # 校验 frame ↔ region
m = scene_io.load_mission("data/missions/demo_001", world=world)

print(m.summary())                    # world_id / frame_hash / n_tasks / n_uavs / n_nests
print(m.tasks[0].x, m.tasks[0].y)     # 已换算成 ENU 米（纯 float，不是任何库的类型）
```

接 **ta1**（需要装了 ta1；`pip install -e "scene_io[ta1]"`）：

```python
import task_allocation1 as ta1
res = ta1.allocate(scene_io.to_ta1_request(m))
```

ta1 想用**垂直切面尺子**时，两个采样器也从世界现造（懒加载 pp1 + shapely）：

```python
cost = ta1.CostConfig(model="vertical_section", agl_min_m=10, agl_max_m=120, clearance_m=5)
ground_z, obstacle_top_z = scene_io.to_ta1_cost_samplers(world)   # 约 3.5 s 一次性建好
res = ta1.allocate(scene_io.to_ta1_request(m, cost_config=cost,
                                           ground_z=ground_z, obstacle_top_z=obstacle_top_z))
```

`mission.json` 里写 `"cost": {"model": "vertical_section", ...}` 时，`to_ta1_request` 会自动带上配置，采样器仍需调用方传。

接 **pp1**（`to_pp1_world_kwargs` 只回一个 dict，**不需要 import pp1**）：

```python
import path_planning1 as pp1
ctx = pp1.create_context({"world": scene_io.to_pp1_world_kwargs(world, max_od_length_m=3000.0)})
```

## 加载期会挡掉什么

`load_mission` 一次报出全部问题（不是逐条抛），抬头写清是哪个任务包（`任务包 <mission_id> 数据校验失败（N 项）`）：
`world_id` / `frame_hash` 不匹配、`schema` 不是 1、缺 `lon/lat`、
用了 `x/y`、缺 `z_abs`、`id` 重复、`uav.nest_id` 不存在、某巢归属机数超 `capacity`、
`time_base` / `z_basis` 取值不支持。

`load_world` 还会确认 `world.json` 的 frame 与 `region.geojson` 现算的 frame 一致。

## frame 口径（两侧必须一致）

```text
x = (lon - lon0) · 111320 · cos(lat_ref)      lat_ref = bbox 中心纬度
y = (lat - lat0) · 110574                     (lon0, lat0) = bbox 左下角
frame_hash = sha1(f"{lon0:.9f},{lat0:.9f},{lat_ref:.9f}")[:12]
```

本包独立实现了一份 3 行算术，**不共享 pp1 的代码**；两边是否真的一致由测试对拍
（`scene_io.check_world_with_pp1`）与 `frame_hash` 共同保证。

## 测试

```bash
uv run pytest scene_io/tests -q
```

其中三项专门守"解耦"：`pyproject` 无运行时依赖、`import scene_io` 不会拉起算法库、
算法库源码里不出现 `scene_io`。

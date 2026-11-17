# 4. 完整流程 Tutorial：从 0 到一次可交付的多机协同任务

> **背景**：拉萨市城关区（`lasa_chengguan`）
> **约定**：本文用 `/` 表示**项目根目录**（`06project-for-uav/`）。所有相对路径都从它算起。
> **目标**：数据准备 → 校验 → `ta1` 任务分配 → 展开航段 → `pp1` 逐段规划 → 回执复核 → 落盘。
> **可执行**：本文每一步都能直接复制运行；配套脚本是 [`docs/examples/run_lasa_demo.py`](examples/run_lasa_demo.py)，
> 示例数据已随仓库落在 `/data/worlds/lasa_chengguan/` 与 `/data/missions/lasa_demo_001/`。

按顺序读完你应该能：换一个城市、换一批任务，独立跑通整条链路，并且知道每个数字从哪来。

**目录**

| § | 内容 |
|---|---|
| 1 | 项目结构：每个包、每个目录是干什么的 |
| 2 | 从零安装（uv / pip 两条路）+ 验证 |
| 3 | 配置体系：三层配置、哪些配置**真的生效** |
| 4 | 数据准备（拉萨城关区）：region / 建筑 / DEM / 禁飞区 / world.json / 任务包 |
| 5 | 数据校验：`scene_io` 一次报全 + pp1 预检 |
| 6 | `ta1` 任务分配 |
| 7 | 展开为航段（**起降不进 pp1**，这是最容易踩的坑） |
| 8 | `pp1` 逐段规划：建场 / 产物复用 / 请求 / 失败码 |
| 9 | 两库对接：用真 `duration_s` 复核，闭环 |
| 10 | 一键跑通 + 输出解读 |
| 11 | 换成你自己的城市（清单） |
| 12 | 常见报错速查 |
| 13 | 下一步与文档地图 |

---

## 1. 项目结构：每个包、每个目录是干什么的

### 1.1 四个包

```text
/
├─ task_allocation1/     【ta1】分配层：谁飞、什么顺序、回哪个巢
├─ path_planning1/       【pp1】落地层：这一条航段到底怎么飞（航迹 + 时长 + 净空）
├─ path_planning2/       【pp2】M0 骨架，算法未实现，**不参与主链路**
└─ scene_io/             数据适配层：世界 / 任务包 → 纯数据（零依赖）
```

| 包 | 回答什么 | 依赖 | 一句话 |
|---|---|---|---|
| `task_allocation1` (ta1) | **谁飞 / 什么顺序 / 回哪个巢** | 只要 `numpy` | 纯计算，**不读文件、不联网**；同巢起降（终点恒=起飞巢） |
| `path_planning1` (pp1) | **这一条航段怎么飞** | `numpy scipy pillow shapely pyshp`（+ 可选 `casadi`） | 读世界目录；RRT + CasADi/IPOPT，目标是**最短飞行时间** |
| `path_planning2` (pp2) | （规划中的第二套算法） | — | M0 骨架，仅契约落地，**别用** |
| `scene_io` | **数据怎么落到磁盘、怎么读进来** | **无**（纯标准库） | 不 import ta1/pp1；对接靠数据契约 + 懒加载桥 |

**三件事先记住**（后面所有内容都建立在这三条上）：

1. **两库互不 import，各自可单独安装上线**。只用分配就只装 ta1，只用规划就只装 pp1。
2. **对接只走数据契约**：同一套 ENU 米、同一个 `frame_hash`、同一份 `world.json`。
3. **高度从头到尾只有一个口径：绝对 `z`**。`AGL`（真高）是 pp1 库内部按 DEM 派生的校验量，
   **不是接口字段、也不落库**。拉萨地面 ~3650 m，"离地 40 m" 就是 `z_abs ≈ 3690`，**不要传 40**。

### 1.2 目录树（数据 / 产物 / 结果）

```text
/
├─ data/                                 # 进版本库（建筑 shp 除外）
│  ├─ worlds/                            # 大、慢变：一城一份，被多个任务包共享
│  │  └─ lasa_chengguan/                 # ← 本教程的世界
│  │     ├─ world.json                   # world_id / frame / 高程基准 / 数据指纹
│  │     ├─ region.geojson               # 必需：frame 的**唯一真源**
│  │     ├─ building/                    # 必需：顶层 *.shp|*.geojson（**不递归**）
│  │     │  ├─ lasa_residential.geojson
│  │     │  ├─ lasa_commercial.geojson
│  │     │  └─ lasa_public_services.geojson
│  │     ├─ dem.tif                      # with_dem=true 时必需
│  │     └─ nofly.geojson                # 可选：静态禁飞区
│  ├─ missions/                          # 小、快变：一个任务包 = 一次分配的全部输入
│  │  └─ lasa_demo_001/                  # ← 本教程的任务包
│  │     ├─ mission.json                 # world_id + frame_hash + current_time + 配置
│  │     ├─ nests.json / uavs.json / tasks.json
│  │     └─ baseline.json                # 可选：回归基线
│  └─ contracts/                         # 两库互操作夹具（契约漂移测试用，不是业务数据）
│
├─ artifacts/                            # 不进版本库：建场产物（跟着世界走）
│  └─ lasa_chengguan/default/            # sdf.npy + occupancy.npy + meta.json
├─ results/                              # 不进版本库：你的输出
│  └─ lasa_chengguan/lasa_demo_001/      # run_001.json
│
├─ docs/                                 # 0 数据 → 1 ta1 → 2 pp1 → 3 对接 → 4 本篇
│  └─ examples/run_lasa_demo.py          # 本教程的可执行配套
└─ (四个包目录)
```

**四条归属规则**：世界目录名 = `world_id`；任务包只**引用**世界不复制世界；场产物跟世界走
（`artifacts/<world_id>/<label>/`）；结果跟世界+任务走（`results/<world_id>/<mission_id>/`）。

### 1.3 依赖关系

```text
scene_io ──(零依赖，不 import 下面两个)──┐
                                        │  数据契约：ENU 米 + frame_hash + world.json
ta1 ◄────────── 互不 import ──────────► pp1
```

`scene_io` 里的 `to_ta1_request()` / `to_ta1_cost_samplers()` 是**懒加载桥**：函数体里才 import，
装了哪个库才用得到哪座桥。所以只用 ta1 的服务不会背上 shapely/casadi。

---

## 2. 从零安装

### 2.1 环境要求

| 项 | 要求 | 说明 |
|---|---|---|
| Python | **3.11 或 3.12** | 仓库钉死 3.11（`.python-version`），casadi 的 Windows wheel 覆盖更稳 |
| 内存 | ≥ 4 GB 可用 | 全域场建场期峰值；小城市 250 MB 级，大城市可达 GB 级 |
| 磁盘 | ≥ 1 GB | 建筑数据 + 场产物 |
| 网络 | 首次装依赖需要 | 之后离线可跑 |

> 只用 `ta1`（不规划路径）的话，只要 `numpy`，1 核笔记本也能跑。

### 2.2 用 uv（推荐）

```bash
cd /
uv sync --all-packages        # ← 这个 flag 不能省，根是 virtual workspace
uv run pytest                 # 280 个用例
uv run ruff check .           # 代码风格
```

`--all-packages` 是关键：根 `pyproject.toml` 是 **virtual workspace**（`package = false`），
不带这个 flag 装不到成员包的依赖，`import task_allocation1` 会直接失败。

### 2.3 没有 uv：用 venv + pip

```bash
cd /
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -U pip

# 按需装：只用分配 → 只装 ta1；要规划 → 加 pp1；数据层零依赖但建议一起装
pip install -e ./task_allocation1
pip install -e ./path_planning1            # 不含 NLP 求解器
pip install -e "./path_planning1[nlp]"     # 要轨迹（时长/速度剖面）就加上 casadi
pip install -e ./scene_io

pip install pytest ruff                    # 开发期才需要
```

国内网络慢可加镜像：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ...`

**三种装法的区别**：

| 装法 | 得到什么 | 什么时候用 |
|---|---|---|
| 只装 `task_allocation1` | 只能分配，米数走欧氏/垂直切面 | 只要"谁飞" |
| `path_planning1` | 几何层完整（折线 + 真几何净空） | 只要"能不能飞、多远" |
| `path_planning1[nlp]` | + 时间域轨迹（`duration_s` / 速度 / 推力） | **要时长就必须装**，否则 `trajectory` 恒为 `None` |

### 2.4 验证安装

```bash
cd /
python -c "import task_allocation1 as ta1; print(ta1.__version__, ta1.ALGORITHM_ID)"
python -c "import path_planning1 as pp1;  print(pp1.__version__, pp1.ALGORITHM_ID)"
python -c "import scene_io;               print(scene_io.__version__)"
python -m pytest -q
```

预期输出（版本号以你装的为准）：

```text
0.2.0 ta1-dmde-home-nest
0.4.0 pp1-rrt-nlp-time
0.2.0
280 个用例：277 passed + 3 skipped（1 个慢测 + 2 个需 guangzhou shp 夹具）
```

> **新 clone 会有 3 个测试跳过**：`scene_io/tests/test_scene_io_contract.py` 的两条对拍用例需要
> `data/worlds/guangzhou/building/` 的 5 个 shp（约 26 MB，按 `.gitignore` 排除，需自备），
> 缺数据时**跳过而不是红**。本教程的 `lasa_chengguan` 建筑是 GeoJSON（211 KB），**已随仓库提供**，
> `docs/examples/run_lasa_demo.py` clone 下来就能直接跑。

---

## 3. 配置体系

### 3.1 三层配置

| 层 | 在哪 | 谁消费 | 什么时候定 |
|---|---|---|---|
| **世界配置** | `data/worlds/<id>/world.json` + `create_context` 的 `world` 段 | pp1 | 建世界时定，季度级 |
| **参数配置** | `params` 段（走廊 / 栅格 / 飞行带 / 机型 / 算法） | pp1 | 启动时定，`create_context` 一次固定 |
| **请求配置** | 每条请求的端点 + `seed` | pp1 | 每条请求 |

ta1 的配置在**任务包** `mission.json` 的 `solver` / `problem` / `cost` 三段，
或直接构造 `SolverConfig` / `ProblemConfig` / `CostConfig` 对象。

**pp1 的一份配置长这样**（`/docs/pp1_full_config.example.json` 是全量版，这里是最小可用版）：

```json
{
  "world": {
    "root": "data/worlds/lasa_chengguan",
    "with_dem": true,
    "field_mode": "startup",
    "max_od_length_m": 1500.0,
    "nofly_path": "nofly.geojson",
    "nofly_altitude_mode": "absolute"
  },
  "params": {
    "roi":       {"width_ratio": 0.2, "margin_m": 50.0},
    "spacing_m": [4.0, 4.0, 2.0],
    "agl_min_m": 10.0, "agl_max_m": 120.0,
    "seed": 0,
    "vehicle":   {"v_max_mps": 20.0, "a_max_mps2": 3.0, "clearance_m": 5.0,
                  "thrust_max_n": 100.0, "mass_kg": 5.0}
  }
}
```

**`max_od_length_m` 是成本旋钮**：它派生建场外扩 `pad_m`，调大它 → 场变大 → 内存/建场时间涨。
本教程设 `1500` → `pad_m = 244` → 场 255 MB、建场 17 s。设成 `3000` 会让 pad 到 ~392 m、
场体积涨到约 2.5 倍。**够用就好**（它只要 ≥ 你最长的一条航段）。

### 3.2 哪些配置**真的生效**（重要）

ta1 的配置面比实际消费的宽。下表是**唯一权威清单**：

| 配置 | 真正生效 | **预留（写了没用）** |
|---|---|---|
| `ConstraintConfig` | `duplicate_task` `range` `endurance` `time_window` `nest_capacity` `nest_availability` `end_nest_reachability` | `require_all_tasks` `require_no_idle` `sequence` `sync` 及全部 `w_*` 权重 |
| `ProblemConfig` | `lambda_cost` `w_violation` `hard_violation_kappa` `default_service_time` `current_time` `big_m` + 三个可选权重覆盖 | `allow_unassigned` `allow_idle_uav` `late_reward_decay_seconds` `coverage_audit` |
| `SolverConfig` | 其余全部 | `coverage_first` `demand_serve_margin` `polish_eps` |
| `Task` | `id` `x/y/z` `reward` `earliest` `latest` | `sequence_group` `sequence_order` |

> 同巢起降约束（`home_return`）**恒开、只读**，没有关闭入口。
> 需要"任务 A 必须在任务 B 之前"或"多机同步到达"，请在上层拆成多次调用，**别指望 `sequence`/`sync`**。

pp1 的配置**每一项都生效**（有 `test_pp1_defaults.py` 逐项守）。

---

## 4. 数据准备（拉萨市城关区）

### 4.0 先定运营区

| 项 | 值 |
|---|---|
| 城市 | 拉萨市城关区（核心建成区） |
| 经纬度范围 | `lon 91.136 – 91.172`，`lat 29.650 – 29.672` |
| 换算后 | **3482.4 m × 2432.6 m**，对角线 4248 m |
| 高程基准 | 拉萨河谷 **~3650 m**（DEM 实测 3650.0 – 3674.6 m） |
| frame 锚点 | bbox **左下角** `(91.136, 29.650)` |
| `lat_ref` | bbox 中心纬度 `29.661` |

局地 ENU 换算公式（`scene_io.world.Frame` 与 pp1 的 `LocalFrame` 同口径，**不要各算各的**）：

```text
x = (lon − lon0) × 111320 × cos(lat_ref)     = (lon − 91.136) × 96734.5
y = (lat − lat0) × 110574                    = (lat − 29.650) × 110574
```

`frame_hash = sha1("91.136000000,29.650000000,29.661000000")[:12] = 0af4e4fb592e`。
**任何一侧要自算 `frame_hash` 都必须用这一行**，否则两库永远对不上。

> 局地切平面近似只适用于跨度 **< 25 km**。城关区 4.2 km 对角线，绰绰有余。
> 别直接甩整个行政区——bbox 是外接矩形，狭长区会把不该算的地方也算进来。

### 4.1 `region.geojson`（frame 的唯一真源）

```json
{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature",
    "properties": {"name": "lasa_chengguan_core"},
    "geometry": {"type": "Polygon", "coordinates": [[
      [91.136, 29.650], [91.172, 29.650],
      [91.172, 29.672], [91.136, 29.672], [91.136, 29.650]
    ]]}
  }]
}
```

- 格式 GeoJSON `FeatureCollection`，几何 Polygon / MultiPolygon（取并集**外接矩形**）；
- 坐标 **EPSG:4326，顺序 `[lon, lat]`**——写反了**不会报错**，一定要肉眼在地图上确认一次；
- `frame` 原点 = bbox 左下角，`bounds_m` 与全域场水平范围也由它决定。

### 4.2 `building/`（建筑）

**两种格式二选一**，不要同一批楼同时放两份（会被读两遍、楼算两层）：

| | Shapefile | GeoJSON |
|---|---|---|
| 文件 | `.shp` + `.shx` + `.dbf`（DBF 为 **GBK**） | 单个 `.geojson`，UTF-8 |
| 扫描 | `building/` **顶层，不递归** | 同左 |
| 高度字段 | DBF 字段 `Height`（大小写敏感） | `properties.Height` |
| 高度含义 | **米、相对地面、有限正数** | 同左 |

本教程用 GeoJSON（3 个图层，329 栋）：

| 文件 | 栋数 | 高度范围 |
|---|---:|---|
| `lasa_residential.geojson` | 186 | 6 – 24 m |
| `lasa_commercial.geojson` | 89 | 12 – 46 m |
| `lasa_public_services.geojson` | 54 | 9 – 30 m |

最小示例（高度在 `properties` 里，**不是顶层字段**）：

```json
{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature",
    "properties": {"Height": 18.5, "kind": "residential"},
    "geometry": {"type": "Polygon", "coordinates": [[
      [91.1420, 29.6630], [91.1424, 29.6630],
      [91.1424, 29.6633], [91.1420, 29.6633], [91.1420, 29.6630]
    ]]}
  }]
}
```

**高度口径只有一条**（这条错了整个净空都错）：

```text
Height    = 建筑**相对地面**的高度（米，> 0）
base_z_m  = footprint 顶点 + 中心处 DEM 的最大值（库内算）
z_top_m   = base_z_m + Height
```

源数据三种常见坏情况：

| 源情况 | 处理 |
|---|---|
| 只有楼层数 `building:levels` | `Height = levels × 3.0`，并在数据说明里写下这个假设 |
| 给的是"楼顶海拔" | **先减去该楼地面高程**再写进 `Height`，否则会被 DEM 再加一次 |
| 空值 / `"12m"` / `"12.5米"` / `NaN` | 无法解析 → 启动即报错，必须清洗成纯数字 |

用 shapefile 的生产路径（`ogr2ogr` 转投影 + GBK 属性表）：

```bash
ogr2ogr -t_srs EPSG:4326 -lco ENCODING=GBK building/lasa_residential.shp in_utm_residential.shp
ogrinfo -al -so building/lasa_residential.shp        # 复核字段名与坐标范围
```

### 4.3 `dem.tif`

| 项 | 要求 | 本教程 |
|---|---|---|
| 格式 | GeoTIFF，单波段 | ✓ |
| 坐标 | EPSG:4326 | ✓ |
| 标签 | **必须保留 `33550`（PixelScale）+ `33922`（ModelTiepoint）** | ✓ |
| 单位 | 米 | ✓ |
| 分辨率 | 城市低空建议 5–30 m | ~15 m（233 × 163） |
| 覆盖 | 必须盖住 `region bbox + field_pad` + **全部建筑** | ✓ |
| 高程 | 与建筑 `base_z`、禁飞区绝对 `z` **同一基准** | 3650.0 – 3674.6 m |

```bash
gdalwarp -t_srs EPSG:4326 -r bilinear in_dem_lasa.tif data/worlds/lasa_chengguan/dem.tif
gdalinfo data/worlds/lasa_chengguan/dem.tif          # 看 33550 / 33922 两个标签在不在
```

> **域外 clamp 是静默错误**：DEM 没盖住某栋楼时不会报错，只会拿边缘值当地面，楼底矮一截。
> 覆盖检查必须在**数据侧**做，别指望库报错。

### 4.4 `nofly.geojson`（可选）

```json
{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature",
    "properties": {"name": "potala_no_fly", "z_min_m": 3650.0, "z_max_m": 3760.0},
    "geometry": {"type": "Polygon", "coordinates": [[
      [91.15825, 29.66313], [91.16347, 29.66313],
      [91.16347, 29.66769], [91.15825, 29.66769], [91.15825, 29.66313]
    ]]}
  }]
}
```

| 项 | 要求 |
|---|---|
| 几何 | Polygon / MultiPolygon，**不支持洞**（有洞请拆成多个） |
| `absolute` 模式（默认） | `properties.z_min_m` / `z_max_m`：**绝对高程** |
| `agl` 模式 | `properties.agl_min_m` / `agl_max_m`：**真高**，必须同时给且 `with_dem=true` |
| 约束 | 两种字段**不可混用**；各多边形必须 `min < max` |

**高海拔地区的头号坑**：`absolute` 模式下 `z_min_m / z_max_m` 是**绝对高程**。
拉萨地面 3650 m，照默认写 `0 / 1000` 的话禁飞层整个落在地面以下，**什么也挡不住**。
本教程写 `3650 / 3760`（地面以上 0–110 m）。

禁飞棱柱可以**悬浮**，上下都可绕；端点落进棱柱 → `endpoint_in_nofly`。
没有禁飞数据就**不要放文件**，放个空的/错的反而报错。

### 4.5 `world.json`

```json
{
  "schema": 1,
  "world_id": "lasa_chengguan",
  "frame": {
    "lon0": 91.136, "lat0": 29.650, "lat_ref": 29.661,
    "frame_hash": "0af4e4fb592e"
  },
  "elevation": {"basis": "absolute_enu_z", "unit": "m"},
  "contents": {
    "region": "region.geojson",
    "buildings": "building",
    "dem": "dem.tif",
    "nofly": "nofly.geojson"
  },
  "dataset_digest": "e9cebbed678f"
}
```

| 字段 | 说明 |
|---|---|
| `schema` | 版本锚，必须是 `1`（升级要写迁移，不靠人记） |
| `frame.frame_hash` | 必须等于按 `frame` 现算的值，`load_world` 会验 |
| `contents` | 四类数据文件的名字；`nofly` 可省 |
| `dataset_digest` | 建筑/DEM/禁飞区**内容指纹**，缓存键的第一段 |

`dataset_digest` **要用 pp1 算**（scene_io 不重复实现，要 shapely/pyshp 会破坏零依赖）：

```python
from path_planning1.core.dataset_identity import dataset_digest
print(dataset_digest({"root": "data/worlds/lasa_chengguan", "with_dem": True}))
# e9cebbed678f
```

> 不要手写 `dataset_digest`，也不要自己拼 `world_id` 当缓存键。改了任何数据文件它都会变，
> 场产物与缓存随之整体过期——这正是它的用途。

### 4.6 任务包 `data/missions/lasa_demo_001/`

#### `mission.json`

```json
{
  "schema": 1,
  "mission_id": "lasa_demo_001",
  "world_id": "lasa_chengguan",
  "frame_hash": "0af4e4fb592e",
  "current_time": 0.0,
  "time_base": "relative_s",
  "z_basis": "absolute_enu_z",
  "solver":  {"pop_size": 24, "max_generations": 60, "seed": 0},
  "problem": {"default_service_time": 30.0},
  "cost":    {"model": "euclid3d"}
}
```

- `world_id` + `frame_hash` **双重绑定**：任务包不是用这个世界的 frame 算的，`load_mission` 直接报错；
- `solver` / `problem` / `cost` 三段分别透传 `SolverConfig` / `ProblemConfig` / `CostConfig`；
- 想用垂直切面尺子（考虑地形与建筑的航程估算）就把 `cost` 换成：

```json
"cost": {"model": "vertical_section", "agl_min_m": 10, "agl_max_m": 120,
         "clearance_m": 5, "climb_rate": 0.15, "num_samples": 200}
```

（此时 `to_ta1_request` 还要给两个采样器，见 §6.3。）

#### `nests.json` / `uavs.json` / `tasks.json`

**水平坐标只认 `lon` / `lat`**；写 `x` / `y` 会被加载器拒绝（提示用 `frame.to_lonlat` 反算）。
`z_abs` 是**绝对高程**。

```json
[
  {"id": 0, "name": "nest_chengguan_north", "lon": 91.1420, "lat": 29.6650,
   "z_abs": 3665.33, "capacity": 2, "available": true},
  {"id": 1, "name": "nest_chengguan_south", "lon": 91.1640, "lat": 29.6540,
   "z_abs": 3682.05, "capacity": 2, "available": true}
]
```

```json
[
  {"id": 0, "nest_id": 0, "lon": 91.1420, "lat": 29.6650, "z_abs": 3665.33,
   "speed": 15.0, "remaining_range": 22000.0, "max_time": 2400.0, "status": "IDLE"},
  {"id": 1, "nest_id": 0, "lon": 91.1420, "lat": 29.6650, "z_abs": 3665.33,
   "speed": 15.0, "remaining_range": 22000.0, "max_time": 2400.0, "status": "IDLE"},
  {"id": 2, "nest_id": 1, "lon": 91.1640, "lat": 29.6540, "z_abs": 3682.05,
   "speed": 12.0, "remaining_range": 18000.0, "max_time": 2000.0, "status": "IDLE"},
  {"id": 3, "nest_id": 1, "lon": 91.1640, "lat": 29.6540, "z_abs": 3682.05,
   "speed": 12.0, "remaining_range": 18000.0, "max_time": 2000.0, "status": "IDLE"}
]
```

```json
[
  {"id": 101, "lon": 91.1445, "lat": 29.6685, "z_abs": 3698.35, "reward": 1.0, "earliest": 0.0,    "latest": 1500.0},
  {"id": 102, "lon": 91.1500, "lat": 29.6625, "z_abs": 3692.87, "reward": 1.2, "earliest": 0.0,    "latest": 1500.0},
  {"id": 103, "lon": 91.1555, "lat": 29.6690, "z_abs": 3691.82, "reward": 1.0, "earliest": 200.0,  "latest": 1800.0},
  {"id": 104, "lon": 91.1610, "lat": 29.6585, "z_abs": 3702.13, "reward": 1.5, "earliest": 0.0,    "latest": 1600.0},
  {"id": 105, "lon": 91.1670, "lat": 29.6655, "z_abs": 3700.09, "reward": 1.0, "earliest": 0.0,    "latest": 1700.0},
  {"id": 106, "lon": 91.1695, "lat": 29.6565, "z_abs": 3708.92, "reward": 1.1, "earliest": 0.0,    "latest": 1700.0}
]
```

#### `z_abs` 怎么定（拉萨必须看这条）

**`z_abs = 该点地面高程 + 期望真高（AGL）`**。这是数据侧**唯一**要做这个加法的地方：

| 对象 | 期望真高 | 本例 `z_abs` | 说明 |
|---|---|---|---|
| 巢 / 无人机 | 离地 12 m | `3665.33` / `3682.05` | 起飞点；pp1 的飞行带下限是 **离地 10 m** |
| 任务点 | 离地 40 m | `3691.82 – 3708.92` | 巡航高度 |

> **拉萨地面 3650 m，"离地 40 m" = `z_abs ≈ 3690`，不是 40。** 传 40 会被判
> `out_of_flight_band`（真高为负）。pp1 收到绝对 z 之后自己按 DEM 派生真高，**你不要再减一次地面**。

#### ENU 对照表（跑通后可以核对）

| 对象 | lon | lat | ENU x | ENU y | z_abs | 地面 |
|---|---|---|---:|---:|---:|---:|
| nest 0 / uav 0,1 | 91.142000 | 29.665000 | 580.4 | 1658.6 | 3665.33 | 3653.3 |
| nest 1 / uav 2,3 | 91.164000 | 29.654000 | 2708.5 | 442.3 | 3682.05 | 3670.1 |
| task 101 | 91.144500 | 29.668500 | 822.2 | 2045.6 | 3698.35 | 3658.4 |
| task 102 | 91.150000 | 29.662500 | 1354.3 | 1382.2 | 3692.87 | 3652.9 |
| task 103 | 91.155500 | 29.669000 | 1886.3 | 2100.9 | 3691.82 | 3651.8 |
| task 104 | 91.161000 | 29.658500 | 2418.3 | 939.9 | 3702.13 | 3662.1 |
| task 105 | 91.167000 | 29.665500 | 2998.7 | 1713.9 | 3700.09 | 3660.1 |
| task 106 | 91.169500 | 29.656500 | 3240.6 | 718.7 | 3708.92 | 3668.9 |

#### 同巢起降对数据的 4 条硬要求

| # | 要求 | 不满足会怎样 |
|---|---|---|
| 1 | 每架机 `nest_id` 非空且该巢存在 | 装不出航次终点，该机不进解 |
| 2 | `uav` 经纬度 = 所属巢坐标 | 不报错，但图上不闭合、航程账偏小 |
| 3 | 每巢归属机数 ≤ `capacity`（含 `BUSY/CHARGING/FAULT`） | `nest_overcapacity_<id>`，解不可行 |
| 4 | 一架机只属于一个巢 | 没有跨巢轮转；换基地就改 `nest_id` 再调一次 |

---

## 5. 数据校验

### 5.1 `scene_io` 一次报全

```python
import scene_io

world = scene_io.load_world("data/worlds/lasa_chengguan")
m = scene_io.load_mission("data/missions/lasa_demo_001", world=world)
print(m.summary())
```

```text
{'mission_id': 'lasa_demo_001', 'world_id': 'lasa_chengguan',
 'frame_hash': '0af4e4fb592e', 'n_tasks': 6, 'n_uavs': 4, 'n_nests': 2,
 'current_time': 0.0}
```

`load_world` 会验两件事（任一失败即报错）：

1. `world.json` 的 `frame_hash` == 按其 `frame` 现算的值；
2. 该 frame == 从 `region.geojson` 现算的 frame（**frame 唯一真源是 region**）。

`load_mission` **一次报出全部数据问题**（不逐条抛）：

| 检查 | 报错示例 |
|---|---|
| `world_id` 不符 | `world_id 不匹配：mission='x' ↔ world='lasa_chengguan'` |
| `frame_hash` 对不上 | `任务包不是用这个世界的 frame 算的，必须先重算坐标` |
| 用了 `x/y` | `tasks.json[0] 用了 x/y（ENU 米）：任务包只收 lon/lat` |
| 缺 `z_abs` | `缺少 z_abs（绝对高程，米；不是 AGL）` |
| `id` 重复 / `nest_id` 不存在 | 逐条列出 |
| 巢超容量 | `巢 0 归属 3 架机，超过 capacity=2` |

### 5.2 深度对拍（可选，CI / 入库时跑）

```python
print(scene_io.check_world_with_pp1(world))
# {'frame_ok': True, 'digest_ok': True, 'frame_pp1': (...), 'digest_pp1': 'e9cebbed678f'}
```

它用 pp1 自己的实现复核 frame 与数据指纹。`scene_io` **不重复实现**这两个算法
（要 shapely/pyshp，会破坏零依赖），所以两边一致靠**对拍**，不靠共享代码。

### 5.3 pp1 预检（不建场，看数据长什么样）

```python
import math
from path_planning1.core.city_dataset import load_city
from path_planning1.core.params_config import world_kwargs_from_config

cfg = {"world": {"root": "data/worlds/lasa_chengguan", "with_dem": True,
                 "nofly_path": "nofly.geojson"}}
ds = load_city(**world_kwargs_from_config(cfg))
print("buildings:", len(ds.buildings))
print("bounds_m:", [round(v, 1) for v in ds.bounds_m])
print("diag_m:", round(math.dist(ds.bounds_m[:2], ds.bounds_m[2:]), 1))
print("height_m:", round(min(b.height_m for b in ds.buildings), 1),
      "-", round(max(b.height_m for b in ds.buildings), 1))
print("dem:", ds.dem.z_m.shape, round(float(ds.dem.z_m.min()), 1),
      "-", round(float(ds.dem.z_m.max()), 1))
```

```text
buildings: 329
bounds_m: [0.0, 0.0, 3482.4, 2432.6]
diag_m: 4247.8
height_m: 6.0 - 46.0
dem: (163, 233) 3650.0 - 3674.6
```

**判读标准**：

- `diag_m > 25000` → 拆子区，别塞进一个 frame；
- `height_m` 上千 → 把"楼顶海拔"写进了 `Height`（本例 6–46 m，合理）；
- `dem` 高程与当地同量级（拉萨 3650 m，不是 0–100，也不是英尺）；
- `bounds_m` 与运营需求匹配。

### 5.4 自检清单

- [ ] `region.geojson` 是 EPSG:4326、`[lon, lat]`、bbox 紧凑、肉眼看过地图
- [ ] `building/` 顶层文件，`.shp/.shx/.dbf` 齐全且 DBF 为 GBK；或 `.geojson`（UTF-8）
- [ ] 没有同一批楼同时存在 `.shp` 与 `.geojson`；没有放子目录
- [ ] `Height` 字段名对得上、有限正数、米、**相对地面**
- [ ] `dem.tif` 含 `33550` / `33922`、米制、覆盖 bbox + pad + 全部建筑
- [ ] `nofly.geojson`（如有）无洞、字段不混用、`min < max`、**绝对高程别写成 AGL**
- [ ] 所有 `z_abs` = 地面 + 期望真高，落在飞行带内（拉萨：3660 – 3790）
- [ ] `scene_io.load_mission(...)` 能过
- [ ] `world.json` 的 `dataset_digest` 用 pp1 现算过，不是手写的

---

## 6. `ta1`：任务分配

### 6.1 最小调用

```python
import scene_io
import task_allocation1 as ta1

world = scene_io.load_world("data/worlds/lasa_chengguan")
m = scene_io.load_mission("data/missions/lasa_demo_001", world=world)

res = ta1.allocate(scene_io.to_ta1_request(m))
print(res.home_return_ok, res.solution.feasible)
for t in res.solution.tours:
    print(f"UAV {t.uav_id}: nest {t.start_nest_id} -> {t.task_ids} -> nest {t.end_nest_id}")
print(res.metrics_dict())
```

实测输出（`seed=0`，同种子逐位可复现）：

```text
耗时 1.15s   model=srp   n_evals=1464
home_return_ok = True   feasible = True
  UAV 0: nest 0 -> [102, 103, 101] -> nest 0   est_cost=3239.5 m  est_reward=3.2
  UAV 1: (闲置)
  UAV 2: nest 1 -> [106, 105, 104] -> nest 1   est_cost=3168.2 m  est_reward=3.6
  UAV 3: (闲置)
{'n_participating': 4, 'model_type': 'srp', 'n_evals': 1464, 'n_executed': 6,
 'path_m': 6407.68, 'return_leg_m': 1033.92, 'f1_reward': 6.8, 'f2_cost': 6407.68,
 'feasible': True, 'home_return_ok': True}
```

> **同巢不变量**：每条非空航次 `end_nest_id == start_nest_id == uav.nest_id`。
> 自己校验：`all(t.is_home_return() for t in res.solution.tours)`。
> 外部改 `tour.end_nest_id` 也逃不过校验，解会被判不可行（`home_return_violation`）。

**算法在做什么**（三段）：

1. **DMDE 进化搜索**：种群 24 × 60 代（本例），差分在**代价空间**做，反映射时按温度退火；
2. **解码修复链**：合并 → 航程修复 → 改派 → 覆盖补全 → 时间窗重排；
3. **交付后局部优化**：跨机 relocate/swap + 组内 2-opt，四道门（航程/迟到/覆盖/成本）。

适应度是 `f = W_hard·n_hard + w_exc·超程 + W_skip·n_skip + w_late·迟到 + λ·航程`
——**完成是约束、成本是目标**（F1 只作事后指标）。

### 6.2 结果怎么读

| 字段 | 含义 |
|---|---|
| `solution.tours[i].task_ids` | 该机按顺序执行的任务（**含顺序语义**） |
| `start_nest_id` / `end_nest_id` | 同巢：两者恒等，且 == `uav.nest_id` |
| `executed_task_ids` | 本周期交付的任务集合 |
| `metrics_dict()["path_m"]` | 代理航程（排序用，**不是交付值**） |
| `metrics_dict()["return_leg_m"]` | 其中"末任务 → 起飞巢"返航腿的米数 |
| `home_return_ok` | 同巢不变量自检 |
| `diagnostics["gate_stats"]` | 覆盖门禁：`n_skip`（能装却没装）/ `n_shortfall`（物理装不下） |

**三种情况要分清**：

1. `feasible=True` → 硬约束全过，可交付；
2. `feasible=False` + `n_hard>0` → 求解器**如实报出违约**（紧航程/紧时间窗常见），不是漏检；
3. `model_type="empty"` → 无任务或无可用机。

### 6.3 换成垂直切面尺子（考虑地形与建筑）

```python
world = scene_io.load_world("data/worlds/lasa_chengguan")
m = scene_io.load_mission("data/missions/lasa_demo_001", world=world)
ground_z, obstacle_top_z = scene_io.to_ta1_cost_samplers(world)   # 约 3.5 s 建好

res = ta1.allocate(scene_io.to_ta1_request(
    m,
    cost_config=ta1.CostConfig(model="vertical_section",
                               agl_min_m=10, agl_max_m=120,
                               clearance_m=5, climb_rate=0.15),
    ground_z=ground_z, obstacle_top_z=obstacle_top_z,
))
```

两档尺子的差别只在**有地形/建筑**时出现：楼越高、飞行带越低，爬升吃掉的水平距离越多。
它仍只是**分配代理**（只沿连线竖直翻越、不横向绕行），交付口径永远是 pp1 的真航迹。

> ⚠ 注入自己的 `distance_fn(pos_a, pos_b) -> 米`（**有向**）会覆盖一切。
> 距离函数**必须纯、稳定、无随机**；它在某个坐标对上失败时库会**直接抛出**，
> 不会悄悄退回欧氏——口径错误必须暴露。

---

## 7. 展开为航段（**起降不进 pp1**）

这是整条链路**最容易踩的坑**。ta1 的一条 tour 是：

```text
巢 → t1 → t2 → t3 → 巢        （同巢起降，首尾都是巢）
```

但 pp1 的飞行带是 **离地 10–120 m**。巢在地面，直接把 `巢 → t1` 喂给 pp1 会得到
`out_of_flight_band`（真高 12 m 其实可以，但**垂直起降段**不是 pp1 的事）。

正确做法：**每个航路点都按"巡航真高"抬到绝对 z，只把水平位移交给 pp1**：

```python
def expand_tours(res, mission):
    by_task = {t.id: t for t in mission.tasks}
    by_nest = {n.id: n for n in mission.nests}
    segs = []
    for tour in res.solution.tours:
        if not tour.task_ids:
            continue
        pts  = [by_nest[tour.start_nest_id]]
        pts += [by_task[i] for i in tour.task_ids]
        pts.append(by_nest[tour.end_nest_id])
        for a, b in itertools.pairwise(pts):
            segs.append({"uav_id": tour.uav_id,
                         "from": {"id": a.id, "x": a.x, "y": a.y, "z": a.z},
                         "to":   {"id": b.id, "x": b.x, "y": b.y, "z": b.z}})
    return segs
```

本例 2 条 tour → **8 条空中航段**：

```text
UAV0  nest0 → 102 → 103 → 101 → nest0
UAV2  nest1 → 106 → 105 → 104 → nest1
```

垂直起降段（离地 0 → 12 m 的爬升、最后的下降）由上层自己管，**不进 `ctx.plan`**。
如果任务要求"离地 25 m 巡航"，就在这一步把 z 抬到 `地面 + 25`，交给 pp1 之后**没有再换算的环节**。

---

## 8. `pp1`：逐段规划

### 8.1 建全域场（进程启动一次）

```python
import path_planning1 as pp1

cfg = {
    "world": scene_io.to_pp1_world_kwargs(
        world, max_od_length_m=1500.0, nofly_altitude_mode="absolute"),
    "params": {},
}
ctx = pp1.create_context(cfg)
```

实测输出：

```text
建场耗时 0.0s(装载) + 17.3s(建场)  field_source=built
dataset_digest = e9cebbed678f
bounds_m = [0.0, 0.0, 3482.4, 2432.6]  max_od_length_m = 1500  pad_m = 244
全域场 shape=(994, 732, 70)  内存 255 MB
```

**"离线一次、在线只取"是结构保证**：`ctx.plan` 的签名里没有"不传场"的入口，
调用方不可能忘记把场传下去。`create_context` 会做**建场契约自检**——
契约允许的最坏 OD 的窗口必须落在场内，不够就**在启动阶段炸**，不留到请求期。

> `field_mode` 两档：`startup`（默认，启动建一次）/ `on_demand`（每条请求窗口现算，
> 低内存调试用）。生产一律 `startup`。

### 8.2 场产物复用（毫秒级启动、多进程共享）

```python
# 构建期：把刚算好的场落成产物
ctx.save_field("artifacts/lasa_chengguan/default")

# 下次启动（或另一个进程）：mmap 只读映射，不重建
ctx2 = pp1.create_context(cfg, field_artifact="artifacts/lasa_chengguan/default")
print(ctx2.field_source, ctx2.build_s)      # artifact, 0.00x s
```

产物是 `sdf.npy` + `occupancy.npy` + `meta.json`，**只读 mmap**，同机多进程共享物理内存。
数据版本 / 建场契约 / 场身份都会被校验，不一致就在启动期抛 `pp1.FieldContractError`
（**部署事故**，别映射成 400）。

实测（同一台机器）：

```text
现场建场  : 17.3 s   field_source=built
读产物    : 0.18 s   field_source=artifact   build_s=0.139 s
复核一跳  : ok  len=850.4 m  dur=49.24 s  clear=29.6 m   ← 与现场建场结果一致
```

### 8.3 逐段请求

```python
plan = ctx.plan((x0, y0), (x1, y1), z0, goal_z_m=z1)
print(plan.status.value, plan.failure_code.value)
print(plan.metrics_dict())
```

`start_xy / goal_xy` 是 ENU 米，`start_z_m / goal_z_m` 是**绝对 z**。

实测 8 段全绿：

| # | UAV | 航段 | status | length_m | min_clearance_m | duration_s | 耗时 |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | 0 | nest0 → 102 | ok | 822.2 | 17.01 | 47.83 | 3.0 s |
| 2 | 0 | 102 → 103 | ok | 894.2 | 24.60 | 51.44 | 0.6 s |
| 3 | 0 | 103 → 101 | ok | 1065.5 | 25.80 | 60.01 | 0.6 s |
| 4 | 0 | 101 → nest0 | ok | 457.5 | 30.0 | 29.59 | 0.6 s |
| 5 | 2 | nest1 → 106 | ok | 601.6 | 9.63 | 36.80 | 0.9 s |
| 6 | 2 | 106 → 105 | ok | 1024.6 | 23.24 | 62.87 | 0.8 s |
| 7 | 2 | 105 → 104 | ok | 967.5 | 30.0 | 55.10 | 0.8 s |
| 8 | 2 | 104 → nest1 | ok | 591.7 | 8.43 | 35.72 | 0.7 s |

**读数说明**：

- `length_m` 是**几何路径长度**（RRT 折线），不是轨迹弧长；
- `min_clearance_m` 是**真几何**净空（shapely 量棱柱），不是 SDF 读数；
  **报 30.0 表示"至少 30 m"**（量程上限 `CLEARANCE_SEARCH_RADIUS_M = 30`）；
- `duration_s` 来自**时间域轨迹**（NLP，最短飞行时间），没有轨迹时是 `None`——
  **绝不拿"路径长度 / 某个默认速度"凑一个假时长**；
- 首段 3.0 s 是 CasADi 首次 JIT/建模开销，之后每段 0.6–0.9 s。

### 8.4 结果字段

```python
plan.status          # ok / infeasible
plan.failure_code    # none / out_of_flight_band / rrt_no_path / ...
plan.path_enu        # (K,3) 几何折线（兜底交付物）
plan.trajectory      # 时空轨迹 x/v/a + 均匀 Δt（主交付物；没装 casadi 则恒 None）
plan.metrics         # length_m / min_clearance_m / duration_s / max_airspeed_mps / ...
plan.stats           # 窗口与几何统计（corridor_width_m / agl_min_m / thrust_ratio / ...）
plan.diagnostics     # solver_status / candidates / warnings / params_hash
plan.params_hash     # 口径指纹（缓存键要用）
```

**两个几何产物别混**：`path_enu` 是 RRT 折线（可飞性由净空闸保证），
`trajectory` 是要飞的那条线（C¹ 连续、走廊半宽是硬约束）。有轨迹时报的是轨迹的真值。

### 8.5 失败码怎么办

| failure_code | 含义 | 处理 |
|---|---|---|
| `out_of_flight_band` | 端点 z 换算出的真高不在 `[10, 120]` | 看报错里的 z / 地面 / AGL 三个数 |
| `outside_world` | 端点超出 `bounds_m` | 查坐标换算 |
| `od_too_long` | 单段超过 `max_od_length_m` | 拆航段，或调大它（会放大场） |
| `endpoint_in_nofly` | 端点落进禁飞棱柱 | 改端点；查禁飞**绝对 z** 是不是写成了 AGL |
| `start_or_goal_blocked` | 端点 SDF 净空不足 | 换端点 |
| `rrt_no_path` | 5 个初值都找不到满足净空/真高带的解 | 先查走廊是否太窄、分辨率与净空是否匹配 |
| `path_clearance_violated` | 找到路但真几何净空不够 | 调 `clearance_m` 或 `spacing_m` |
| `roi_no_path` | 折线出半宽且 NLP 在半宽内无解 | 放宽 `roi.width_ratio` 或细化 `spacing_m` |
| `FieldContractError` | 场覆盖不足 / 产物过期 | **部署事故**，重启重建场，别当 400 |

> **失败只局部重试**（改端点 / 换机 / 拆任务），不要回到全局重解。

---

## 9. 两库对接：用真 `duration_s` 复核，闭环

ta1 给的米数/时长只用于**排序**；pp1 的真值回来后必须复核三项：

```python
service = 30.0                    # mission.json 的 problem.default_service_time
cursor = mission.current_time
for tid in tour.task_ids:
    t = by_task[tid]
    cursor += leg.duration_s                    # ← 真时长，不是 ta1 的估算
    arrive = max(cursor, t.earliest or 0.0)
    late = max(0.0, arrive - t.latest) if t.latest else 0.0
    cursor = arrive + service                   # 地面服务时间（pp1 不含）
```

本例复核结果（**0 迟到**）：

```text
 UAV   task     到达(s)       最晚    迟到(s)
   0    102      47.8   1500.0      0.0
   0    103     200.0   1800.0      0.0
   0    101     290.0   1500.0      0.0
   2    106      36.8   1700.0      0.0
   2    105     129.7   1700.0      0.0
   2    104     214.8   1600.0      0.0

迟到任务数 = 0 / 6
```

**三种回执的处理完全不同**：

| 回执 | 含义 | 你该做什么 |
|---|---|---|
| `status=ok` 且 `trajectory` 非空 | 有可执行轨迹 | 用 `duration_s` 重判 ETA / 时间窗 / 续航，下发 |
| `status=ok` 但 `trajectory is None` | 几何可飞、时间剖面没解出来（NLP 降级） | 标 `DEGRADED`，只能给折线；**别当成已验证时间可行性** |
| `status=infeasible` | 这一段无解 | 按 `failure_code` 改端点 / 换机 → 局部重排（限 1–2 轮） |

**方向是单向的**：ta1 的答案 → 展开航段 → pp1 逐段落地 → 回执复核 → 违约才局部重排。
pp1 永远不反过来影响分配搜索。原子接口是**单航段**（不是整条 tour）：
缓存按 OD 命中、失败可局部重试、天然可并行（实际用 8 路进程池）。

---

## 10. 一键跑通 + 输出解读

```bash
cd /
python3 docs/examples/run_lasa_demo.py            # 全程（约 1 分钟）
python3 docs/examples/run_lasa_demo.py --quick    # 只跑 ta1（约 2 秒）
```

脚本做 6 件事，对应本文 §5–§9：

```text
① scene_io 装载世界 + 任务包（经纬度 → ENU，校验 frame / frame_hash）
② ta1.allocate()            —— 谁飞、什么顺序、回哪个巢（同巢起降）
③ 展开 tours → 航段          —— 只把空中航段交给 pp1，起降另算
④ pp1.create_context()      —— 建全域场（一次），保存产物供复用
⑤ 逐段 ctx.plan()           —— 真航迹 + 时长 + 净空
⑥ 用真 duration_s 复核时间窗与航程，写 results/
```

落盘到 `/results/lasa_chengguan/lasa_demo_001/run_001.json`：

```json
{
  "mission_id": "lasa_demo_001",
  "world_id": "lasa_chengguan",
  "frame_hash": "0af4e4fb592e",
  "dataset_digest": "e9cebbed678f",
  "ta1": {"n_executed": 6, "path_m": 6407.68, "home_return_ok": true, "...": "..."},
  "tours": [{"uav_id": 0, "task_ids": [102, 103, 101],
             "start_nest_id": 0, "end_nest_id": 0}, "..."],
  "segments": [{"seg": 1, "uav_id": 0, "from_id": 0, "to_id": 102,
                "status": "ok", "length_m": 822.22, "min_clearance_m": 17.01,
                "duration_s": 47.83, "has_trajectory": true}, "..."],
  "review": [{"uav_id": 0, "task_id": 102, "arrive_s": 47.8,
              "latest": 1500.0, "late_s": 0.0}, "..."]
}
```

`frame_hash` + `dataset_digest` + `params_hash` 三个指纹一起构成**缓存键**，
任何一个变了结果就不可比、缓存必须失效。

---

## 11. 换成你自己的城市（清单）

按 §4 的顺序做，每步做完跑一次 §5 的校验：

1. **定运营区**：裁一个 **5–12 km** 的紧凑 `region.geojson`，看 bbox 对角线（< 25 km）。
2. **建筑**：裁到运营区、高度统一为**米/正数/相对地面**、字段名 ASCII、碎片过滤、格式二选一。
3. **DEM**：转 EPSG:4326、转米、填 NoData、确认覆盖 bbox + pad + 全部建筑、`gdalinfo` 看两个标签。
4. **禁飞区**（可选）：选 `absolute` 或 `agl` 口径，**字段不混写**；高海拔城市必须写绝对高程。
5. **写 `world.json`**：`world_id` + frame（锚点 bbox 左下角、`lat_ref` 取 bbox 中心纬度）；
   `dataset_digest` **用 pp1 现算**，别手写。
6. **pp1 预检 + 小区域建场**：看 `field.shape` / `nbytes` 可接受再扩；产物落
   `artifacts/<world_id>/<label>/`。
7. **写任务包**：`mission.json` 填 `world_id` + `frame_hash`；点位写 `lon/lat` + `z_abs`
   （= 地面 + 期望真高）；`nest_id` 补齐；`uav` 坐标 = 所属巢坐标。
8. **ta1 冒烟**：`scene_io.load_mission` 过一遍，3–5 个任务、2 架机跑通，看 `home_return_ok`。
9. **存基线**：跑一次把指标写进 `baseline.json`，以后改数据先对基线。
10. **对接待验证**：抽样航段比 ta1 代理排序与 pp1 真值排序（Kendall τ / top-k）。

坐标转换常用命令：

```bash
ogr2ogr -t_srs EPSG:4326 -lco ENCODING=GBK out.shp in_utm.shp    # Shapefile，DBF 用 GBK
ogr2ogr -t_srs EPSG:4326 out.geojson in_utm.shp                  # GeoJSON，UTF-8
gdalwarp -t_srs EPSG:4326 -r bilinear in_dem.tif dem.tif         # DEM
ogrinfo -al -so out.shp && gdalinfo dem.tif                      # 复核
```

**改了任何数据 → 三件事一起做**：重算 `dataset_digest` 写回 `world.json`、
删掉 `artifacts/` 重建场、重算 `baseline.json`。

---

## 12. 常见报错速查

| 报错 / 现象 | 大概率原因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: task_allocation1` | `uv sync` 没带 `--all-packages`，或没 `pip install -e` | §2 |
| `没有 GeoTIFF 地理标签（33550/33922）` | DEM 转换时丢了标签 | `gdalwarp` 重出 |
| `缺少区域轮廓 / 建筑目录 / dem.tif` | 目录结构不对，或 `with_dem=true` 却没 DEM | §4 |
| `没有可用建筑` | `building/` 空、几何是点/线、高度全非法 | 核对文件/字段/几何 |
| `高度字段 'Height' 缺失或非正` | 字段名错、空值、文本、楼顶海拔 | §4.2 清洗 |
| `任务包只收 lon/lat` | 写了 ENU 的 `x/y` | `world.frame.to_lonlat(x, y)` 反算 |
| `缺少 z_abs（绝对高程…）` | 只给了 AGL 或相对高度 | `z_abs = 地面 + AGL` |
| `frame_hash 不匹配` | 任务包不是用这个世界的 frame 算的 | 重算经纬度，或换 `world_id` |
| `巢 X 归属 N 架机，超过 capacity` | 机队规模超过巢容量 | 加巢 / 加容量 / 减机 |
| `out_of_flight_band` | 高海拔城市把 `z_abs` 写成了 AGL（拉萨：传 40 而不是 3690） | §4.6 |
| `endpoint_in_nofly` | 禁飞区 `z_min/z_max` 写成 `0/1000`（在地面以下） | 写**绝对高程** |
| `FieldContractError: 窗口…超出全域场` | 场覆盖不足、`max_od`/pad/region 变过、产物过期 | 重建场 |
| `rrt_no_path` / `roi_no_path` | 走廊太窄、净空太大、分辨率不匹配 | 调 ROI/净空，**别只加迭代** |
| `trajectory is None` 但 `status=ok` | 没装 casadi，或 NLP 降级 | 装 `[nlp]`；降级看 `diagnostics.warnings` |
| ta1 解出"从 A 起飞、回 B" | 该机 `nest_id` 与坐标不一致 | 对齐 `nest_id` 与巢坐标 |
| ta1 全程满足但航程明显偏小 | 把经纬度当米传了 | 检查坐标量级 |
| 建场 OOM / 被 SIGKILL | 运营区太大或 `max_od_length_m` 太大 | 缩 bbox 或调小 `max_od`（内存 ∝ 体积） |

---

## 13. 下一步与文档地图

| 文档 | 用途 | 什么时候看 |
|---|---|---|
| **本篇（4）** | 从 0 跑通整条链路 | 先看这篇 |
| [`0数据准备与处理.md`](0数据准备与处理.md) | 数据组织、格式、自检的**权威口径** | 建数据时 |
| [`1怎么用-ta1算法.md`](1怎么用-ta1算法.md) | ta1 单独用：调用、同巢口径、距离注入、结果字段 | 调分配时 |
| [`2怎么用-pp1算法.md`](2怎么用-pp1算法.md) | pp1 单独用：配置、建场、请求、结果、失败码 | 调规划时 |
| [`3任务分配-路径规划对接.md`](3任务分配-路径规划对接.md) | 两库对接：契约、代理排序验收、失败回环 | 联调时 |
| [`../task_allocation1/README.md`](../task_allocation1/README.md) | ta1 算法三段、模块地图、已知限制 | 改 ta1 代码时 |
| [`../path_planning1/README.md`](../path_planning1/README.md) | pp1 算法主线、参数体系、实测基线 | 改 pp1 代码时 |
| [`../scene_io/README.md`](../scene_io/README.md) | 零依赖数据适配层 | 接自己的数据源时 |
| [`ta1_full_config.example.jsonc`](ta1_full_config.example.jsonc) | ta1 全量配置（带注释） | 调参时 |
| [`pp1_full_config.example.jsonc`](pp1_full_config.example.jsonc) | pp1 全量配置（带注释） | 调参时 |

**最后三句**：

1. **ta1 的难点是"单位与同巢口径"，pp1 的难点是"高程口径 + 数据覆盖 + 场规模"。**
   这两件事在调参之前必须先做对。
2. **对接本身不该有难点**——两边只共享一份数据契约（ENU 米 + `frame_hash` + `world.json`）。
3. **代理不是交付值**：ta1 的米数只用于排序，交付值永远是 pp1 的真航迹与 `duration_s`。

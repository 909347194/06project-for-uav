# 06project-for-uav

多无人机协同的**两个算法库**：`task_allocation1`（分配层）+ `path_planning1`（落地层），均为纯计算库，未来服务化。

## 结构

```
06project-for-uav\
├─ pyproject.toml        # uv virtual workspace（根不打包）+ 全局 pytest/ruff 配置
├─ .python-version       # 3.11（casadi Windows wheel 覆盖更稳）
├─ uv.lock               # 依赖锁定；改 pyproject 后 `uv lock`
├─ .gitignore            # 排除缓存 / artifacts / results / building 的原始 shp
├─ task_allocation1\     # 分配层：DMDE + 修复链 + 局部优化，同巢起降（终点恒=起飞巢）
├─ path_planning1\       # 落地层：RRT 多初值 + CasADi/IPOPT，目标 = 最短飞行时间
├─ scene_io\             # 数据适配层：零依赖，读世界/任务包 → 纯数据（不 import 两库）
├─ data\
│  ├─ worlds\guangzhou\  # 世界：region + building(5 个 shp 图层) + dem.tif + world.json（frame 真源）
│  ├─ missions\demo_001\ # 任务包：mission/nests/uavs/tasks（经纬度真源）+ baseline
│  └─ contracts\         # 航段请求夹具（pp1 契约漂移测试用）
├─ docs\                 # 六份文档：0 数据 → 1 ta1 → 2 pp1 → 3 对接 → 4 完整流程 Tutorial → 5 交付说明
│                        #   + 两库各一份全量配置示例（json / jsonc）+ examples/ 可执行配套
├─ artifacts\  (可选，自己建)  # 建场产物 <world_id>/<label>/（1.15 GB，跟世界走，不进版本库）
└─ results\    (可选，自己建)  # 你的输出（实验/回归），按 <world_id>/<mission_id>/ 分层
```

`experiments/` 与 `results/`、`artifacts/` 都不在本发行版：前者是参考项目的一次性入口脚本，
后两者是可再生的产物目录。跑航段的等价写法见 `path_planning1/README.md` §9 或 `docs/2怎么用-pp1算法.md` §2。

**两库各自独立**：互不 import，各自能单独安装、单独上线（只用分配就只装 ta1）。
`import task_allocation1 as ta1` / `import path_planning1 as pp1` 是各自唯一门面。
对接靠**数据契约**：同一套 ENU 米、同一个 `frame_hash`、同一份 `world.json` +
`data/contracts/plan_request_example.json` 的契约测试。ta1 自带可选 `terrain` 直接读世界地形；
`scene_io` 只是另一条可选的数据适配层，`dependencies = []`，不 import 这两个库。

## 文档分工（按读者）

| 文档 | 内容 | 给谁看 |
|---|---|---|
| `task_allocation1/README.md` | 同巢改造、算法三段、模块地图、验证实验、已知限制 | **开发者（ta1 维护）** |
| `path_planning1/README.md` | 设计分层、算法主线、参数归属、结果契约、实测基线、扩展点 | **开发者（pp1 维护）** |
| `docs/0数据准备与处理.md` | **数据只讲一份**：第一部分 ta1 数据、第二部分 pp1 数据、第三部分两库同时使用（目录 / 坐标 / 字段 / 流程 / 报错） | 数据工程 / 后端 |
| `docs/1怎么用-ta1算法.md` | ta1 单独用：配置、最小调用、实体、距离口径（`vertical_section` 默认）、同巢口径、结果字段 | 后端工程师 |
| `docs/2怎么用-pp1算法.md` | pp1 单独用：配置、建场、请求、结果字段、失败码、部署与自检 | 后端工程师 |
| `docs/3任务分配-路径规划对接.md` | **只讲对接**：两库关系、对接三步、请求/回执字段、失败回环、延迟预算、验收清单 | 上层 / 联调 |
| `docs/4完整流程Tutorial.md` | **完整使用**（拉萨市城关区）：安装、配置、校验、ta1 → 展开航段 → pp1 → 回执复核、一键脚本；数据格式回指 0 | 新人 / 联调 |
| `docs/5交付说明-给后端.md` | **接之前必读**：交付状态、6 条接入要点、**已知问题（K≤N-1 / DEM 栅格类型 / 合成数据）**、服务壳职责、联调验收清单 | 后端 / 联调 |
| `scene_io/README.md` | 零依赖数据适配层：world/mission 加载、frame 校验、懒加载桥（pp1 世界入参 / ta1 请求 / ta1 切面采样器）；ta1 的 `terrain` 子模块可替代切面采样器这一桥 | 数据工程 / 后端 |

算法口径、实测基线、已知偏差、整理记录只在两库各自的 README（改代码时看）。

## 两库对比

| | task_allocation1（ta1） | path_planning1（pp1） |
|---|---|---|
| 层次 | **分配层**：谁飞、什么顺序、回哪个巢 | **落地层**：这条航段到底怎么飞 |
| 目标 | 最小化 硬违约 + 超程 + 跳过 + 迟到 + `λ·航程`；完成是约束 | **最短飞行时间**（`Objective.TIME`），无风 |
| 方法 | DMDE（代价空间差分 + 温度退火反映射）→ 解码修复链 → 交付后局部优化 | RRT 多初值选优 → 真几何净空终检 → CasADi/IPOPT 时间域 NLP |
| 输入 | 内存中的 `tasks` / `uavs` / `nests` + `cost_config`（或注入 `distance_fn`） | 建筑（`*.shp` / `*.geojson`）+ 可选 DEM / 禁飞区 + OD |
| 输出 | `AllocateResult.tours`（任务序列 + 终点巢）、`metrics_dict()` | `RouteSegmentPlan`（`trajectory` 主交付 + `path_enu` 兜底 + 指标） |
| 速度 | 毫秒级排序代理（默认垂直切面，需采样器；欧氏零开销） | 单航段 5.7–8.2 s（有场，本机实测 7.3–7.5 s；`on_demand` 5.5–5.6 s）；建场 120–140 s 一次 |
| 状态 | **v0.2.0**：同巢起降 + 可配置距离尺子（垂直切面默认 / 欧氏），47 测试通过 | **v0.4.0**：几何层 + 时间域轨迹层已交付后端 |

调用链：`ta1.allocate(...)` 定稿 → 展开每条 tour（含"末任务 → 起飞巢"返航腿）→ 逐航段 `pp1.ctx.plan(...)` → 真 `duration_s` 回来复核时间窗与航程。详见 `docs/3任务分配-路径规划对接.md`。

## 已定参数（与用户确认，不再是待定项）

### 两库共同

| 项 | 结论 |
|---|---|
| 对外门面 | `import task_allocation1 as ta1` / `import path_planning1 as pp1`；各自顶层 `__all__` 是唯一交付面 |
| 计算边界 | 两库都不碰 HTTP / 缓存 / 进程池 / 落库；服务壳归上层 |
| 库间关系 | **各自独立**：互不 import，可单独安装上线。对接只走数据契约（ENU 米 + frame_hash + world.json），不引入共享包；`scene_io` 仅做数据翻译，且不 import 两库 |
| 坐标 | 经纬度是规范输入；局地 ENU 米制在适配器里统一 frame，不允许"两边都是 ENU"就直接传 x/y |
| 默认值 | 各库只在自己的 `contract/defaults.py` 定义一次，其余引用 |

### ta1

| 项 | 结论 |
|---|---|
| 同巢起降 | **终点恒=起飞巢**（`end_nest_id == start_nest_id == uav.nest_id`），无开关；异巢 MCF / 投影已删除 |
| 距离口径 | 两档：`"vertical_section"`（默认，垂直切面：`agl_min/agl_max`、`clearance_m`、`climb_rate`、`num_samples` 全可配，地面/建筑包络由采样器注入）或 `CostConfig(model="euclid3d")`（无地形数据显式）；也可直接注入 `distance_fn(pos_a,pos_b)->米`（**有向**，优先） |
| 搜索预算 | 种群 50 × 200 代；`ζ=3`、`λ=0.1`、迟到 `0.01/s`、硬违约系数 `10` |
| 可复现 | `SolverConfig.seed` 单通道 `np.random.default_rng`；同种子逐位一致 |
| 正确性 | 独立重算账目，不靠库自带校验自证（实验脚本不在本发行版，方法见 `task_allocation1/README.md` §5） |

### pp1

| 项 | 结论 |
|---|---|
| 目标 | 最短飞行时间（`Objective.TIME`）；能耗需风场，暂不实现（`energy_j` 恒 `null`） |
| 飞行高度带 | **10–120 m，真高（AGL）**：端点收绝对 z，真高由库内按 DEM 派生 |
| 地形 | DEM 当实体（地面以下一律占据；`base_z` 由 DEM 抬升） |
| 净空 | **5 m**（`defaults.DEFAULT_CLEARANCE_M`，唯一来源）；规划看 SDF、交付看真几何 |
| 禁飞区 | 静态已实现（`nofly.geojson`，输入可 absolute / AGL，库内只存绝对 z）；动态/时间窗由服务层冻结静态集合 |
| 栅格 | 4/4/2 m，沿走廊轴向（斜栅格）；建筑按半格对角线保守膨胀 |
| 全域场 | 启动阶段建一次 + 最坏窗口覆盖自检（`field_mode=startup`，默认）；`on_demand` 显式现算 |
| 场产物 | `ctx.save_field(目录)` → `create_context(cfg, field_artifact=目录)` 走 mmap；多进程共享，取场 0.03 s |
| 数据版本 | `ctx.dataset_digest`（建筑/DEM/禁飞区内容指纹）是缓存键第一段；事故抛 `pp1.FieldContractError` |
| 建场契约 | `pad_m` 由 `max_od_length_m`（默认 3000）派生并对齐格距（392 m）；域外/超长 OD 返回结果码 |

## 命令

```powershell
uv sync --all-packages     # 根是 virtual workspace；缺这个 flag 装不到成员依赖
uv run pytest              # 全量用例；3 skipped（1 个慢测 + 2 个需 guangzhou shp 夹具）
uv run ruff check .

# 数据适配层：任务包 → ta1（零依赖加载 + frame 校验）
uv run python -c "import scene_io,task_allocation1 as ta1; m=scene_io.load_mission('data/missions/demo_001',world=scene_io.load_world('data/worlds/guangzhou')); print(ta1.allocate(scene_io.to_ta1_request(m)).metrics_dict())"

# 默认口径（垂直切面）：ta1 自己读世界地形，不依赖 pp1
uv run python -c "import scene_io,task_allocation1 as ta1; from task_allocation1.terrain import load_world_terrain; m=scene_io.load_mission('data/missions/demo_001',world=scene_io.load_world('data/worlds/guangzhou')); g,o=load_world_terrain('data/worlds/guangzhou').samplers(); print(ta1.allocate(scene_io.to_ta1_request(m,ground_z=g,obstacle_top_z=o)).metrics_dict())"

# 单段航段（pp1）：可复制片段见 path_planning1/README.md §9 或 docs/2怎么用-pp1算法.md §2
```

Python 固定 3.11（`.python-version`）：casadi 的 Windows wheel 覆盖更稳。
广州样例数据在 `data/worlds/guangzhou/`：`world.json` / `region.geojson` / `dem.tif` 进版本库；
`building/` 的 5 个 shp 图层（约 26 MB）按 `.gitignore` 排除，本地自备（该规则已在 `.gitignore` 里）。
`artifacts/`、`results/` 是可再生产物，同样不进版本库。

## 状态

| 阶段 | 内容 | 状态 |
|---|---|---|
| M0 | uv workspace + 两库各自契约 + 布局守卫 + 契约夹具 | ✅ |
| M1a | pp1 几何层：建筑/DEM/SDF/走廊/RRT/真几何净空 | ✅ |
| M1b | pp1 时间域轨迹层：最短时间、推力上限、走廊硬约束、净空复核 | ✅ |
| M0.5 | 数据层：`worlds/` + `missions/` 三层结构、`world.json` 帧指纹、零依赖 `scene_io` 加载与校验 | ✅ |
| M1c | ta1 分配层：DMDE + 修复链 + polish，同巢起降纯计算封装 | ✅ |
| M1d | ta1 独立正确性实验（5 组，全部通过） | ✅ |
| M1 | pp1 能耗（需风场）；多初值选优已实现（默认 5） | ⬜ |
| M2 | 服务壳（HTTP / 缓存 / 进程池 / 超时）——归后端；pp1 预热入口已固化在 `core/planning_context.py::create_context` | ⬜ |

仍待实测：代理与真值的 **Kendall τ / top-k 排序一致性**（见 `docs/3任务分配-路径规划对接.md` §6 第 2 条）。ta1 现已自带两档尺子（垂直切面默认 / 欧氏，参数可配），与 pp1 真航迹的偏差异测见两库 README；排序一致性仍需按真实航段实测。

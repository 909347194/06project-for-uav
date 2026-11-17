# path_planning1

城市低空 3D 轨迹规划内核：**建筑 + DEM + 可选静态禁飞区 → RRT 多初值选优 → CasADi/IPOPT 连续优化**，目标 **最短飞行时间**，无风。
口径来自 Lin et al. 2021（裁剪版内核，不含风场预测与论文实验脚手架）。

纯计算库：不联网、无全局状态、无隐式缓存、不碰 HTTP / 落库 / 序列化。
**交付面**：后端只 `import path_planning1 as pp1`；顶层 `__all__`（26 个名字）是正式 API 清单。`core`
深路径与 `contract` 里的 `PlanAlgorithm / PlanRequest / PlanResult / WorldSnapshot` 是无实现者的内部遗留
形状：仍可从子包导入（现有测试引用），但已移出 `contract.__all__`，不要按它们写接入代码。
**能耗仍未实现**（需要风场）；静态禁飞区已实现，动态/时间窗禁飞由服务层冻结静态集合。

**在系统中的位置**：本库是落地层，另一个算法库 `task_allocation1`（ta1）是分配层。ta1 只决定"谁飞、什么顺序、回哪个巢（恒=起飞巢）"，用注入的距离函数做排序代理；本库在分配定稿后逐航段出真航迹与时长。两库各自带契约、不共享代码，对接口径见 `../docs/3任务分配-路径规划对接.md`。

**v0.2.0 交付增量**（默认行为与全部基线 hash 不变）：

* 构建期只读场产物：`ctx.save_field(目录)` 落盘，`create_context(cfg, field_artifact=目录)` 用 mmap
  只读映射启动（实测取场 0.03 s vs 现算 123.63 s，结果逐字段一致）；同机多进程共享物理内存；
* `ctx.dataset_digest`：建筑/DEM 内容指纹，缓存键第一段，不再让上层自己猜 `world_id`；
* `pp1.FieldContractError`：产物过期 / 契约漂移 / 窗口越界 = 部署事故（`RuntimeError`，不是 `ValueError`）；
* 产物四项校验（schema / 场身份签名 / 数据版本 / 建场契约）与一条真数据端到端慢回归。

**v0.4.0 增量**（默认行为、392 m pad 契约与已有 `params_hash` / `dataset_digest` 基线均不变）：

* 建筑数据支持两种入口：`building/*.shp`（DBF 为 GBK）与 `building/*.geojson`（UTF-8，
  高度取 `properties.Height`），共用 Polygon / 最大块 / 洞填实 / 有限正高度 / 面积阈值口径；
* 目录里一栋可用建筑都没有时 `load_city` 直接 `ValueError`，不再静默建出"楼不挡路"的空世界；
* `Inf` / `NaN` 高度启动期直接拒绝（防止膨胀整片栅格）；
* 禁飞区新增 `world.nofly_altitude_mode`：`absolute` 读 `z_min_m/z_max_m`（默认），`agl` 读
  `agl_min_m/agl_max_m` 并在 `load_nofly` 入口用 DEM 换算；`NoFlyZone` 内部仍只存绝对 z。

**v0.3.0 增量**（默认行为、392 m pad 契约与已有 `params_hash` 基线均不变）：

* 静态禁飞区：数据目录可选 `nofly.geojson`（EPSG:4326；v0.3 仅支持绝对高程，v0.4 起支持 AGL 输入），
  与建筑共用 occupancy/SDF 与真几何终检；端点命中返回 `endpoint_in_nofly`；
* 禁飞区文件内容进 `ctx.dataset_digest`：禁飞区一改，旧场产物启动即判过期；
* `world.nofly_path` 可显式指定文件（默认 `nofly.geojson`，文件可没有；显式配置不存在则报错）。

后端接入清单见 `docs/2怎么用-pp1算法.md` §4–§6。

## 0. 一页速览

| 项 | 值 |
|---|---|
| 输入 | 建筑（footprint + 高度，**必填**）、DEM（**可选**，`None` = 平地 z=0）、禁飞区（**可选**，`nofly.geojson`，输入高度可 absolute / AGL）、起终点 ENU 米 + 参数 |
| 端点 | `xy` + **绝对 z**（与 `x/y` 同一高程基准）；真高由库内按 DEM 派生，只用于飞行带校验 |
| 输出 | `Path`（RRT 折线，兜底）+ `Trajectory`（60 节点 `x/v/a` + 均匀 `Δt`）+ 指标/诊断 |
| 目标 | `Objective.TIME`（`ENERGY` / `LENGTH` 是契约占位、无实现者） |
| 口径 | 飞行带真高 10–120 m（端点 z 派生）、净空 5 m、栅格 4/4/2 m（走廊轴向斜栅格）、推力 100 N / 5 kg |
| 一次航段（建好全域场） | 2.8 km **7.3–8.2 s**（轨迹 NLP 4.7–5.0 + RRT 1.9–2.0 + 取样格 0.5–0.6 + 净空 0.1；本机两次 7.34 / 7.54 s，默认 5 初值） |
| 一次航段（`on_demand` 窗口现算） | 2.8 km **5.5–5.6 s**（每次重建窗口 occupancy/SDF ≈3.0 s，轨迹 NLP 只 1.2 s） |
| 全域场 | `1409×1396×117` = **2.30 亿格** / 常驻 **1.15 GB** / 建场 **120–140 s**（实测）/ 峰值约 4.1 GB（推算）；建一次并做覆盖自检，之后每条 OD 只取样 |
| 场产物 | `ctx.save_field()` 落 `sdf.npy` + `occupancy.npy` + `meta.json`（约 1.15 GB）；`field_artifact=` 启动时 mmap 只读映射，取场 **0.03 s**，多进程共享同一份物理内存 |
| 对外门面 | `import path_planning1 as pp1`（v0.4.0）：`create_context` / `ctx.plan` / `ctx.save_field` / `RouteSegmentPlan` / `FieldContractError` / 配置解析 / 结果类型全在顶层 |
| 边界 | 只算法；服务壳（HTTP / 缓存 / 进程池）归后端；**建场契约**：`pad_m` 由 `max_od_length_m`（默认 3000）派生并对齐格距（392 m），域外/超长 OD 由装配层返回结果码，核心规划不回退 |

## 1. 产出：route / path / trajectory

| 词 | 是什么 | 本包字段 |
|---|---|---|
| **route** | 走廊：沿它无遮挡的空间域 | `CorridorWindow`；`half_width` **既是计算域，也是交付轨迹的硬约束** |
| **path** | 空间曲线（无时间维） | `res.path_enu`：RRT 折线，可飞但**折角处不能按原速跟随** |
| **trajectory** | path + 时间维 | `res.trajectory`：60 节点 `x/v/a` + 均匀 `Δt`，切线连续 C¹ |

**要飞的是 `trajectory`**，`path_enu` 只作几何兜底与对照。实测 0-1 航段（现算口径）：轨迹弧长 2846.1 m（折线 2859.6 m），巡航段（≥10 m/s）航向逐节点最大变化 **5.7°**（折线远大于此）；两端 `v=0`，端点那一步航向无定义。

## 2. 分层与依赖

```
contract/   纯数据 + 协议（不 import core）      defaults / enums / 几何 / 世界 / 请求 / 结果
core/       算法（只依赖 contract）            数据 → 全域场 → 走廊 → 取样 → RRT → 净空闸 → 时间域 NLP
```

- 依赖方向单向：`contract ← core`，禁止反向；服务壳不得长在 core 里。
- 演示/出图脚本（参考项目里的 `experiments/`）不属于本发行版，也不属于库的对外 API；要跑就按 §9 的片段自己写一个入口。
- 单文件上限由 `tests/test_pp1_layout.py` 强制：`contract/ ≤120` 行、`core/ ≤300` 行（现最大 `city_loader.py` 300，`trajectory_optimization.py` 293，`params_config.py` 280）。
- 默认数值**只有一处**：`contract/defaults.py`（41 个常量）。其它模块只能引用；`tests/test_pp1_defaults.py` 逐项比对契约 dataclass 的默认值，写第二份会被抓。
- 契约不承接 I/O：读文件的代码全在 `core/city_loader.py` / `core/city_dataset.py`。

## 3. 算法主线

### 3.1 几何层（RRT + 真几何闸）

```
起终点 + 真高带 → ROI 走廊(斜栅格窗口) → occupancy/SDF（全域场取样，或窗口现算）→ RRT 3D ×N（seed+i，每条内部捷径化）
              → 真几何净空预筛/选优 → 失败则报 failure_code，绝不静默放行
```

- **全域场（服务化默认）**：启动时按轴对齐全域格一次算完 occupancy + SDF（广州 1409×1396×117 = 2.30 亿格，4/4/2，120–140 s，常驻 1.15 GB），并立即做覆盖自检；之后每条 OD 只把走廊窗口的节点从场里取出来（0.67 s），不再栅格化建筑、不再 EDT。**没有运行期回退**：pad 由 `required_pad_m()` 从 `max_od_length_m` 派生并对齐格距，覆盖不够在启动阶段报错。
- **栅格化保守膨胀**：footprint 外扩半个格对角线、竖向各扩半格再判格心。否则比格距窄的楼会整栋漏检，SDF 还会比真几何乐观。
- **地形当实体**：地面以下视为占据（否则路径会钻地）；`Building.base_z_m` 取 footprint 内 DEM 最大值（宁可垫高，不切半栋楼）。
- **多初值选优**：`rrt_initializations=5`（范围 1–20），按 `seed+i` 跑 N 次 RRT；候选先按真几何净空硬过滤，再按“走廊内 > 满足 NLP 净空阈值 > 几何更短 > 净空更大 > 点更少”选一条给 NLP。只选几何初值，不跑 N 次 NLP；候选数/可行数进 `diagnostics.candidates/feasible_candidates`。
- **采样在索引空间**：窗口是斜的（沿走廊轴向），在 ENU 外接矩形里均匀采会有一半点落在带外。
- **净空两套口径**：规划期用 SDF（快）；**交付数字一律真几何**（`core/clearance_check.py`，独立于 SDF，另有第二套实现 `tests/reference_clearance.py` 对账，三条 OD 差 0.000 m）。

### 3.2 轨迹层（时间域 NLP）

决策变量 `x_i, v_i, a_i`（`n_resample=60`）+ 单个标量 `Δt`；目标 `min Δt`。约束：

- 运动学：`‖v‖≤v_max`、`‖a‖≤a_max`，梯形积分（速度/位置耦合）；
- 推力：`‖a+g·ê_z‖≤T_max/m`（力平衡），且 `a_z≥-g`（推力只朝上）——否则 `mass_kg/thrust_max_n` 是死参数；
- 净空：每个节点 + 每段内插 `segment_samples` 个点，`SDF ≥ clearance_m + sdf_tolerance_m(=竖直格距)`；
- 走廊：`|n·(x−start)|≤half_width`（交付物口径，ENU 线性式，不绑栅格索引）；
- 窗口索引边界、两端 `v=0`。

为什么不用"沿固定路径做前进式速度剖面"：那条路要求路径先被 Dubins 平滑成 C¹，否则每个 90° 折角处速度必须瞬间归零。这里让 `x` 也在走廊里自由，把"平滑"与"时间最优"合成一个 NLP，省掉 Dubins，也不再有"平滑后是否还避障"的第二套口径。

### 3.3 三级退让（只换初值/离散/预算，不换模型）

1. **走廊冷启动失败 → 无走廊版当种子**：带半宽约束冷启动时 IPOPT 可能退化（约束违反停在 1e-3 量级，预算加到 800 次也不动），而无走廊版 72 次就收敛。
2. **节点数 60→40→30**：窄走廊 60 节点 200 次不收敛，40 节点 41 次收敛，时长只差 0.2%。
3. **迭代预算 ×3**：只在 `Maximum_Iterations_Exceeded` 时走；"真的不可行"直接跳过，不白烧。

每次退让都写进 `message → diagnostics.warnings`，不做对调用方不可见的静默降级。NLP 失败**不**判整条航段不可行：`status` 仍 `ok`、`trajectory=None`，唯一例外是"折线出了半宽且解不出轨迹" → `roi_no_path`。

### 3.4 全域场：一次预计算，逐 OD 取样

服务化默认路径（`core/planning_context.py` 固化启动装配，`core/world_field.py` 只做建场/取样）：

```
启动一次：create_context(cfg)  →  load_city + build_world_field → 轴对齐 WorldField
                                 occupancy (nx,ny,nz) + SDF (nz,nx,ny)
                                 pad = required_pad_m(roi, spacing, max_od_length_m) = 392 m
                                 120–140 s / 常驻 1.15 GB
                              →  check_contract_coverage()：城市四角 × 16 方向 × L_max 的最坏窗口
                                 不够就在这里报错，不留到请求期
每次请求：ctx.plan(OD)  →  准入（端点出界 / OD 超长 → 结果码）
                        →  plan_route_segment(..., world_field=ctx.field)
                           建走廊窗口（不变）→ window_sdf 逐层取样（0.67 s）→ RRT → 净空闸 → NLP
```

`field_mode=startup`（默认）时 `create_context` 一定取得场并自检：不给 `field_artifact` 就现场建，
给了就读构建期产物（mmap 只读，取场 0.03 s）；`on_demand` 时 `ctx.field is None`，
`ctx.plan()` 走显式窗口现算（不做 OD 长度准入）。在线路径只有 `ctx.plan()` 一个入口，调用方无法漏传场。

- **只取样、不重算**：窗口内 6.9 M 个节点的 SDF 全部来自全域场；建筑栅格化与两遍 EDT 只在建场时发生一次。
- **竖直包络**：全域只能有一份，取 `[全城最低地面+10, max(最高地面+120, 最高楼顶)+1 格]`，所以层数 117（窗口的 56 层是它的子集）。窗口仍按窗口内最低地面定自己的包络——这是与现算唯一保留的语义。
- **取样补偿**：全局轴对齐格 → 斜窗口节点多一层三线性插值，距离场在凸角附近会偏高；取样后统一扣 `WORLD_FIELD_SAMPLE_BIAS_M = 1.5 m`，实测把交付净空拉回现算量级。
- **越界不回退**：startup 下规划器不判断场够不够、不重建 SDF；`FieldContractError`（窗口越界子类 `WindowOutsideField`）只应在建场配置错时出现，直接冒泡成部署事故。以前「同一个缓存键下两种 SDF 基准」的问题已由两件事消除：启动自检 + `params_hash(extra=field_signature(field))`。
- **场身份进指纹**：`field_signature()` 把场的 origin / spacing / shape **与取样补偿量 `WORLD_FIELD_SAMPLE_BIAS_M`** 写进 `params_hash`；补偿量只在场路径生效，改了会换一套 SDF 基准却不改格网，必须跟着场身份进指纹。场版 `d4ecf1210e78`，现算版 `fb7b2c24ca34`。on_demand 传空字典，不带场额外键。该字段也在 `metrics_dict()` 里，后端只序列化扁平字典也不会丢口径。
- **occupancy 不上在线路径**：SDF 符号已含占据信息，`WorldField.occupancy` 留作离线检查/可视化，避免在线再算一遍。
- **产物是带身份的快照**：`core/field_artifact.py` 加载时校验 schema / 场身份签名 / `dataset_digest` /
  建场契约（`max_od` / `pad` / `spacing` / 飞行带 / 城市边界），任一漂移就在启动期抛 `FieldContractError`；
  `meta.json` 最后写，中断不会留下半新半旧的有效产物。构建与产物启动两条路径的 `params_hash` 完全一致。
- **数据版本不进 `params_hash`**：`ctx.dataset_digest` 是独立的一段缓存键（内容哈希，不含绝对路径/mtime），
  这样数据版本不会改写参数指纹，产物还能跨机器分发；基线 `d4ecf1210e78` / `fb7b2c24ca34` 只随参数与场身份变化。

## 4. 参数体系

### 4.1 四个归属，不要串门

| 层 | 类型 | 管什么 |
|---|---|---|
| 数值 | `contract/defaults.py` | 所有默认值的唯一来源（41 个常量，含全域场取样补偿 1.5 m、建场时机与 `max_od_length_m`） |
| 请求口径 | `core/route_segment_params.py::RouteSegmentParams` | 走廊、栅格间距、真高带、seed（引用 defaults） |
| 算法 | `contract/planner_params.py::PlannerParams` | RRT 步长/迭代/采样率 + NLP 节点数/预算/步长（14 项） |
| 车型 | `contract/vehicle_limits.py::VehicleLimits` | `v_max / a_max / T_max / mass / clearance` |

上游只需一份 JSON：`core/params_config.py`（`world` / `request` / `params` 三段，缺省即默认、未知键报错、量值有界）。字段语义与"怎么定"属于接入方的事，见 `../docs/2怎么用-pp1算法.md` §5–§6。

### 4.2 三个容易踩的点

- **`spacing_m` 是唯一算力旋钮，且一次改三样**：耗时/内存 ∝ 1/spacing³、膨胀量（半格对角线）、SDF 容差（= 竖直格距）同时变。缓存的 SDF 与基线数字全部失效。
- **`clearance_m` 不等于几处阈值**：NLP/RRT 阈值 = `clearance_m + sdf_tolerance_m`；走全域场时 SDF 还要再扣 `WORLD_FIELD_SAMPLE_BIAS_M=1.5`（补一层插值）；交付净空 = 真几何；`CLEARANCE_SEARCH_RADIUS_M=30` 只是邻域查询上限（报 30 = "至少这么远"）。
- **`half_width` 与 `margin` 不是一回事**：`half_width` 约束交付轨迹；`margin` 只垫栅格与端部，**不放宽**交付边界。

## 5. 结果契约

`plan_route_segment(...) -> RouteSegmentPlan`：`status` / `failure_code` / `message` / `path_enu` / `trajectory` / `metrics`（`metrics_dict()` 32 键，含 `params_hash`）/ `stats` / `diagnostics`（含 `params_hash`、`warnings`）/ `timings`。

端点只有一种高度口径：**绝对 z**。核心入口 `plan_route_segment(buildings, frame, dem, start_xy, goal_xy, start_z_m, ..., goal_z_m=..., world_field=...)`，库内在入口处用 DEM 求两端地面（`dem=None` 时按 0），得到真高去判飞行带；`core/route_segment_xyz.plan_route_segment_xyz(...)` 只是 `(x,y,z)` 三元组的薄封装（校验后原样传 z），不在 `pp1.__all__`、也不在 `ctx` 上。`world_field=None` 或窗口越界时按窗口现算。端点 z 低于地面 → 真高为负 → `out_of_flight_band`，不会静默抬高。
失败是结果不是异常：`status` 实际只出现 `ok / infeasible`（`Status` 枚举另有 `timeout` / `invalid_input` / `internal_error` 三个预留值，本包不返回），细节看 9 个会真的返回的 `failure_code`（`none` / `outside_world` / `od_too_long` / `out_of_flight_band` / `start_or_goal_blocked` / `endpoint_in_nofly` / `rrt_no_path` / `roi_no_path` / `path_clearance_violated`）。
字段逐项、失败映射、HTTP 建议、批量与并发：**`../docs/2怎么用-pp1算法.md` §4–§6**（本文不重复）。

`params_hash`（`core/params_fingerprint.py`）是缓存键的唯一来源：`planner + vehicle + roi + spacing + agl 带 + seed + 净空查询半径` → 12 位 sha1；两端绝对 z 与 `world_id` 属请求/世界，需额外拼进键。

## 6. 模块地图

```
src/path_planning1/
├─ contract/        对外边界（纯数据，11 个文件，每个 ≤120 行）
│   ├─ __init__.py
│   ├─ defaults.py        全部默认数值（41 常量）
│   ├─ enums.py / local_frame.py / geometry_types.py / world_spec.py
│   ├─ vehicle_limits.py / planner_params.py / plan_request.py
│   ├─ plan_result.py     Trajectory / Metrics / Diagnostics / PlanResult
│   └─ planner_protocol.py
└─ core/            算法（27 个文件 / 26 个模块，每个 ≤300 行）
    ├─ city_dataset.py           数据加载唯一入口 load_city（frame → buildings → DEM → base_z → nofly）
    ├─ city_loader.py            shp(DBF GBK)/geojson(UTF-8) → ENU + 区域 bbox + nofly.geojson
    ├─ nofly_altitude.py         nofly 输入口径（absolute / AGL）与 DEM 地面范围换算
    ├─ dem_terrain.py            DEM 读标签 + 地面插值 + 逐栋接 base_z
    ├─ corridor_window.py        ROI → 走廊轴向栅格；`required_pad_m` 建场外扩唯一出处
    ├─ occupancy_sdf.py          occupancy（建筑+禁飞区，保守膨胀 + 地形实体）/ SDF（两遍 EDT）
    ├─ world_field.py            **全域场**：建场/取样 + 场身份签名 + `check_contract_coverage` 启动自检
    ├─ dataset_identity.py       数据集内容指纹（缓存键的 world_id 段，不进 params_hash）
    ├─ field_artifact.py         **场产物**：npy mmap + meta，四项契约校验 + FieldContractError
    ├─ field_contract.py         建场契约派生：pad 与 build_world_field 入参的唯一出处
    ├─ planning_context.py       **启动装配**：create_context 一次取场 + 自检；plan 在线固定取场并做准入
    ├─ clearance_check.py        BuildingPrisms（建筑+禁飞区）：棱柱真几何净空（交付数字 + 终检闸）
    ├─ rrt_path_search.py        RRT 3D（SDF 自适应步长、米制）+ 确定性捷径 + 多初值
    ├─ rrt_candidates.py         多初值候选：真几何评估 + NLP 初值选优
    ├─ trajectory_sampling.py    轨迹离散：弧长重采样 + v/a/Δt 初值
    ├─ sdf_casadi.py             SdfField → CasADi 三线性插值（求解器的转接头）
    ├─ trajectory_optimization.py 时间域 NLP（x/v/a/Δt；推力/半宽/净空；三级退让）
    ├─ route_segment_trajectory.py 时间剖面装配：净空复测 + 抬阈值重解 + 走廊失败判定
    ├─ route_segment_planner.py  **上层入口** plan_route_segment（一条流水线）
    ├─ route_segment_window.py   单航段窗口装配：走廊 → 窗口 + 地形 + z 包络
    ├─ route_segment_xyz.py      (x,y,z) 三元组薄封装（校验后原样传 z）
    ├─ route_segment_params.py   请求口径 RouteSegmentParams（含 sdf_tolerance / rrt_config）
    ├─ route_segment_outcome.py  结果形状 RouteSegmentStats / RouteSegmentPlan / rejected_plan
    ├─ params_config.py          **配置入口**：一份 JSON 配齐（缺省即默认；`field_mode` / `max_od_length_m` 单列）
    ├─ params_fingerprint.py     params_hash / digest_for（缓存键唯一来源）
    └─ request_check.py          入参自检（NaN/零间距 → ValueError）+ 机型可行性（0 秒拒绝）
```

`tests/`：布局守卫、默认值守卫、配置、建场契约（`test_pp1_field_contract.py`）、启动上下文（`test_pp1_context.py`）、约束（推力/走廊）、净空独立复核、绑定、core 冒烟。共 205 项，含 `test_pp1_public_api.py` 顶层门面守卫、`test_pp1_coord_guard.py` 坐标格式自检、`test_pp1_nofly.py` 禁飞区、
`test_pp1_dataset_identity.py` / `test_pp1_field_artifact.py` 与默认跳过的 `test_pp1_slow_integration.py`。

## 7. 实测基线

广州 20,655 栋 + DEM 157×172（2–75 m）；alt 25、`ratio=0.2`、净空 5 m、seed 0、4/4/2、`rrt_initializations=5`。`params_hash` 按口径区分：场版 `d4ecf1210e78`、现算版 `fb7b2c24ca34`。

**两套口径不可逐位互比**：全域场是轴对齐格、窗口是斜格，SDF 离散化不同，路径与迭代数会变；两者都过"真几何 ≥ 5 m"同一道闸。服务化默认用全域场；窗口现算**只在显式 `on_demand` 时启用**，不再作为越界兜底。pad 从 100 m 增到 392 m 是格距整数倍，内部格点未平移，所以下面历史场版数字仍然成立。

### 7.1 全域场（默认，启动阶段建场）

| OD | 水平距离 | 长度（折线） | 净空（轨迹·真几何） | 时长 | 推力（占比） | 横偏 轨迹/折线 | NLP 迭代 | 航段耗时 |
|---|---|---|---|---|---|---|---|---|
| 0-1 | 2833.5 m | 2862.9 m | 8.81 m | **149.0 s** | 52.6 N（0.53） | 17.6 / 44.0 m | 118 | 7.7–8.2 s |
| 0-2 | 2584.2 m | 2590.8 m | 6.61 m | **136.3 s** | 51.9 N（0.52） | 72.7 / 74.6 m | 68 | 3.4–3.5 s |
| 1-2 | 2433.3 m | 2452.5 m | 6.49 m | **128.8 s** | 56.7 N（0.57） | 9.8 / 60.0 m | 25 | 3.3 s |

三条都 `inside_corridor=True`。默认 5 初值后解仍确定：同一条 OD 重跑指标逐位一致（0-1 `2862.9 / 8.81 / 149.015`，耗时 7.7–8.2 s；0-2 `2590.8 / 6.61 / 136.272`，1-2 `2452.5 / 6.49 / 128.836`，各 3.3–3.5 s）。取样耗时 0.55–1.08 s。

另两档：

| 场景 | 结果 | 与现算的差别 |
|---|---|---|
| 窄走廊 `--ratio 0.1 --margin 30`（半宽 141.7 m，hash `086877052e18`） | 149.3 s / 2891.2 m / 净空 8.54 m / 22 迭代 / 1.5–1.6 s | 现算要"首解 4.78 m → 阈值 +2 m 重解"；场版首解就够，无告警 |
| 异高 `--alt-goal 60`（hash 同默认） | 149.2 s / 2859.9 m / 净空 8.33 m / 36 迭代 / 1.9–2.0 s | 现算 148.8 s / 7.15 m / 29 迭代 / 6.9 s（含无走廊热启动）；场版无 NLP 降级告警 |

### 7.2 窗口现算（`--no-field` / `field_mode=on_demand`，显式模式）

| OD | 长度（折线） | 净空 | 时长 | 推力 | 横偏 轨迹/折线 | NLP 迭代 | 航段耗时 |
|---|---|---|---|---|---|---|---|
| 0-1 | 2859.6 m | 6.96 m | **149.2 s** | 52.9 N（0.53） | 90.7 / 106.1 m | 44 | 5.7 s |
| 0-2 | 2589.0 m | 6.47 m | **136.1 s** | 52.1 N（0.52） | 13.5 / 8.7 m | 17 | 4.6 s |
| 1-2 | 2453.2 m | 5.80 m | **129.4 s** | 56.7 N（0.57） | 8.8 / 63.3 m | 6 | 7.5 s |

半宽 283.4 / 258.4 / 243.3 m，三条都 `inside_corridor=True`（1-2 轨迹 8.8 m 比折线 63.3 m 更靠内）。现算解确定：0-1、0-2 各重跑两次，长度/净空/时长一致；1-2 未重跑。真几何净空的第二套实现对账见测试，不在这里逐条声称 0.000 m。

窄走廊与异高两档仍是单初值基线（未随默认 5 重测）：窄走廊 149.3 s / 2918.9 m / 净空 8.39 m / 33 迭代 / 3.9–4.1 s；异高 148.8 s / 净空 7.15 m / 29 迭代 / 6.9–7.4 s。v15（`v_max_mps=15`、`rrt_initializations=5`）场版：194.831 s / 6.92 m / 4.18 s / hash `44095e1a2c8e`。本轮默认 5 初值实跑：startup 0-1 `2862.9 m / 8.81 m / 149.015 s / hash d4ecf1210e78`；on_demand 0-1 `2859.6 m / 6.96 m / 149.235 s / hash fb7b2c24ca34`。

容量：全域场建好后 2.8 km ≈ 7.7–8.2 s / 航段（默认 5 初值；`v_max_mps=15` ≈4.2 s），内存恒定 1.15 GB（与 OD 长度无关）；不建场时 4.3–7.6 s，5.6 km 对角 ≈ 17 s / 1.5 GB（但 startup 下该 OD 会被 `od_too_long` 拒，除非放大 `max_od_length_m`）。`load_city` ≈ 2.5 s（92 MB，进程内复用）。6/6/3 粗栅格 ≈ 1.3–1.5 s，但净空虚高 4–7 m，只能当"有没有解"的粗筛。

## 8. 已知限制

1. 净空 5 m 是**纸面值**：保守膨胀后实得 6.2–10.4 m（要贴近就调细 `spacing_m`）。
2. 轨迹三种简化：`Δt` 均匀、两端 `v=0`、节点数固定（5.6 km 时转弯粒度 ≈96 m）；窄走廊/慢机会降节点或放大预算，`warnings` 写明。
3. **SDF 乐观误差实测可达 2.2 m > 容差 2 m**（屋顶附近三线性 + 距离场非线性；场版再叠一层全局格插值，已固定扣 1.5 m 补偿）：交付前真几何复核，不达标就抬净空 +2 m **重解一次**，还不过才 `path_clearance_violated`。
4. 走廊半宽约束交付轨迹；`path_enu` 只是内部初值，可以出半宽（`path_max_lateral_m` 就是这个数）。
5. 推力约束**不含阻力**（20 m/s 时约 37.8 N，计入须重算预算）；5 kg 机型远未咬上限（52 N / 100 N），9.5 kg 重载会咬到 100 N。
6. 没有能耗目标（无风，`E=∫P dt` 口径不成立）；`energy_j` 恒 `None`。
7. DEM 30 m 分辨率 / `uint8`：楼间起伏被抹平；屋顶取 `Height` 字段而非 DSM。
8. 默认 5 个初值（`seed..seed+4`）；样本只有 3 条 2.4–2.8 km 航段 + 1 条 5.6 km 对角。
9. 净空上限 30 m（邻域查询半径）。
10. 多初值只选几何初值，不跑 N 次 NLP；默认 5。候选多不等于 NLP 一定收敛：NLP 失败仍按原降级/`roi_no_path` 口径。
11. **全域场是另一套离散化**：与窗口现算的基线不能逐位互比；取样再扣 1.5 m 补偿后，净空仍可能在个别 OD 上比现算低/高（1-2 场版 6.49 vs 现算 5.80，0-1 场版 8.81 vs 现算 6.96）。交付数字始终以真几何为准。
12. **场成本**：pad 100 → 392 m 后格数 1.847 亿 → 2.301 亿（1.246×）；建场 **120–140 s**（实测）、峰值约 **4.1 GB**（`build_sdf` 同时持有两份 float64 EDT，按数组尺寸推算，未用进程计量实测）、常驻 **1.15 GB**。这是「确定 SDF 口径 + 稳定延迟」的代价。多 worker 各建一份会乘以并发数：建议单实例建好后共享（Linux fork / Windows 用同一进程串行），或接受每进程 ≈120 s 启动 + 每份 1.15 GB。
13. **场会过期 / 契约会变**：建筑/DEM 任何改动都必须重建；`params_hash` 现在含场格网身份与取样补偿量（origin/spacing/shape/sample_bias_m）但不含数据版本，缓存键仍要自己拼 `world_id`。`max_od_length_m` 改了会同时改 pad 与 shape，hash 自动跟着变。**不再有越界回退**：覆盖不够在启动自检报错；端点出界或 OD 超契约由 `ctx.plan()` 返回 `outside_world` / `od_too_long`。直接调 `plan_route_segment` 没有这两条准入。
14. **契约上限是产品决策**：默认 `L_max=3000 m` / `ratio=0.2`。业务要更长 OD 或更大 ratio 就调 `world.max_od_length_m` 并接受更大的内存；`required_pad_m()` 会按最终 lateral 派生 pad，别手填 `pad_m`。配置超过城市对角线会自动收敛到对角线（端点都在边界内，更长 OD 不存在），防止 pad/内存无界放大；`ctx.max_od_length_m` 是实际生效值。

## 9. 测试与复现

```powershell
# 开发环境：仓库根，venv 在 .venv（用 uv 的话先 uv sync --all-packages）
& ".\.venv\Scripts\ruff.exe" check path_planning1
& ".\.venv\Scripts\python.exe" -m pytest path_planning1\tests --no-header -q
$env:PP1_SLOW_TESTS="1"; & ".\.venv\Scripts\python.exe" -m pytest path_planning1\tests\test_pp1_slow_integration.py -q   # 真数据端到端（约 2.5 min）
```

跑一条航段：本发行版不带 `experiments/` 脚本，用下面这段代替原来的 `plan_route_segment.py`：

```python
import json, pathlib, path_planning1 as pp1

ROOT = pathlib.Path.cwd()          # 在仓库根执行
cfg = json.loads((ROOT / "docs/pp1_full_config.example.json").read_text(encoding="utf-8"))
cfg["world"]["root"] = str(ROOT / "data/worlds/guangzhou")

ctx = pp1.create_context(cfg)                                   # startup：现场建全域场（120–140 s）
res = ctx.plan((1636.8, 4198.2), (2040.6, 1393.6), 37.50, goal_z_m=37.09)
print(res.params_hash, res.metrics_dict(), res.timings)

ctx.save_field(ROOT / "artifacts/guangzhou/field_maxod3000")    # 构建期：落只读产物（约 1.15 GB）
ctx = pp1.create_context(                                    # 服务期：mmap 取场 0.03 s
    cfg, field_artifact=ROOT / "artifacts/guangzhou/field_maxod3000")
```

- `world.field_mode`：`startup`（默认，建场 + 覆盖自检）/ `on_demand`（省掉 1.15 GB，单段慢些）。
- 放大契约：`cfg["world"]["max_od_length_m"] = 6000`（pad 自动重算，启动自检验证覆盖）。
- 全量默认模板：`print(pp1.default_config())`，与 `docs/pp1_full_config.example.json` 逐键一致。
- 换机型速度：改 `cfg["params"]["vehicle"]["v_max_mps"]`（15 → 单段 ≈4.2 s、时长 194.8 s、hash `44095e1a2c8e`）。
- 输出目录自己定（`results/`、`artifacts/` 都不进版本库，本发行版也没有这两个目录）。

## 10. 改哪里（扩展点）

| 想要 | 动哪里 | 注意 |
|---|---|---|
| 换数据/换城市 | `core/city_dataset.py`（`load_city`） | 加载顺序有语义，漏 `attach_base_z` 会静默错；建筑/DEM 一改，全域场必须重建 |
| 改栅格/走廊/真高带 | `RouteSegmentParams` + `defaults.py` | 三者互相牵连（见 §4.2），基线全要重测 |
| 改算法旋钮 | `PlannerParams` | 进 `params_hash`，等于换缓存域 |
| 换机型 | `VehicleLimits` | 先过 `check_vehicle`（悬停/垂直加速门槛） |
| 加约束（爬升角、动态障碍） | `trajectory_optimization._solve_once` / `world_spec` | 新增约束要一起进报告字段，否则"配了不生效" |
| 改多初值/选优准则 | `rrt_path_search.py` + `rrt_candidates.py` | 结果仍归一成 `RouteSegmentPlan`；选优只看几何，不跑 N 次 NLP |
| 能耗目标 | 先接风场 | 无风时能耗口径不成立，别先做模型 |
| 服务化（HTTP/缓存/进程池） | **不在本包** | 后端负责；本包只提供纯函数与 `params_hash` |

## 11. 文档分工

| 文档 | 读者 | 内容 |
|---|---|---|
| 本文 | 开发者 | 设计、算法、参数归属、基线、测试、扩展点 |
| `../docs/0数据准备与处理.md` | 数据工程/后端 | pp1 数据目录与字段格式、DEM/禁飞区口径、预检脚本；ta1 实体与 JSON 映射另占一半 |
| `../docs/2怎么用-pp1算法.md` | 后端工程师 | 装、配 JSON、建全域场、调、读结果、失败映射、容量与并发、自检 |
| `../docs/1怎么用-ta1算法.md` | 后端工程师 | 分配层的调用、同巢口径、结果字段、距离函数注入、独立验证 |
| `../docs/3任务分配-路径规划对接.md` | 对接/后端 | 分配层代理与落地层的分工、接口契约草案、失败回环、验收口径 |
| `../task_allocation1/README.md` | 开发者 | 分配算法、同巢改造、模块地图、验证实验 |

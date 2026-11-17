# task_allocation1

多机巢多无人机协同任务分配的**纯计算内核**：把论文分配算法封装成可 `import` 的库，并改造为**同巢起降**——每架机从自己的机巢起飞、做完任务返回同一个机巢，终点不再是自由决策变量。论文三段主线都保留：DMDE 进化搜索 → 解码修复链 → 交付后局部优化。

纯计算核心：不联网、不读文件、不落库、无全局随机状态。距离口径由请求级配置 `cost_config` 决定（`vertical_section` 垂直切面默认 / `euclid3d` 无地形自动），或直接注入 `distance_fn`；随机性由 `seed` 注入。`terrain` 子模块随包安装，直接读世界地形，不依赖 pp1。

**在系统中的位置**：本库是分配层，另一个算法库 `path_planning1` 是落地层。本库只回答"谁飞、按什么顺序飞、回哪个巢（恒=起飞巢）"，米数只是排序代理；真航迹与时长由 pp1 在分配定稿后逐航段给出。两库各自带契约、不共享代码，对接口径见 `../docs/3任务分配-路径规划对接.md`。

## 0. 一页速览

| 项 | 值 |
|---|---|
| 对外入口 | `import task_allocation1 as ta1` → `ta1.allocate(ta1.AllocateRequest(...))`；顶层 `__all__`（25 个名字）是正式 API |
| 核心改造 | **终点机巢恒 = 起飞巢**（`end_nest_id == start_nest_id == uav.nest_id`）；删掉异巢降落 / 最小费用流，返航代价照常计入航程与目标 |
| 输入 | 内存中的 `tasks` / `uavs` / `nests` + `cost_config`（或 `distance_fn`）+ 配置；没有任何文件 I/O |
| 距离尺子 | `CostConfig` 两档：`vertical_section`（默认，垂直切面；飞行带 `agl_min/agl_max`、净空 `clearance_m`、爬升率 `climb_rate`、采样 `num_samples` 全可配；有地形时注入采样器）/ `euclid3d`（三维直线，无地形自动） |
| 输出 | `AllocateResult`：`solution.tours`（每航次的任务序列+终点巢）、`executed_task_ids`、`metrics_dict()`、`home_return_ok` |
| 算法三段 | ① DMDE（代价空间差分 + 温度退火反映射）② 修复链（合并/航程修复/改派/覆盖补全/时间窗重排）③ 局部优化（跨机 relocate/swap + 组内 2-opt，四道门） |
| 口径 | 完成是约束、成本是目标；适应度 `f = W_hard·n_hard + w_exc·超程 + W_skip·n_skip + w_late·迟到 + λ·航程`（覆盖门禁，F1 只作事后指标） |
| 可复现 | 同一个 `SolverConfig.seed` 逐位可复现；随机性单通道 `np.random.default_rng(seed)` |
| 默认预算 | pop 50 × gen 200；`ζ=3`、`w_late=0.01/s`、`λ=0.1`、`κ=10`（改动进 `contract/defaults.py`） |

## 1. 最小用法

```python
import task_allocation1 as ta1

req = ta1.AllocateRequest(
    tasks=[ta1.Task(id=0, x=0, y=0, reward=1.0, latest=3600),
           ta1.Task(id=1, x=800, y=0, reward=1.0, latest=3600)],
    uavs=[ta1.UAV(id=0, x=0, y=0, nest_id=0, speed=15.0, remaining_range=1e5)],
    nests=[ta1.Nest(id=0, x=0, y=0, capacity=4)],
    distance_fn=ta1.euclidean3d,                       # 无地形可不传，默认自动欧氏；有地形给采样器（§1.1）
    solver_config=ta1.SolverConfig(pop_size=30, max_generations=100, seed=42),
)
res = ta1.allocate(req)

res.solution.executed_task_ids   # 本周期交付哪些任务
res.solution.tours               # 每架机的航次：task_ids + end_nest_id
res.home_return_ok                # True：每条非空航次都返回起飞巢
res.metrics_dict()                # 扁平指标，供上层直接序列化
```

### 1.1 距离尺子：两档，全可配

```python
from task_allocation1.terrain import load_world_terrain

ground_fn, envelope_fn = load_world_terrain("data/worlds/<id>").samplers()
cost = ta1.CostConfig(agl_min_m=10.0, agl_max_m=120.0,   # 默认即 vertical_section
                      clearance_m=5.0, climb_rate=0.15, num_samples=200)
req = ta1.AllocateRequest(..., cost_config=cost,
                          ground_z=ground_fn,        # (xs, ys) -> 各点地面高程（绝对 z）
                          obstacle_top_z=envelope_fn) # (xs, ys) -> 各点 max(地面, 屋顶顶面)
```

- `vertical_section`（默认）：沿 S-T 连线采样包络做地形跟随；带记忆化（有序键，方向相关），单周期内每条边只算一次。两个采样器都不给时自动走欧氏。
- `euclid3d`：零开销；无地形不传采样器即可，也可显式选。
- `over_band_mode`：`allow`（带外照爬，只报米数）/ `infeasible`（带内飞不过去的腿记 `inf`）。
- 仍可用 `distance_fn=` 完全接管（二者同时给时 `distance_fn` 优先）。有世界数据时用 `ta1.terrain.load_world_terrain(root).samplers()` 现造两个采样器（不依赖 pp1）；没有地形不传采样器，默认自动欧氏。`scene_io.to_ta1_cost_samplers(world)` 是走任务包时的另一条路。

实测（广州 `demo_001`，同 seed）：欧氏 `path_m=4501.9` → 切面 `4517.4`（+0.34%），航次与交付不变，0.28 s → 0.31 s（该任务包显式固定 `euclid3d` 以复现基线）。

整包配置示例：`../docs/ta1_full_config.example.json`（`solver` / `problem` / `cost` 三段，值等于代码默认；同目录 `.jsonc` 是带注释的阅读版），`tests/test_ta1_config_example.py` 守住示例与默认值不漂移。

## 2. 同巢起降如何落地

| 层 | 做法 |
|---|---|
| 解结构 | `UAVTour.from_uav()` 缺省把 `end_nest_id` 写为 `uav.nest_id`（= 起点巢快照） |
| 修复链 | 装配期 `_assemble()` 显式写终点 = 起飞巢；航程修复/改派/覆盖补全的代价核算全部用「末任务 → 起飞巢」返航腿 |
| 可行性 | `check_feasible()` 把「非空航次终点 ≠ 起飞巢」记为硬违约 `home_return_violation`；`solution.home_return_ok()` 为 False 直接判不可行。外部篡改终点逃不过校验 |
| 不变量测试 | `tests/test_ta1_same_nest.py` 对多个种子断言每条航次 `end_nest_id == start_nest_id`，并验证异巢篡改被拒 |

刻意不做（与参考项目的差别）：终点巢最小费用流 / 最近合法巢投影 / 终点巢反事实诊断——这些是「异巢降落」及其对照实验的产物；本库只有同巢一种口径。不提供关闭开关：`ConstraintConfig.home_return` 只读恒 `True`。

## 3. 分层与依赖

```
contract/   纯数据 + 协议（不 import core）   实体 / 解 / 配置 / 请求 / 结果 / distance 协议 / defaults
core/       算法（只依赖 contract）           cost → tour → problem → encoding/inverse/operators → repair → polish → solver
tests/      布局守卫 / 默认值守卫 / 公开 API / 代价 / 约束 / 同巢不变量 / 端到端
```

- 依赖单向 `contract ← core ← 门面`，禁止反向（`tests/test_ta1_layout.py` 守卫）。
- 单文件上限：contract ≤120 行、core ≤300 行（同上守卫）。
- `core/` 不得出现文件/网络 I/O 调用（同上守卫）。
- `contract/defaults.py` 是默认数值唯一来源，`test_ta1_defaults.py` 逐项比对。

## 4. 模块地图

```
src/task_allocation1/
├─ contract/
│   ├─ defaults.py        全部默认常量（唯一来源）
│   ├─ enums.py           状态 / 模型形态 / EndNestPolicy(仅 HOME)
│   ├─ distance.py        DistanceFn 协议 + euclidean3d（euclid3d 档口径）
│   ├─ entities.py        Task / UAV / Nest
│   ├─ solution.py        UAVTour / AllocationSolution（含 home_return_ok）
│   ├─ config.py          ConstraintConfig / ProblemConfig / SolverConfig
│   ├─ cost_config.py     CostConfig（距离尺子档位 + 飞行带/净空/爬升率）+ SurfaceSampler 协议
│   ├─ request.py         AllocateRequest
│   └─ result.py          AllocateResult / metrics_dict
└─ core/
    ├─ cost.py            节点键编码(任务 id / 巢 offset+id / 起点 -1-id) + 注入式 CostProvider
    ├─ vertical_section.py 垂直切面航程估算器 + 带记忆化的 build_distance_fn
    ├─ tour.py            航次代价 / 到达-迟到 / 同巢返航腿
    ├─ problem.py         问题实例 + 可行性 + 四层适应度 + 同巢硬约束
    ├─ encoding.py        Gene / Individual / 种群初始化（规则 3.1~3.3）
    ├─ repair_rules.py    最近邻匹配 / 唯一性掩码 / 无效修补 / 段规范化
    ├─ inverse_map.py     反映射 φ'（公式 3-7，温度退火 + 迭代行替换）
    ├─ operators.py       动态 CR(3-9) / 混合差分(3-10) / 动态 F(3-11) / GMR(3-12)
    ├─ dmde.py            DMDE 主循环（算法 3.1）
    ├─ participating.py   参与机选择（cheapest/first/marginal/demand）
    ├─ repair.py          解码修复链（合并/航程/改派/覆盖补全/重排/装配）
    ├─ coverage.py        覆盖门禁审计（voluntary_skip vs capacity_shortfall）
    ├─ polish.py          交付后局部优化（四道门，无随机）
    └─ solver.py          allocate() 顶层编排
```

## 5. 测试与复现

```powershell
# 仓库根 E:\06project-for-uav，venv 在 .venv
& ".\.venv\Scripts\python.exe" -m pytest task_allocation1\tests --no-header -q   # 47 passed
& ".\.venv\Scripts\ruff.exe" check task_allocation1                              # All checks passed
# 以可编辑模式安装后即可从任意目录 import：
uv pip install -e task_allocation1
# 独立正确性实验（不靠库自带校验自证）：16 随机实例 + 复现 + 注入 + 退化 + 收敛
# 脚本不在本发行版（目录里没有 experiments/）；按 §5 的方法自己写，
# 输出落你自己定的目录，如 results/checks/ta1_verify/{summary.json,report.md,route_map.png}
```

## 6. 边界 / 已知限制

1. **单周期内核**：不做跨周期滚动、任务流到达、休眠/唤醒、事件重规划——那些是有状态服务层的事。本库每次调用是纯函数式的一次分配。
2. **核心不接 DEM / 文件**：有地形时用 `ta1.terrain` 或注入采样器走 `vertical_section`；没地形自动走欧氏。
3. **同巢口径的代价**：机队归属与起飞巢绑定，不做跨周期重部署/轮转巢；这是同巢起降的语义取舍，不是缺陷。
4. 覆盖门禁是一阶判据（能开班/能插入），不是全局最大覆盖证明。
5. 纯计算，不含 HTTP / 缓存 / 进程池 / 超时；由上层服务自己包。

## 7. 来源

算法口径来自参考项目 `D:\SchoolAffairs\groupmeeting\thesis\dissertation\ExperimentForGraduation\task_allocation`（第 3 章 MUAS + DMDE + 修复链 + polish）；本库是其**同巢起降、纯计算化**的重新封装，不是逐文件拷贝。端到端同巢契约见 `tests/test_ta1_same_nest.py`，独立重算的正确性实验（脚本不在本发行版）方法见 `docs/1怎么用-ta1算法.md` §6。

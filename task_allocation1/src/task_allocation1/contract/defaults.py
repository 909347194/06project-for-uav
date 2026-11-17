"""**默认数值清单：本包所有默认常量只在这里出现一次。**

规矩：
1. 本文件只放数值与字符串常量，不放逻辑、不放类型定义。
2. 其他模块只能 ``from ..contract.defaults import X`` 引用，不许写第二份默认值
   （``tests/test_ta1_defaults.py`` 逐项比对）。改这里 = 改算法口径。
3. ``HOME_END_NEST_MODE`` 是本包**唯一口径**：终点机巢恒等于起飞机巢。
   本库不提供 nearest / mcf / projection 等异巢模式（对照实验请回参考项目）。
"""

from __future__ import annotations

# —— 节点键编码（三类空间互不冲突）---------------------------------------
# 任务键 = task.id（非负整数）；机巢键 = offset + nest.id；UAV 起点键 = -(1+uav.id)
DEFAULT_NEST_OFFSET = 1_000_000

# —— 终点机巢口径（本库固定为 home，不提供异巢对照模式）------------------
HOME_END_NEST_MODE = "home"

# —— 航程超限米数权重推导：None = 自动 1/m --------------------------------
DEFAULT_RANGE_EXCESS_WEIGHT_MODE = "auto"

# —— DMDE 搜索（论文公式 3-9/3-10/3-11/3-12）-----------------------------
DEFAULT_POP_SIZE = 50
DEFAULT_MAX_GENERATIONS = 200
DEFAULT_ZETA = 3                 # 动态交叉率曲率指数
DEFAULT_EXTINCTION_DELTA = None  # GMR 灭绝阈值；None = 原式 GMR = 1 - CR
DEFAULT_SEED = 0

# —— 反映射温度退火与扰动 -------------------------------------------------
DEFAULT_ASSIGN_PERTURB_FACTOR = 0.4
DEFAULT_TOUR_PERTURB_FACTOR = 0.5
DEFAULT_PERTURB_TEMP_GATE = 0.1

# —— 参与机选择 -----------------------------------------------------------
DEFAULT_PARTICIPATING_POLICY = "demand"  # cheapest / first / marginal / demand
DEFAULT_DEMAND_RANGE_FACTOR = 0.5

# —— 解码修复链 -----------------------------------------------------------
DEFAULT_REPAIR_MAX_PASSES = 6

# —— 适应度 ---------------------------------------------------------------
DEFAULT_LAMBDA_COST = 0.1
DEFAULT_W_PENALTY = 1.0
DEFAULT_W_VIOLATION = 0.01       # 迟到软计价 0.01/s
DEFAULT_HARD_VIOLATION_KAPPA = 10.0
DEFAULT_DEFAULT_SERVICE_TIME = 0.0
DEFAULT_CURRENT_TIME = 0.0

# —— 交付后局部优化（四道门）---------------------------------------------
DEFAULT_POLISH_ROUNDS = 30
DEFAULT_COST_POLISH_EPS = 1.0

# —— 距离代价口径（分配阶段的代理尺子，见 ``contract/cost_config.py``）------
DEFAULT_COST_MODEL = "vertical_section"  # vertical_section（默认）| euclid3d（无地形数据时显式选）
DEFAULT_COST_AGL_MIN_M = 10.0          # 飞行带下界（真高，相对当地地面）
DEFAULT_COST_AGL_MAX_M = 120.0         # 飞行带上界（真高）；over_band=infeasible 时生效
DEFAULT_COST_CLEARANCE_M = 5.0         # 高于障碍包络（屋顶）的最小竖直净空
DEFAULT_COST_CLIMB_RATE = 0.15         # 爬升/下降斜率上限（米 / 米水平）
DEFAULT_COST_NUM_SAMPLES = 200         # 垂直切面采样点数
DEFAULT_COST_OVER_BAND_MODE = "allow"  # allow | infeasible

# —— 数值下限 -------------------------------------------------------------
EPS = 1e-12

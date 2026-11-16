"""**配置清单：本包所有默认数值只在这里出现一次。**

规矩（照这个文件读，不要到处找）：

1. 本文件只放**数值**，不放逻辑、不放类型定义——类型在 ``params`` / ``vehicle`` / ``world``。
2. 其他任何模块（含 ``core``、``experiments``）**不许再写同名默认值**，只能 ``from ..contract
   import defaults`` / ``from .defaults import X`` 引用；``tests/test_pp1_defaults.py`` 会逐项
   比对"契约 dataclass 的默认值 == 这里的常量"，写第二份就会被抓住。
3. 改这里 = 改实验口径。凡改动都要重跑 OD，并把新数字写进 ``path_planning1/README.md`` 的「实测基线」。
"""

from __future__ import annotations

# —— 飞行带（真高 AGL，米）------------------------------------------------
DEFAULT_ALT_MIN_M = 10.0
DEFAULT_ALT_MAX_M = 120.0

# —— 走廊 ROI（走廊总宽 = ratio × 起终点水平距离，再向两侧各留 margin）------
DEFAULT_ROI_WIDTH_RATIO = 0.2
DEFAULT_ROI_MARGIN_M = 50.0

# —— 服务启动：全域场建场时机 ------------------------------------------------
# startup    = 进程启动阶段一次建好全域场，在线请求只按走廊取样（默认，落实"离线一次、在线只取"）
# on_demand  = 不预建，每条请求窗口现算（低内存/冷启动调试）
WORLD_FIELD_MODE_STARTUP = "startup"
WORLD_FIELD_MODE_ON_DEMAND = "on_demand"
DEFAULT_WORLD_FIELD_MODE = WORLD_FIELD_MODE_STARTUP

# —— 栅格与数据（间距单位米，轴序 ENU）--------------------------------------
DEFAULT_GRID_SPACING_M = (4.0, 4.0, 2.0)
# 全域场按走廊取样时会多出一层三线性插值（全局轴对齐格 → 斜窗口节点）。三线性对距离场
# 在凸角附近偏高，实测让交付净空掉 1–2 m；扣掉这个补偿量后与"窗口现算"基线基本重合。
WORLD_FIELD_SAMPLE_BIAS_M = 1.5
# 全域场水平外扩（**建场契约，不是常量**）：pad 由 core/corridor_window.required_pad_m()
# 按"契约允许的最大 OD 走廊"派生，并向上取整到水平格距的整数倍——否则格网原点平移会让
# 所有窗口的三线性取样位置改变、历史基线不可复现。下面两个只是派生的下界与额外余量。
WORLD_FIELD_PAD_MIN_M = 100.0
WORLD_FIELD_PAD_EXTRA_M = 30.0
# 建场契约支持的最大 OD 水平长度（米）：端点须在城市 bounds_m 内，OD 不得超过这个值。
# 由装配层（PlanningContext）在请求期校验；调大它会自动把 pad 一起放大，内存与建场时间随之上涨。
WORLD_FIELD_MAX_OD_LENGTH_M = 3000.0
# 建筑 footprint 面积下限（m²）：低于此值视为贴合/孔洞残片，丢弃
MIN_BUILDING_AREA_M2 = 1.0
# 接地形时 base_z 的下限（m）：防止 DEM 局部噪点把楼压到地平线以下
TERRAIN_MIN_Z_M = 0.0
# 禁飞区数据文件（相对数据目录）：缺省时该文件可有可无，没有就是没有禁飞区；
# 显式配置 world.nofly_path 后文件必须存在，拼错路径不能被静默当成"没有禁飞区"。
DEFAULT_NOFLY_PATH = "nofly.geojson"
# 禁飞区高度口径：absolute = 数据直接给 z_min_m/z_max_m（绝对高程，库内统一存储口径）；
# agl = 数据给 agl_min_m/agl_max_m，load_nofly 在入口处按 DEM 换算成绝对 z，
# NoFlyZone 契约内部永远只存绝对 z。
NOFLY_ALTITUDE_ABSOLUTE = "absolute"
NOFLY_ALTITUDE_AGL = "agl"
DEFAULT_NOFLY_ALTITUDE_MODE = NOFLY_ALTITUDE_ABSOLUTE

# —— 车辆上限（论文1 Eq.36 口径）----------------------------------------
DEFAULT_V_MAX_MPS = 20.0
DEFAULT_A_MAX_MPS2 = 3.0
DEFAULT_THRUST_MAX_N = 100.0
DEFAULT_MASS_KG = 5.0
# 净空：广州老城区巷宽 10–15 m，5 m 才让街道峡谷可通行（论文 10 m 会封死低空）
DEFAULT_CLEARANCE_M = 5.0

# —— 净空度量（交付数字的口径，不改变路径，只改变"报多远"）------------------
# 真几何净空只在路径周边这个半径内查找建筑；更远就没有意义（这条就是报出的净空上限）。
# 它同时决定净空复核对**建筑粗筛**要外扩多少：外扩不足会把窗外 3 m 的楼漏掉、把净空虚报成 30。
CLEARANCE_SEARCH_RADIUS_M = 30.0

# —— 能耗模型常数（论文1 Appendix E.5；无风 ⇒ 空速 = 地速）------------------
# 只作 ``EnergyModelParams`` 的默认值存在：pp1 的目标是最短时间，不产 ``energy_j``。
ENERGY_RHO_KG_M3 = 1.18
ENERGY_CD = 1.0
ENERGY_A_FRAME_M2 = 0.16
ENERGY_ROTOR_AREA_M2 = 0.5
ENERGY_P_PROFILE_HOVER_W = 100.0
GRAVITY_MPS2 = 9.81

# —— 算法：RRT（在用）--------------------------------------------------
# 参考实现是"格"：15/5 格在 (4, 1.5, 4) 的各向异性栅格上 = 水平 60 m、竖直 22.5 m
RRT_STEP_M = 60.0
RRT_MIN_STEP_M = 20.0
RRT_SDF_STEP_SCALE = 0.4          # 无量纲：step = clip(scale·sdf_m, min, max)
RRT_MAX_ITERS = 3000
RRT_GOAL_SAMPLE_RATE = 0.1
RRT_INITIALIZATIONS = 5           # 多初值：不同 seed 跑 N 次 RRT，选一条作为 NLP 初值
RRT_INITIALIZATIONS_MAX = 20      # 防误配成很大的值，把 RRT 时延线性放大

# —— 算法：轨迹层 NLP（在用；时间域，最短飞行时间）-----------------------
NLP_N_RESAMPLE = 60               # 固定节点数 N（论文 F.3；参考实现漏了这一步）
NLP_INITIAL_TIME_STEP_S = 1.0
NLP_SEGMENT_SAMPLES = 2
NLP_DT_MIN_S = 1e-3
NLP_DT_MAX_S = 500.0
NLP_IPOPT_MAX_ITER = 200

# —— RRT 实现细节（不属于可调口径，放这里只为"数值不散落"）----------------
RRT_SHORTCUT_ITERS = 200
# 沿线检查步长：RRT 判段内净空、交付前真几何复核，**共用这一个**（比栅格更密）。
# 不为两边各设一个步长：检查松、复核紧 = 求解器自认为合格却被自己判不合格。
CLEARANCE_SAMPLE_STEP_M = 1.0

# —— 请求默认 -------------------------------------------------------------
DEFAULT_SEED = 0                  # 属于请求：同算法不同 seed 是不同请求


def snapshot() -> dict[str, object]:
    """本文件的全部配置项（排好序），用于日志 / ``describe()`` / 对账。"""
    return {k: v for k, v in sorted(globals().items()) if k.isupper()}

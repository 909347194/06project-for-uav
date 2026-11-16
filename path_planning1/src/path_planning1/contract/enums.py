"""枚举：优化目标、运行状态、失败码。

失败码是本包对外承诺的一部分（服务化后按码降级），改动等于改契约。
"""

from __future__ import annotations

from enum import Enum


class Objective(str, Enum):
    """本包的优化目标。**只有 ``TIME`` 有实现者**（轨迹层 minimize 总时长）。

    ``ENERGY`` / ``LENGTH`` 是已定契约、尚无实现者的值：没有风场，`E = ∫P dt` 的口径不成立，
    别按 ``ENERGY`` 比较方案（见 README「已知限制」）。
    """

    TIME = "time"       # 最短飞行时间（默认，论文口径）
    ENERGY = "energy"   # 能耗最小（需要风场与功率模型，未实现）
    LENGTH = "length"   # 三维弧长最短（未实现，仅备选）


class Status(str, Enum):
    """运行状态。**现在只会产出 OK / INFEASIBLE**；其余三个是服务层预留（超时、入参非法、
    内部异常），没有代码会返回它们——后端别为它们写死分支，
    也别据此认为"超时会被自动标成 timeout"。
    """

    OK = "ok"
    INFEASIBLE = "infeasible"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"
    INTERNAL_ERROR = "internal_error"


class FailureCode(str, Enum):
    """失败细分：服务化后用于区分"该重试 / 该改参数 / 该人工介入"。

    **实际会返回的现在有 9 个**：``NONE``、``START_OR_GOAL_BLOCKED``、
    ``OUT_OF_FLIGHT_BAND``、``RRT_NO_PATH``、``PATH_CLEARANCE_VIOLATED``、``ROI_NO_PATH``
    （最后这个=折线出了走廊半宽、而 NLP 在半宽约束下解不出可交付轨迹）、
    ``ENDPOINT_IN_NOFLY``（端点落在静态禁飞区内），加上装配层准入的
    ``OUTSIDE_WORLD`` / ``OD_TOO_LONG``。
    ``OUTSIDE_WORLD`` / ``OD_TOO_LONG`` 由 ``PlanningContext`` 判（``plan_route_segment``
    拿不到世界边界与建场契约，所以不在这里判），属于"请求不该进核心"的拒单。
    其余是已定契约、但还没有实现者的码：``SOLVER_UNAVAILABLE``
    属服务层，``TIME_BUDGET_EXCEEDED`` 属预算控制。
    ``NLP_*`` 三个**不会返回**：NLP 失败是降级（``trajectory=None`` + ``solver_status`` + warning），
    因为"几何可飞、时间剖面没解出来"不是"这条航段飞不过去"。
    """

    NONE = "none"
    START_OR_GOAL_BLOCKED = "start_or_goal_blocked"
    ENDPOINT_IN_NOFLY = "endpoint_in_nofly"      # 端点落在静态禁飞棱柱内（含边界）
    OUT_OF_FLIGHT_BAND = "out_of_flight_band"
    OUTSIDE_WORLD = "outside_world"          # 端点超出城市 bounds_m（装配层准入）
    OD_TOO_LONG = "od_too_long"              # OD 超过建场契约 max_od_length_m（装配层准入）
    RRT_NO_PATH = "rrt_no_path"
    PATH_CLEARANCE_VIOLATED = "path_clearance_violated"   # 解出来了，但不满足净空（求解器缺陷，不是"没路"）
    NLP_INFEASIBLE = "nlp_infeasible"
    NLP_MAX_ITER = "nlp_max_iter"
    ROI_NO_PATH = "roi_no_path"
    SOLVER_UNAVAILABLE = "solver_unavailable"
    TIME_BUDGET_EXCEEDED = "time_budget_exceeded"

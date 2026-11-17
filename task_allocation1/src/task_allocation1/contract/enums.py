"""枚举：状态、模型形态、终点机巢策略。

终点机巢策略 ``EndNestPolicy`` 只导出 ``HOME`` 一个值：本库固定同巢起降。
保留枚举形状是为了让调用方/测试能显式声明口径，而不是留了个能切换的开关。
"""

from __future__ import annotations

from enum import Enum


class TaskStatus(str, Enum):
    """任务状态（单周期内核只用 PENDING；其余供上游状态机复用）。"""

    PENDING = "PENDING"
    SELECTED = "SELECTED"
    EXECUTING = "EXECUTING"
    DONE = "DONE"
    EXPIRED = "EXPIRED"
    UNREACHABLE = "UNREACHABLE"
    DORMANT = "DORMANT"


class UAVStatus(str, Enum):
    """无人机状态；只有 IDLE 参与分配。"""

    IDLE = "IDLE"
    BUSY = "BUSY"
    CHARGING = "CHARGING"
    FAULT = "FAULT"


class ModelType(str, Enum):
    """编码模型形态（K=参与机数，N=任务数）。

    本库口径：任务唯一分配（一个任务只被一架机完成），一机可串飞多任务，
    K 恒满足 ``K <= max(1, N-1)``；机多于任务时未选中的机闲置。
    因此实际只会产生 ``SRP`` 与 ``EMPTY`` 两种。
    """

    BALANCED = "balanced"      # K == N。已停用：本库不再产生，仅为兼容保留
    OVERLOADED = "overloaded"  # K > N。已停用：本库不再产生，仅为兼容保留
    SRP = "srp"                # K <= N-1（N=1 时 K=1，巡游维为 0）
    EMPTY = "empty"            # 无任务或无可用机


class EndNestPolicy(str, Enum):
    """终点机巢口径。本库唯一实现 HOME：从哪个巢起飞就返回哪个巢。"""

    HOME = "home"

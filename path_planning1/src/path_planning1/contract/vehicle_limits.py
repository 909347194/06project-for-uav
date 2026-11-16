"""车辆上限与能耗模型常数（数值来自 ``contract/defaults.py``）。

净空 ``clearance_m`` 只在这里出现**一次**（参考实现在两处重复，易漂移）。
"""

from __future__ import annotations

from dataclasses import dataclass

from . import defaults


@dataclass(frozen=True)
class VehicleLimits:
    """多旋翼的运动学/动力学上限（论文1 Eq.36 口径）。"""

    v_max_mps: float = defaults.DEFAULT_V_MAX_MPS
    a_max_mps2: float = defaults.DEFAULT_A_MAX_MPS2
    thrust_max_n: float = defaults.DEFAULT_THRUST_MAX_N
    mass_kg: float = defaults.DEFAULT_MASS_KG
    clearance_m: float = defaults.DEFAULT_CLEARANCE_M


@dataclass(frozen=True)
class EnergyModelParams:
    """多旋翼能耗模型常数（论文1 Appendix E.5；无风 ⇒ 空速 = 地速）。"""

    rho_kg_m3: float = defaults.ENERGY_RHO_KG_M3
    cd: float = defaults.ENERGY_CD
    a_frame_m2: float = defaults.ENERGY_A_FRAME_M2
    rotor_area_m2: float = defaults.ENERGY_ROTOR_AREA_M2
    p_profile_hover_w: float = defaults.ENERGY_P_PROFILE_HOVER_W
    gravity_mps2: float = defaults.GRAVITY_MPS2

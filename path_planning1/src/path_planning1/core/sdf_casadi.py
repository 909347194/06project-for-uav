"""``SdfField`` → CasADi 三线性插值：NLP 里的"净空"就是这一层的读数。

单独成文件，是因为它是**SDF 与求解器之间的转接头**（索引空间、插值器命名、ENU↔索引换算），
和"时间最优怎么建模"是两件事；混在一起会让 ``trajectory_optimization.py`` 超过单文件上限，
也会让"换个距离场"变成改 NLP 文件。

注意 ``ca.interpolant`` 是**一个 3 维输入**的函数（不是三个标量入参），``at()`` 里必须 vertcat。
"""

from __future__ import annotations

import itertools

try:  # casadi 是**可选依赖**（``pip install "path_planning1[nlp]"``）：没装时几何层照常用
    import casadi as ca
except ImportError:  # pragma: no cover - 只在瘦装环境触发
    ca = None

import numpy as np

from .occupancy_sdf import SdfField

_INTERPOLANT_SEQ = itertools.count()   # 同一进程多次求解时给 Function 换名字，避免重名


class SdfOnGrid:
    """ENU 米制入参、索引空间内插的 SDF 读数器。"""

    def __init__(self, field: SdfField) -> None:
        if ca is None:
            raise RuntimeError("未安装 casadi：pip install \"path_planning1[nlp]\" 后重跑")
        grid = field.grid
        self.shape = tuple(int(v) for v in grid.shape)
        axes = [np.arange(n, dtype=float).tolist() for n in self.shape]
        values = np.asarray(field.sdf, dtype=float).ravel(order="F").tolist()  # i 变化最快
        self._fun = ca.interpolant(f"sdf_linear_{next(_INTERPOLANT_SEQ)}", "linear", axes, values)
        self._a0, self._a1 = grid.axes_xy
        self._spacing = tuple(float(v) for v in grid.spacing_m)
        self._origin = tuple(float(v) for v in grid.origin_enu)

    def index_of(self, x3: ca.MX) -> tuple[ca.MX, ca.MX, ca.MX]:
        """3×N 的 MX → 三个连续索引（沿走廊轴、横向、绝对 z）。"""
        sx, sy, sz = self._spacing
        ox, oy, oz = self._origin
        dx, dy = x3[0, :] - ox, x3[1, :] - oy
        ii = (self._a0[0] * dx + self._a0[1] * dy) / sx
        jj = (self._a1[0] * dx + self._a1[1] * dy) / sy
        kk = (x3[2, :] - oz) / sz
        return ii, jj, kk

    def at(self, p: ca.MX) -> ca.MX:
        """单点（3×1 MX）处的 SDF 值（米，正 = 自由）。"""
        ii, jj, kk = self.index_of(p)
        return self._fun(ca.vertcat(ii, jj, kk))

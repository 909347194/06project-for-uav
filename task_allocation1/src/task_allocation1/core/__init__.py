"""算法层：只依赖 contract，不碰 I/O。

数据流：``cost``（代价/距离注入）→ ``tour``（航次原语）→ ``problem``（问题实例）
→ ``encoding``/``inverse_map``/``operators``（DMDE）→ ``repair``（修复链）
→ ``polish``（交付后精修）→ ``solver.allocate``（编排）。
``vertical_section`` 是默认的距离尺子（垂直切面估计），``euclid3d`` 供无地形场景显式选用，
两者都由 ``cost_config`` 选择。

终点机巢恒为起飞巢（同巢起降）；本层不提供异巢降落求解器。
"""

"""算法内核 + 启动装配（只依赖 ``contract``、第三方库与本地数据文件，无 Web 依赖）。

对外不直接从这里导入：正式门面在包顶层 ``path_planning1``（``__init__.py`` 做 re-export）。
``core`` 里的 ``planning_context.py`` 是有状态的启动装配层（装载世界、建场、覆盖自检、
准入），其余模块是无状态的纯计算；两者都不碰 HTTP / 缓存 / 进程池。

模块清单（一个文件一件事，文件名 = 它干的事）::

    city_dataset.py       数据加载的唯一入口：CityDataset + load_city（顺序固化）
    dataset_identity.py   数据集内容指纹（缓存键的 world_id 段；不进 params_hash）
    field_artifact.py     全域场只读产物 save/load（npy mmap + meta，含契约校验）
    field_contract.py     建场契约派生：pad 与 build_world_field 入参的唯一出处
    planning_context.py   启动装配 + 建场契约：一次装载/取场/覆盖自检，准入与在线只取
    city_loader.py        shp / geojson → ENU 米制数据（GBK 容错），含 nofly.geojson
    dem_terrain.py        DEM 读标签 + 地面高程插值 + 逐栋接 base_z
    corridor_window.py    ROI 走廊 → 栅格窗口；required_pad_m 建场外扩契约；真高带 → z 包络
    occupancy_sdf.py      Building + NoFlyZone → occupancy / SDF（窗口内两遍 EDT）
    world_field.py        全域场：整城一次算完 occupancy / SDF；覆盖自检 + 场身份签名
    clearance_check.py    真几何净空 BuildingPrisms（建筑+禁飞区）：交付数字 + 失败终检闸
    rrt_path_search.py    RRT 3D 寻路（SDF 自适应步长、米制）
    route_segment_planner.py        单航段规划纯函数（服务主线经 PlanningContext.plan 调用）
    route_segment_window.py         一条航段的窗口装配：走廊 → 窗口 + 地形 + z 包络
    route_segment_xyz.py            绝对 (x,y,z) 端点入口（按 DEM 换算真高后转调上层入口）
    route_segment_params.py         请求口径 RouteSegmentParams（含默认值）
    route_segment_trajectory.py     轨迹层装配：几何路径 → 时间剖面（含净空复测与降级留痕）
    route_segment_outcome.py        结果形状：RouteSegmentStats / RouteSegmentPlan / rejected_plan
    trajectory_optimization.py      时间域 NLP（CasADi/IPOPT，目标=最短时间；casadi 可选）
    trajectory_sampling.py          NLP 的离散化与初值（重采样、中心差分、采样矩阵）
    sdf_casadi.py                   栅格 SDF → CasADi 三线性插值（供 NLP 的净空约束）
    params_config.py      **配置入口**：一份 JSON 配齐全部参数（缺省即默认、未知键报错、量值有界）
    params_fingerprint.py 参数指纹（缓存键的唯一来源）
    request_check.py      请求入参自检（NaN/零间距 → ValueError，0 秒，属调用方 bug）
"""

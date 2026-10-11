# 28 cm 上台后规范站立：第二阶段

## 2026-10-11：降低三脚静止的正奖励

仅修改 Stand/Stand-PLAY 任务的上台后收腿阶段，不改变平地、旧攀爬任务或攀爬阶段奖励：

- 少于四脚实际支撑时，线速度和偏航跟踪奖励乘 `unsupported_scale=0.1`。三脚静止且速度完全匹配时，这两项的合计上限由 4.5/秒降至 0.45/秒。
- 支撑奖励由“接触脚数/4”改为“四脚均有效支撑才给”；三脚不再获得 75% 的支撑奖励。
- 站姿质量奖励也要求四脚支撑；悬一脚时该项为零。
- 四脚支撑时保留原有跟踪、支撑、质量奖励；姿态惩罚和严格成功标准不变。
- `unsupported_scale` 可在 `go2_demo_high_step.py` 中两项速度奖励的 `params` 内修改，范围 0～1。

这是奖励配置变更，不会自动改变已训练的 `model_19491.pt` 的动作。要使策略学会新偏好，需要后续微调；本次修改没有启动新一轮训练，也没有覆盖旧 checkpoint 或旧测评报告。

## 与第一阶段的区别

第一阶段 `model_17492.pt` 的 128/128 是旧的攀爬判定通过率：允许至少两脚支撑、脚悬在台面上方、速度小于 0.35 m/s，保持 0.6 秒。本阶段不把这个数字当作规范站姿的通过率。

新增独立任务 `Go2-High-Step-28cm-Stand-v0`，从 `model_17492.pt` 微调。旧任务与旧模型保留；仍然是 235 维观测、原网络、0.25 动作缩放、相加奖励，不使用 AER。

## 规范站立的明确标准

上台后速度指令归零，以下条件必须**连续保持 2 秒**：

- 四个足端全部超过台阶边缘，每脚向上的支撑力 >5 N；推台阶立面的横向力不计作支撑。
- 足端高度在台面附近，不能把抬在空中的腿算作落地。
- 机身线速度 <0.08 m/s，角速度 <0.20 rad/s，合成倾斜角 <10°。
- 机身离台面高度接近默认站姿，误差 <5 cm。
- 每个髋关节相对默认角度误差 <0.12 rad，其他关节误差 <0.35 rad。
- 每个足端在机身坐标系中的横向/纵向位置与默认站姿相差 <5 cm。

这些是本阶段可测量的验收标准，不保证关节精确等于默认值，也不是所有地形上的通用站姿。默认姿态是左髋 +0.1、右髋 -0.1、前大腿 0.8、后大腿 1.0、小腿 -1.5 rad。足端参考由本地 Go2 几何正运动学计算；单元测试与本地 MuJoCo 模型核对，实测脚本还核对 Isaac 初始默认姿态的足端位置。

## 训练设计

- 攀爬阶段保持原奖励，允许抬腿和俯仰。
- 已进入平台的收腿阶段才启用关节、足端站距、机身水平、高度和静止约束，并停止速度指令。
- 训练回合 75% 从台阶下方开始，25% 从平台上方以默认姿态开始，专门练静止与收腿；课程日志分别统计整段攀爬站立与平台练习，不能混算。
- **PLAY 与最终测评 100% 从台阶下方出发**，无平台出生回合。
- 第一轮预算追加 2000 轮、1024 环境；固定学习率 5e-5；探索动作 std 从旧模型约 0.5 重置为 0.15，熵系数 0.0003。只改变随机探索噪声，不重置确定性动作网络或观测归一化器。
- 完成预算不等于达标；结束后自动用确定性动作做 128 次严格测评。检查攀爬率和严格站立率，不只看总奖励。

## 一行命令

在 `D:\RobotProject\go2_demo`、`env_isaaclab` 环境运行：

```bat
python scripts\rsl_rl\train.py --task Go2-High-Step-28cm-Stand-v0 --headless --num_envs 1024 --resume --load_policy_only --checkpoint "D:\RobotProject\go2_demo\logs\rsl_rl\go2_high_step_28cm\2026-10-09_22-52-43_28cm_from12493_20261009_225234\model_17492.pt" --reset_action_noise_std 0.15 --max_iterations 2000 --run_name stand_from17492
```

追加 2000 轮通常保存至 `model_19491.pt`，新模型目录是 `logs/rsl_rl/go2_high_step_28cm_stand/日期时间_运行名/`。

```bat
python scripts\rsl_rl\evaluate_high_step.py --headless --task Go2-High-Step-28cm-Stand-Play-v0 --checkpoint "实际新模型完整路径" --step_height 0.28 --num_envs 32 --episodes_per_env 4 --output "严格测评结果.json"
```

```bat
python scripts\rsl_rl\play.py --task Go2-High-Step-28cm-Stand-Play-v0 --num_envs 4 --real-time --checkpoint "实际新模型完整路径"
```

## 主要改动

- 新增 `mdp/high_step_stand.py`：默认姿态 FK、收腿阶段、四脚受力及规范站姿判定、分阶段奖励、分别统计的日志。
- `go2_demo_high_step.py`：新增 Stand 和 Stand-PLAY 类，不更改旧任务标准。
- `agents/rsl_rl_ppo_cfg.py`：新增 `HighStepStandPPORunnerCfg`。
- 任务和 MDP 的 `__init__.py`：注册/导出新任务。
- `train.py`：新增可选 `--reset_action_noise_std`；不传时旧训练行为不变。
- `evaluate_high_step.py`：新增任务/seed 参数，严格结果输出支撑数、关节偏差、足端偏差、倾角和速度等数据。
- `test_high_step.py`：验证悬腿、立面接触、内收姿态、错误足端站位和未停止的情况不能通过；正常姿态保持、分阶段奖励和测评快照测试。
- `run_high_step_training.ps1`：支持新任务与新日志目录，并在等待前保留进程句柄、记录明确退出码。不会因为只有训练日志就假定训练/测评成功。

## PLAY 显示：哑光浅蓝机器人与白色无贴图地面

按用户选中的版本，Stand-PLAY 恢复不透明哑光浅蓝 `PreviewSurface`：diffuse_color=(0.35,0.60,0.85)、roughness=0.8、metallic=0、opacity=1。只覆盖播放实例的机器人视觉材质，原有 Go2 字样/贴图会被纯色覆盖；原始 USD、训练用机器人、碰撞/摩擦和控制参数未修改。没有恢复此前昂贵的 DLAA 或额外光照采样设置。

机器人与地面是两个独立的视觉材质：当前 PLAY 地面改为白色无贴图的 `PreviewSurface`，覆盖从 rough-terrain 配置继承的石砖 MDL。28 cm 台阶几何保持不变，不能为了画面像早期平地而把地形换成 plane；地面物理材质与摩擦不变。早期平地的基础 GroundPlane 显示与生成网格地形的默认石砖显示不同。

先前新增的 DLAA/FXAA、额外光照采样和反射/降噪覆盖也已撤掉，直接使用 `--rendering_mode performance` 预设。单机器人、较小窗口可降低播放开销；默认 viewer 分辨率设置为 960×540。

该策略控制周期为 0.02 秒（50 Hz）。`--real-time` 会限制播放循环的速度；省略它可观察最大播放速度，但仿真时间可能快于现实时间。没有为了显示 60 FPS 修改物理 dt 或控制 decimation，不能仅凭材质变更保证恒定 60 FPS。

```bat
python scripts\rsl_rl\play.py --task Go2-High-Step-28cm-Stand-Play-v0 --num_envs 1 --device cuda:0 --rendering_mode performance --checkpoint "D:\RobotProject\go2_demo\logs\rsl_rl\go2_high_step_28cm_stand\2026-10-10_23-54-18_stand_resume18300_20261010_235359\model_19491.pt" env.scene.terrain.terrain_generator.num_cols=1
```

已经打开的 Python 进程不会自动加载这些显示改动，需要关闭后重新执行播放命令。点击视口空白处取消对机器人 Mesh 的选择，可去掉选中高亮/操纵轴的干扰。

### GPU 拖动报错与窗口复位

GPU 模式不支持原生鼠标加力拖拽：其 CPU `PxArticulationLink::addForce/addTorque` 调用与 Direct GPU API 不兼容。`play.py` 现在启动时关闭 `/physics/mouseInteractionEnabled`，并在播放循环中保持关闭；没有关闭 GPU 物理、Fabric 或屏蔽错误日志。GPU 播放请不要重新启用原生 Shift 拖拽，也不要用 Mesh 的 USD 变换代替物理状态修改。

右侧 `IsaacLab → Simulation Settings` 新增 **Reset robot to start** 按钮。按钮只排队请求，由播放循环安全调用环境 `reset()`，同时重置动作/观测缓存和策略状态。暂停时点击也会完成复位并恢复暂停；该按钮重置所有播放机器人，不改变模型、不启动训练。

复位检查可在播放命令末尾加 `--play_steps 20 --reset_at_step 5`，运行 20 个策略步，在第 5 步执行一次复位后退出。正常播放不需要这些检查参数。

## 意外重启后的恢复

从最近保存的 `model_18300.pt` 继续到原计划的最终编号，追加约 1192 轮。相同任务的断点恢复要保留优化器和当前动作噪声：直接运行 `train.py --resume` 时不要加 `--load_policy_only` 或 `--reset_action_noise_std`；使用后台脚本时传入 `-LoadOptimizer`，并不设置 `-ActionNoiseStd`。原始模型和旧日志保留，新作业前缀为 `stand_resume18300`。

这次恢复保留学习率 5e-5 和 checkpoint 中约 0.10 的探索 std，不重新从 0.15 开始。重新启动不等于严格站姿已通过；最终仍需查看该作业的严格测评结果。

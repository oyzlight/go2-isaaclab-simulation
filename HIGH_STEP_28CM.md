# Go2 单级 28 cm 台阶续训

## 目标与边界

本阶段训练的是**从低处爬上一个垂直高度 28 cm、顶部足够宽的平台**，不是楼梯累计升高 28 cm，也不包含下 28 cm 台阶。允许机器人通过抬前腿、俯仰机身、后腿推进完成攀爬；没有添加坡道或降低真实碰撞高度。

初始化模型：

```text
D:\RobotProject\go2_demo\logs\rsl_rl\go2_velocity_stairs_height_transfer\2026-10-01_10-58-57_stairs_lateral_speed\model_12493.pt
```

保留 235 维输入（48 本体观测 + 187 高度点）、原网络 `[128,128,64]`、观测归一化器、动作缩放 0.25、PD 和 23.5 Nm 电机限力。使用普通相加 RewardManager，不使用 AER。旧任务和旧 checkpoint 不覆盖。高度网格与动作/关节顺序不变，所以不需要再扩展 checkpoint。

## 课程与奖励

- 单级高度：6、8、10、12、14、16、18、20、22、24、26、28 cm。初始随机分配 6～10 cm。
- 每个环境连续成功 3 次上升一级；连续失败 3 次下降一级。最终级不绕回低级。
- 成功要求：四只脚都越过台阶边缘且达到台面高度，至少两脚支撑，机身足够高且基本直立，平移速度 <0.35 m/s，角速度 <1 rad/s，持续 0.6 秒。
- 起点面朝台阶，速度指令约 0.35～0.55 m/s；接近台上目标位置后减速，并纠正横向偏离与航向。
- 保留原速度跟踪及动作、力矩、关节加速度、非足部触地惩罚；增加实际前进奖励、每脚首次上台奖励、完成奖励和失败惩罚。脚奖励不能通过反复踩台刷分。
- 垂直速度惩罚从 -2 调到 -0.2，角速度 xy 从 -0.05 调到 -0.02；关闭平地姿态惩罚，改为仅惩罚侧倾。允许爬高所需俯仰，但不取消关节软限位和机身触地终止。
- 正式续训默认 1024 环境、每轮每环境 32 步、追加 5000 轮，固定学习率 1e-4。保留左右镜像，镜像损失系数 0.05。
- 加载网络与归一化器，**不恢复旧 Adam 优化器**；这避免新任务沿用旧优化器的学习率/动量。模型迭代编号仍从 12493 起，追加 5000 轮通常到 `model_17492.pt`。

这是一项新的高台攀爬技能，5000 轮是本轮训练预算，并不是成功保证。不能仅看总奖励或平均课程等级判断能否爬 28 cm。

## 启动

在 `D:\RobotProject\go2_demo`、`env_isaaclab` 环境运行（一行）：

```bat
python scripts\rsl_rl\train.py --task Go2-High-Step-28cm-v0 --headless --num_envs 1024 --resume --load_policy_only --checkpoint "D:\RobotProject\go2_demo\logs\rsl_rl\go2_velocity_stairs_height_transfer\2026-10-01_10-58-57_stairs_lateral_speed\model_12493.pt" --max_iterations 5000 --run_name high_step_28cm
```

需要训练结束后自动测试，可用 PowerShell 的 `scripts/rsl_rl/run_high_step_training.ps1`。它将训练与测试顺序执行，记录 PID、状态和错误，不会同时运行两个 GPU 仿真。作业日志位于 `logs/high_step_jobs/28cm_from12493_日期时间/`；模型和 TensorBoard 位于 `logs/rsl_rl/go2_high_step_28cm/日期时间_运行名/`。`status.json` 的 `completed` 仅表示训练和测评完成，实际能力看 `evaluation_result.success_rate`。

## 看效果

替换实际 checkpoint 路径后，固定 28 cm 场景播放：

```bat
python scripts\rsl_rl\play.py --task Go2-High-Step-28cm-Play-v0 --num_envs 16 --checkpoint "实际新模型完整路径"
```

定量测试（32 环境 × 每环境 4 次 = 128 次，不使用随机动作）：

```bat
python scripts\rsl_rl\evaluate_high_step.py --headless --checkpoint "实际新模型完整路径" --step_height 0.28 --num_envs 32 --episodes_per_env 4 --output "logs\rsl_rl\go2_high_step_28cm\evaluation_28cm.json"
```

训练前后使用相同脚本、seed、台阶高度和重置分布对照。初步验收目标：固定 28 cm 的 128 次测试成功率达到 80%，再测试不同 seed、起点、速度和动力学扰动；最后导出到 MuJoCo 重验，Isaac Lab 通过不等于 sim2sim 已通过。

```bat
tensorboard --logdir logs\rsl_rl\go2_high_step_28cm
```

重点查看 `Curriculum/terrain_levels/mean_height_cm`、`fraction_at_28cm`、`success_cumulative_28cm`、`attempts_28cm`，以及 `Episode_Termination/base_contact`。**没有 28 cm 尝试时的成功率 0，不代表已有测试失败；课程训练累计成功率也不等于固定高度的确定性测评成功率。**

## 代码位置与测试

- `source/.../go2_demo/go2_demo_high_step.py`：独立任务、台阶高度、奖励、重置。
- `source/.../go2_demo/high_step_geometry.py`：真实垂直单台阶网格。
- `source/.../go2_demo/mdp/high_step.py`：到平台目标的速度指令、四足成功判定和课程。
- `agents/rsl_rl_ppo_cfg.py`：`HighStepPPORunnerCfg`，保持 checkpoint 网络尺寸。
- `scripts/rsl_rl/test_high_step.py`：地形各级高度、成功保持条件、课程升降级测试。
- `scripts/rsl_rl/evaluate_high_step.py`：固定高度测评，结果 JSON。

设计依据：[Isaac Lab 地形生成接口](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.terrains.html)支持自定义几何及难度分层；[Robot Parkour Learning](https://arxiv.org/abs/2309.05665)将高障碍攀爬作为专门技能学习。本任务只是采用专门场景与渐进训练思路，并未复现论文算法，也不据此保证 Go2 必定爬过 28 cm。

# 两层彩色台阶：高画质播放场景

## 当前使用：已切回一级 28 cm

按后续要求，当前展示使用独立的 `Go2-High-Step-28cm-Color-Play-v0`：一个 28 cm 台阶，橙色台阶、白地面、浅蓝机器人及高画质设置不变。两级版本仍作为备用任务保留，但不再用于当前播放。单级目标位置和顶部高度也恢复为一级场景的值。

```bat
python scripts\rsl_rl\play.py --task Go2-High-Step-28cm-Color-Play-v0 --num_envs 1 --device cuda:0 --rendering_mode quality --real-time --checkpoint "D:\RobotProject\go2_demo\logs\rsl_rl\go2_high_step_28cm_stand\2026-10-10_23-54-18_stand_resume18300_20261010_235359\model_19491.pt"
```

## 备用：两级版本

- 第一阶：28 cm。
- 第二阶：再升高 28 cm，顶部距原地面 56 cm。
- 第一阶踏面深 70 cm；第二阶上方是宽平台。
- 白色无贴图地面、两档橙色台阶、哑光浅蓝机器人。
- 恢复 DLAA、4 次光照采样、反射、全局光照、环境遮蔽、AI 降噪；播放命令使用 `--rendering_mode quality`。这一版以画质为主，不承诺 60 FPS。

新增独立任务 `Go2-High-Step-Two-Level-Play-v0`，旧的单台阶训练/播放场景不变。目标速度会一直指向第二阶顶部，四脚支撑和站姿判定也以总高 56 cm、第二阶边缘为准，不会在第一阶提前结束。

```bat
python scripts\rsl_rl\play.py --task Go2-High-Step-Two-Level-Play-v0 --num_envs 1 --device cuda:0 --rendering_mode quality --real-time --checkpoint "D:\RobotProject\go2_demo\logs\rsl_rl\go2_high_step_28cm_stand\2026-10-10_23-54-18_stand_resume18300_20261010_235359\model_19491.pt"
```

这只是用原有 235 维策略观察新场景，不是新的训练成果。旧模型只训练过单级 28 cm，尚未证明能连续爬两级，也不把旧场景的成功率沿用到这个场景。当前每回合最多 24 秒仿真时间；超时、倒地或真正完成都会复位。

`Reset robot to start` 按钮及 GPU 原生鼠标加力禁用保护保留。没有修改模型权重、物理步长、PD、电机限力或地面摩擦；台阶颜色只作用于显示。

配置在 `go2_demo_two_step.py`。可修改 `TwoStepTerrainCfg.height_levels=(0.28,)` 和 `step_depth=0.70`；总高由两级相加得出。场景通过三维真实台阶网格实现，没有使用视觉模型覆盖碰撞体，也没有把台阶做成斜坡。

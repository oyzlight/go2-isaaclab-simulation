# Unitree Go2 强化学习训练总结

## 1. 最终有效训练路线

本项目最终形成了以下训练流程：

```text
原始 48 维平地策略
    ↓
无损扩展到 235 维高度扫描
    ↓
微起伏地形适应
    ↓
混合复杂地形课程
    ↓
楼梯专项训练
    ↓
前进速度课程
    ↓
左右镜像数据增强
    ↓
Mirror Loss
    ↓
增加横向移动速度
```

当前最新模型：

```text
logs/rsl_rl/go2_velocity_stairs_height_transfer/
2026-10-01_10-58-57_stairs_lateral_speed/model_12493.pt
```

最新模型最近 50 次平均指标：

| 指标 | 数值 |
|---|---:|
| 楼梯课程等级 | 7.51 / 9 |
| XY 速度误差 | 0.314 |
| yaw 误差 | 0.226 |
| 机身触地率 | 5.66% |
| 平均回合长度 | 969 / 1000 |
| Symmetry Loss | 0.121 |

当前大致掌握约 8～9 cm 台阶。

---

## 2. 训练阶段回顾

| 阶段 | 训练方法 | 结果与结论 |
|---|---|---|
| 原始平地训练 | 48 维本体观测、普通奖励相加 | 效果良好，`model_2999.pt` 能稳定行走 |
| AER 奖励实验 | 正奖励乘辅助惩罚指数 | 调参困难，容易出现站立、不走或奖励被压低 |
| 直接加入高度扫描 | 输入从 48 维增加到 235 维并从零训练 | 原步态丢失，容易出现姿态异常、倒地或不动 |
| 增大网络 | 从 `[128,128,64]` 改成 `[512,256,128]` | 增加了容量，但没有解决原策略行为丢失的问题 |
| 无高度扫描迁移 Rough | 使用旧 48 维模型适应粗糙地面 | 能适应部分粗糙面，但无法提前感知第一个台阶和落差 |
| Checkpoint 无损扩展 | 复制旧 48 维权重，新增 187 维权重初始化为 0 | 关键突破，加入高度扫描后原行为保持不变 |
| 微起伏 Stage 0 | 最高约 1.5 cm 随机起伏 | 触地率 0%，XY 误差约 0.094 |
| 混合复杂地形 | 楼梯、方块、坡面、粗糙面 | 1024 环境明显比 128 环境稳定 |
| 楼梯专项初版 | 台阶 2～10 cm，速度范围 `-0.4～0.4` | 很稳定，但课程等级长期停在 0.49，移动速度太慢 |
| 前进速度课程 | x 速度改成 `0.2～0.8` | 课程等级升至 5.89，掌握约 6～8 cm 台阶 |
| 镜像数据增强 | 在线生成左右镜像观测和动作 | 课程等级升至 6.55，左右差异有所改善 |
| Mirror Loss | 直接约束左右镜像动作一致 | Symmetry Loss 从约 0.65 降至 0.134 |
| 提升前进速度 | x 速度改成 `0.3～1.0` | 保持稳定，同时提高前进速度 |
| 提升横向速度 | y 速度从 `±0.05` 提高到 `±0.15` | 课程等级升至 7.51，获得更明显横移能力 |

---

## 3. 关键经验

### 3.1 不要直接改变已训练 Actor 的输入维度

原模型第一层为：

```text
[128, 48]
```

直接加入高度扫描后变为：

```text
[128, 235]
```

这种变化会导致旧 checkpoint 无法直接加载。从零训练 235 维网络也容易丢掉已经形成的稳定步态。

最终采用的正确方法：

```text
复制原有 48 维权重
新增 187 维高度权重全部初始化为 0
```

这样迁移后的初始动作与旧模型一致，高度通道再通过后续训练逐渐学习。

转换脚本：

```text
scripts/rsl_rl/expand_height_checkpoint.py
```

迁移后的初始 checkpoint：

```text
logs/rsl_rl/go2_velocity_flat_height_transfer/model_2999_height_init.pt
```

### 3.2 最终有效路线使用普通加法奖励

最终 Height Transfer、Rough Transfer 和 Stairs Transfer 使用：

```text
ManagerBasedRLEnv
RewardManager
```

即普通奖励加权求和。

AER 文件仍保留在项目中，但最终有效迁移任务没有使用 AER。AER 并非一定错误，但在当前项目中增加了奖励尺度和调试难度。

### 3.3 环境数量提高了训练稳定性

128 个环境时：

- 地形类型覆盖不充分；
- PPO 批次较小；
- 训练曲线波动明显；
- 摔倒率较高。

提高到 1024 个环境后：

- 不同地形覆盖更加充分；
- 梯度估计更稳定；
- 速度误差更平滑；
- 触地率明显下降。

但增加环境数不能修复错误的奖励设计或课程条件。

### 3.4 速度指令直接影响地形课程升级

Isaac Lab 的 terrain curriculum 通常要求机器人单回合移动超过半个地形块。本项目每块地形长度约 8 m，因此升级需要移动约 4 m。

原楼梯专项指令：

```python
lin_vel_x = (-0.4, 0.4)
```

其中大量接近零的速度指令无法在 20 秒内移动 4 m，因此机器人虽然稳定，课程等级却停在约 0.49。

修改后：

```python
lin_vel_x = (0.3, 1.0)
lin_vel_y = (-0.15, 0.15)
ang_vel_z = (-0.2, 0.2)
rel_standing_envs = 0.0
rel_heading_envs = 0.0
heading_command = False
```

课程等级最终升至 7.51。

### 3.5 混合地形指标会掩盖楼梯问题

混合训练中，机器人可能主要掌握：

- 随机粗糙面；
- 坡地；
- 简单方块。

总体 reward 较高并不代表已经掌握上下楼梯。

因此后来增加楼梯专项：

```text
50% 上楼梯
50% 下楼梯
其他地形比例为 0
```

这一步是楼梯能力提升的关键。

### 3.6 左右镜像不是简单交换腿

完整的左右镜像需要处理：

- FL 与 FR 关节交换；
- RL 与 RR 关节交换；
- hip 关节符号翻转；
- y 方向速度翻转；
- roll 和 yaw 角速度翻转；
- 横向速度和 yaw 指令翻转；
- 高度图左右翻转；
- 输出动作左右镜像。

实现文件：

```text
source/go2_demo/go2_demo/tasks/manager_based/go2_demo/mdp/symmetry.py
```

训练配置：

```python
RslRlSymmetryCfg(
    use_data_augmentation=True,
    use_mirror_loss=True,
    data_augmentation_func=mdp.compute_symmetric_states,
    mirror_loss_coeff=0.1,
)
```

最终 Symmetry Loss：

```text
约 0.74 → 0.121
```

### 3.7 训练指标和 Play 表现不同

训练时动作来自：

```text
策略均值 + 随机探索噪声
```

Play 使用：

```text
策略均值
```

因此曾出现训练日志看起来会移动，但 Play 中原地站立的情况。根因是策略依赖随机动作，而确定性均值没有学会行走。

后续通过以下措施缓解：

- 无损 checkpoint 扩展；
- 减小 entropy；
- 低学习率微调；
- 从稳定旧策略继续训练。

### 3.8 最后一个 checkpoint 不一定最好

训练过程中曾出现：

- 速度误差继续下降；
- 但触地率回升；
- terrain level 下降。

因此不能只看总 reward，也不能默认最新 checkpoint 最好。应共同观察：

```text
Curriculum/terrain_levels
Episode_Termination/base_contact
Metrics/base_velocity/error_vel_xy
Metrics/base_velocity/error_vel_yaw
Train/mean_episode_length
Loss/symmetry
```

---

## 4. 当前主要配置

### 楼梯地形

```python
step_height_range = (0.02, 0.10)
pyramid_stairs.proportion = 0.5
pyramid_stairs_inv.proportion = 0.5
```

训练地形共 10 级，台阶高度大致为：

| 等级 | 台阶高度范围 |
|---:|---:|
| 0 | 2.0～2.8 cm |
| 1 | 2.8～3.6 cm |
| 2 | 3.6～4.4 cm |
| 3 | 4.4～5.2 cm |
| 4 | 5.2～6.0 cm |
| 5 | 6.0～6.8 cm |
| 6 | 6.8～7.6 cm |
| 7 | 7.6～8.4 cm |
| 8 | 8.4～9.2 cm |
| 9 | 9.2～10.0 cm |

### 当前速度范围

```python
lin_vel_x = (0.3, 1.0)
lin_vel_y = (-0.15, 0.15)
ang_vel_z = (-0.2, 0.2)
```

### PPO 迁移设置

```python
learning_rate = 3e-4
entropy_coef = 0.001
actor_hidden_dims = [128, 128, 64]
critic_hidden_dims = [128, 128, 64]
```

---

## 5. 代码结构

| 文件 | 职责 |
|---|---|
| `source/go2_demo/go2_demo/tasks/manager_based/go2_demo/go2_demo_velicoty.py` | 平地环境和原始奖励配置 |
| `source/go2_demo/go2_demo/tasks/manager_based/go2_demo/go2_demo_rough.py` | Rough、混合地形和楼梯专项课程 |
| `source/go2_demo/go2_demo/tasks/manager_based/go2_demo/agents/rsl_rl_ppo_cfg.py` | PPO、迁移、镜像训练配置 |
| `source/go2_demo/go2_demo/tasks/manager_based/go2_demo/mdp/symmetry.py` | Go2 左右镜像变换 |
| `scripts/rsl_rl/expand_height_checkpoint.py` | 48维 checkpoint 无损扩展到235维 |
| `scripts/rsl_rl/train.py` | 训练入口，支持 `--load_policy_only` |
| `scripts/rsl_rl/play.py` | checkpoint 播放和模型导出 |
| `source/go2_demo/go2_demo/tasks/manager_based/go2_demo/aer_env.py` | AER 实验实现，最终迁移任务未使用 |

---

## 6. 重要 checkpoint

### 原始平地模型

```text
logs/rsl_rl/go2_velocity/2026-09-28_15-19-37/model_2999.pt
```

### 高度扫描零初始化迁移模型

```text
logs/rsl_rl/go2_velocity_flat_height_transfer/model_2999_height_init.pt
```

### 混合复杂地形最终模型

```text
logs/rsl_rl/go2_velocity_rough_height_transfer/
2026-09-30_07-58-17_full_terrain_1024_envs_5000/model_7498.pt
```

### 楼梯前进课程模型

```text
logs/rsl_rl/go2_velocity_stairs_height_transfer/
2026-10-01_00-32-09_stairs_forward_curriculum/model_9996.pt
```

### 镜像数据增强模型

```text
logs/rsl_rl/go2_velocity_stairs_height_transfer/
2026-10-01_01-31-53_stairs_symmetry/model_10995.pt
```

### Mirror Loss 与提速模型

```text
logs/rsl_rl/go2_velocity_stairs_height_transfer/
2026-10-01_10-13-28_stairs_mirror_loss_speed/model_11994.pt
```

### 当前最新侧移模型

```text
logs/rsl_rl/go2_velocity_stairs_height_transfer/
2026-10-01_10-58-57_stairs_lateral_speed/model_12493.pt
```

---

## 7. 最新模型播放

```bat
python scripts\rsl_rl\play.py --task Go2-Velocity-Stairs-Height-Transfer-Play-v0 --num_envs 16 --checkpoint "D:\RobotProject\go2_demo\logs\rsl_rl\go2_velocity_stairs_height_transfer\2026-10-01_10-58-57_stairs_lateral_speed\model_12493.pt"
```

---

## 8. 后续建议

1. 使用最新模型分别测试上楼和下楼，不只观察总体成功率。
2. 检查左腿是否仍存在明显跛行；如果存在，可将 `mirror_loss_coeff` 从 `0.1` 小幅提高到 `0.15`，不要一次提高过多。
3. 逐步将横向速度从 `±0.15` 提升到 `±0.2`，每次改变后短训并观察触地率。
4. 不要立刻把台阶上限提高到 15 cm；先确保 8～10 cm 台阶稳定。
5. 继续保留关键 checkpoint，不要只保存和评估最后一个模型。
6. 真机部署前应加入传感器噪声、延迟、摩擦和质量随机化，并确认高度扫描可以由真实传感器提供。


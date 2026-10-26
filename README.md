# 四足 Isaac Lab 仿真

基于 Isaac Lab 的 Unitree Go2 四足强化学习项目：平地行走、高度扫描迁移、复杂地形与楼梯训练、28cm上台与站姿约束，以及 Isaac Lab → MuJoCo Sim2Sim 部署。

## 演示视频

[八段独立视频目录](video/all_categories_20261011/README.md) · [完整合辑（约108秒）](video/all_categories_20261011/00_all_categories_12493_and_19491.mp4)

以下八段视频直接在首页播放，无需展开。标题标明地形类别与使用的策略权重；第8段展示一次完整成功的28cm下台回合。原始文件和合辑下载链接保留在上方。

### 1. 平地移动 · model_12493

https://github.com/user-attachments/assets/d554e843-0586-4d47-81d6-71048c7055dd

### 2. 密集碎石路 · 600块 / 4–8cm · model_12493

2026-10-26重录：相同区域的石块数从96增加到600，采用分层随机分布，减少空白通道，并提高石块起伏；其余六段保持原样。仍使用原权重，没有重新训练。合辑和下载文件也已同步更新。

https://github.com/user-attachments/assets/a75a9b4d-63e1-4cbc-bc95-9f780ff0a527

### 3. 上15°坡 · model_12493

2026-10-27重录：坡角从10°提高到15°，水平长度4m、高差约1.07m，录像约18秒，展示完整上坡及到达平台。16组并行测试全部达到本次判据，录像过程中无重置。

https://github.com/user-attachments/assets/1f321702-c768-4e57-8ac2-3b75646c32e5

### 4. 下15°坡 · model_12493

同样使用15°坡面和约18秒完整录像；下坡16组并行测试全部达到本次判据，录像过程中无重置。[测试条件与逐组结果](video/all_categories_20261011/SLOPE_EVALUATION_20261027.json)。判据为越过坡面后再稳定前进1秒、无重置、倾角小于45°，侧偏保持在坡面中间半幅。这不是最大爬坡能力测定或任意场景成功率保证。

https://github.com/user-attachments/assets/3751591c-b78d-45d3-b4a9-f719dd5a29f0

### 5. 上8cm楼梯 · model_12493

https://github.com/user-attachments/assets/2f96d26f-5c66-4bea-b098-6287053baa80

### 6. 下8cm楼梯 · model_12493

https://github.com/user-attachments/assets/389eb453-3660-4668-9730-cbb5b0c7767b

### 7. 上28cm单级台阶 · model_19491

https://github.com/user-attachments/assets/c40ba55b-c742-40bc-9cad-f23a78c25fcd

### 8. 下28cm单级台阶 · 单次成功演示 · model_12493

本段使用12493、种子46、0.5m/s，完整记录真实下28cm台阶并继续行走，整段约12秒没有倒地或重置。录像达到既定通行判据，未重新训练或剪接动作；这里只展示一次成功回合，不代表统计成功率。[录像参数与验证记录](video/all_categories_20261011/STEP_DOWN28_EVALUATION_20261027.json)。

https://github.com/user-attachments/assets/1e8dba26-268c-402b-ba3f-1321b7e33e28

| 场景 | 策略 |
|---|---|
| 平地、碎石路、上下15°坡、上下8cm楼梯 | model_12493.pt |
| 上28cm单级台阶 | model_19491.pt |
| 下28cm单级台阶（单次演示） | model_12493.pt |

视频是固定条件下的单次仿真演示，不是统计成功率，也不是同一个策略通过所有地形。碎石为固定小石块；8cm楼梯共4级。本仓库不包含训练权重、训练日志或本机环境。

## 训练流程与心得

通用行走训练路线：48维平地策略 → 扩展到235维高度扫描 → 微起伏适应 → 混合地形课程 → 楼梯专项 → 镜像增强与Mirror Loss → 前进及侧移提速。之后从通用行走策略开展独立的28cm上台与站姿训练。

1. **先保住有效步态，再增加感知和难度。** 加高度扫描不是只多一个标量，而是新增187维输入。保持原48维观测顺序和归一化一致，用旧权重初始化、新增输入权重置零，再逐步微调；同时检查Actor、Critic及归一化参数的维度。
2. **奖励先做可解释的基线。** 本项目普通加法奖励效果更容易稳定复现。AER实验中曾出现不走、倒地或奖励尺度过低；更复杂的奖励形式并不自动带来更好的行为。
3. **课程和指令要配套。** 从低起伏、低台阶开始。过多接近零的速度指令可能让机器人走不够距离、课程无法升级；加大指令前也要检查真实速度跟踪，而不是只看机器人没倒。
4. **环境数量是采样手段，不是万能修复。** 增至1024个环境在本次实验中改善了地形覆盖与曲线稳定性，但不能修复错误的观测、奖励或课程条件。
5. **复杂地形需要分类评估。** 混合地形总奖励不错，不代表上下楼梯都能通过；因此增加楼梯专项，并分别观察上楼、下楼、坡面和碎石。
6. **镜像需要完整的物理变换。** 不仅交换左右腿，还要处理髋关节符号、横向速度、角速度、指令、高度图和动作。镜像增强与Mirror Loss用于缓解左右跛行，不应强制所有过障碍动作完全对称。
7. **上台成功不等于站姿合格。** 28cm专项出现过三脚静止、后腿悬空和内八。站姿约束应在上台完成、低速或静止阶段生效，减少三脚静止仍获得的奖励；不要惩罚正常迈步时的短暂三脚支撑。
8. **不要默认最新权重最好。** 同时看速度误差、触地终止、回合长度、地形等级和确定性Play。本次通用地形采用12493，上28cm采用19491，避免把专项能力当成通用能力。
9. **Sim2Sim先对齐接口，再调策略。** 检查关节顺序和符号、速度参考系、last_action时序、动作缩放、PD参数及力矩限制。曾通过修复部署接口改善MuJoCo表现；不要一遇到失败就认定必须重新训练。
10. **教师学生是下一阶段，不是关闭高度扫描。** 先评估并冻结带高度信息的教师，再训练只读部署可得本体观测与历史的GRU学生；分别配置观测与归一化。低模仿损失不代表无高度信息也能通过全部地形。
11. **上下台阶需要分别验证。** 下台应单独检查落地姿态、冲击、终止和完整通行；一次成功展示与统计成功率是两回事，不能仅凭上台成功或8cm楼梯表现推断28cm下台能力。

### 教师—学生网络（后续方案，尚未训练）

拟用当前235维高度感知策略作教师，让学生通过GRU利用本体反馈历史，学习教师的12维动作。原48维基础观测含仿真真值线速度：真机学生需要状态估计器提供该速度，或移除它并重新定义45维输入。GRU能学习历史与接触后的反馈，但不能凭空提前看见未知台阶，28cm高台和下台能力必须重新验证。

现有八个视频仍是带高度扫描的PPO策略，没有完成学生蒸馏。PPO的Actor/Critic也不等于教师/学生。后续需新增观测分组、循环学生、蒸馏训练、隐藏状态重置及Sim2Sim适配；详见[训练总结第9节：教师—学生网络](TRAINING_SUMMARY.md)。算法思路参见[RSL-RL官方蒸馏说明](https://leggedrobotics.github.io/rsl_rl/guide/overview.html)。

完整记录：[通用行走训练总结与历史指标](TRAINING_SUMMARY.md) · [28cm上台训练](HIGH_STEP_28CM.md) · [站姿问题与约束](HIGH_STEP_STAND.md) · [Sim2Sim接口与验证](sim2sim_mujoco/README.md)。历史指标对应各阶段记录，不能视为当前八个视频的统计成功率。

## 目录

- `source/go2_demo/`：机器人配置、任务注册、观测、奖励、地形、课程与PPO配置。
- `scripts/rsl_rl/`：训练、播放、评估、高度观测权重扩展和录像脚本。
- `sim2sim_mujoco/`：MuJoCo部署、方向键控制、测试场景与接口测试。
- `video/all_categories_20261011/`：八段独立录像、合辑和分类说明。
- `TRAINING_SUMMARY.md`：训练过程、问题与心得。
- `HIGH_STEP_28CM.md`、`HIGH_STEP_STAND.md`：28cm台阶与站姿训练说明。

## 环境与安装

原项目在 Windows、Python 3.11、Isaac Sim 5.1.0 和 Isaac Lab 环境中运行。请先安装并配置 Isaac Lab，再使用该环境的Python：

```bash
python -m pip install -e source/go2_demo
python scripts/list_envs.py
```

Isaac机器人USD资产不随本仓库发布。请从 Unitree 的官方模型仓库获取：

```bash
git clone https://huggingface.co/unitreerobotics/unitree_model source/go2_demo/go2_demo/tasks/manager_based/go2_demo/assets/robot/unitree_model
```

确保 Git LFS 下载了真实USD文件，而非指针。所需入口为 `Go2/usd/go2.usd` 及其相对引用文件。也可通过环境变量 `UNITREE_MODEL_DIR` 指定已经下载的模型仓库路径。请遵守上游资产的许可条款。

## 训练与播放

平地高度扫描训练示例：

```bash
python scripts/rsl_rl/train.py --task Go2-Velocity-Flat-Height-Transfer-v0 --headless --num_envs 128
```

楼梯训练示例（需要自行提供兼容235维观测的checkpoint）：

```bash
python scripts/rsl_rl/train.py --task Go2-Velocity-Stairs-Height-Transfer-v0 --headless --num_envs 1024 --resume --checkpoint "PATH/TO/model.pt" --max_iterations 5000
```

播放28cm台阶：

```bash
python scripts/rsl_rl/play.py --task Go2-High-Step-28cm-Color-Play-v0 --num_envs 1 --checkpoint "PATH/TO/model.pt"
```

训练和播放时的观测维度、网络结构、归一化参数必须与checkpoint匹配。48维策略不能直接加载到235维环境；高度扩展流程见训练总结和 `scripts/rsl_rl/expand_height_checkpoint.py`。

## 录像与Sim2Sim

在 Windows 的项目目录中运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/rsl_rl/record_terrain_demos.ps1 -Checkpoint "PATH\TO\model.pt" -Python "PATH\TO\python.exe"
```

MuJoCo部署需自行提供导出的TorchScript策略；参见 [Sim2Sim说明](sim2sim_mujoco/README.md)：

```bash
python -m pip install -r sim2sim_mujoco/requirements.txt
python sim2sim_mujoco/deploy.py --scene playground --keyboard --policy "PATH/TO/policy.pt"
```

## 注意事项

项目仍在实验中。三脚静止、站姿、速度跟踪和不同仿真器的稳定性需要分别评估，不应仅以累计奖励判断性能。不要直接部署到真实机器人。

部分历史说明保留了原训练目录；复现时请替换为自己的路径。上游源码版权声明保留，MuJoCo Go2模型及其许可证位于 `sim2sim_mujoco/unitree_mujoco/`。本发布未替作者新增或变更项目许可。

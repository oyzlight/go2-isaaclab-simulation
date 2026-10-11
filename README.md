# 四足 Isaac Lab 仿真

基于 Isaac Lab 的 Unitree Go2 四足强化学习项目：平地行走、高度扫描迁移、复杂地形与楼梯训练、28cm上台与站姿约束，以及 Isaac Lab → MuJoCo Sim2Sim 部署。

## 演示视频

[七段独立视频目录](video/all_categories_20261011/README.md) · [完整合辑（约84秒）](video/all_categories_20261011/00_all_categories_12493_and_19491.mp4)

### 七类地形完整合辑

点击下方播放器播放；原始 MP4 下载链接保留在上方。

https://github.com/user-attachments/assets/9d5a960f-168a-42c4-a76b-e51768c4e702

<details>
<summary>展开七段独立视频（可在本页播放）</summary>

### 1. 平地移动 · model_12493

https://github.com/user-attachments/assets/d554e843-0586-4d47-81d6-71048c7055dd

### 2. 碎石路 · model_12493

https://github.com/user-attachments/assets/e919be32-9b13-4ffb-bb17-69ba9458a8b6

### 3. 上10°坡 · model_12493

https://github.com/user-attachments/assets/4cbb06e3-1b8d-467e-908c-c20f3a995980

### 4. 下10°坡 · model_12493

https://github.com/user-attachments/assets/69ca4bf7-00a4-4ee4-a380-24ee8554e7c5

### 5. 上8cm楼梯 · model_12493

https://github.com/user-attachments/assets/2f96d26f-5c66-4bea-b098-6287053baa80

### 6. 下8cm楼梯 · model_12493

https://github.com/user-attachments/assets/389eb453-3660-4668-9730-cbb5b0c7767b

### 7. 上28cm单级台阶 · model_19491

https://github.com/user-attachments/assets/c40ba55b-c742-40bc-9cad-f23a78c25fcd

</details>

| 场景 | 策略 |
|---|---|
| 平地、碎石路、上下10°坡、上下8cm楼梯 | model_12493.pt |
| 上28cm单级台阶 | model_19491.pt |

视频是固定条件下的单次仿真演示，不是统计成功率，也不是同一个策略通过所有地形。碎石为固定小石块；8cm楼梯共4级。本仓库不包含训练权重、训练日志或本机环境。

## 目录

- `source/go2_demo/`：机器人配置、任务注册、观测、奖励、地形、课程与PPO配置。
- `scripts/rsl_rl/`：训练、播放、评估、高度观测权重扩展和录像脚本。
- `sim2sim_mujoco/`：MuJoCo部署、方向键控制、测试场景与接口测试。
- `video/all_categories_20261011/`：七段独立录像、合辑和分类说明。
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

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from .assets.unitree import UNITREE_GO2_CFG


@configclass
class GO2RobotDemoEnv(LocomotionVelocityRoughEnvCfg):
    """Flat-terrain velocity-tracking environment for the Unitree Go2."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # 用本地 Go2 模型
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # -- Go2 专属调参（对应官方 Unitree Go2 配置）--
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        self.actions.joint_pos.scale = 0.25
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None
        # -- 奖励 --
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            ".*thigh",
            ".*calf",
        ]
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # --终止--
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"

        # -- 平地 --
        self.rewards.flat_orientation_l2.weight = -2.5
        # self.rewards.feet_air_time.weight = 0.25
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None


@configclass
class GO2RobotDemoEnv_PLAY(GO2RobotDemoEnv):
    """Flat-terrain evaluation configuration without observation corruption."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class GO2RobotDemoLegacyPlayEnvCfg(GO2RobotDemoEnv_PLAY):
    """Evaluation configuration for legacy 48-D checkpoints."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

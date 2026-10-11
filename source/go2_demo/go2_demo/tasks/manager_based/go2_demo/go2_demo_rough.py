from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from . import mdp
from .assets.unitree import UNITREE_GO2_CFG


@configclass
class GO2RobotRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Adaptive rough-terrain velocity task for the Unitree Go2."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Robot and observations.
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        # Stage 1 keeps exteroceptive noise disabled. Noise is introduced only
        # after the policy can traverse changing height maps reliably.
        self.observations.policy.height_scan.noise = None
        self.actions.joint_pos.scale = 0.25

        # Go2 is smaller than the robots used by the generic terrain preset.
        terrain_cfg = self.scene.terrain.terrain_generator
        if terrain_cfg is not None:
            terrain_cfg.curriculum = True
            terrain_cfg.sub_terrains["pyramid_stairs"].step_height_range = (0.03, 0.15)
            terrain_cfg.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.03, 0.15)
            terrain_cfg.sub_terrains["boxes"].grid_height_range = (0.025, 0.10)
            terrain_cfg.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
            terrain_cfg.sub_terrains["random_rough"].noise_step = 0.01
            terrain_cfg.sub_terrains["pyramid_stairs"].proportion = 0.2
            terrain_cfg.sub_terrains["pyramid_stairs_inv"].proportion = 0.2
            terrain_cfg.sub_terrains["boxes"].proportion = 0.2
            terrain_cfg.sub_terrains["random_rough"].proportion = 0.2
            terrain_cfg.sub_terrains["hf_pyramid_slope"].proportion = 0.1
            terrain_cfg.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.1

        # Start on the two easiest rows. The curriculum moves individual
        # environments up or down according to their locomotion performance.
        self.scene.terrain.max_init_terrain_level = 1

        # Domain randomization. Pushes are introduced only after terrain
        # locomotion is stable; terrain difficulty itself is already adaptive.
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # AER task rewards and auxiliary penalties.
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*thigh"
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.dof_pos_limits.weight = -0.2
        self.rewards.joint_deviation_l1 = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.01,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.energy_new_actual = RewTerm(
            func=mdp.energy_new_actual,
            weight=0.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "sigma_lin": 1000.0,
                "sigma_ang": 500.0,
                "clip_lin": 0.2,
                "clip_ang": 0.2,
            },
        )

        # Start rough-terrain transfer with the command range already mastered
        # on flat ground. It can be widened after terrain traversal stabilizes.
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class GO2RobotRoughEnvCfg_PLAY(GO2RobotRoughEnvCfg):
    """Smaller rough-terrain scene for visual evaluation."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None

        terrain_cfg = self.scene.terrain.terrain_generator
        if terrain_cfg is not None:
            terrain_cfg.num_rows = 5
            terrain_cfg.num_cols = 5
            terrain_cfg.curriculum = False

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class GO2RobotRoughHeightTransferEnvCfg(GO2RobotRoughEnvCfg):
    """Rough-terrain transfer task using the original additive reward set."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # Remove the experimental reward terms used by the AER trials.
        self.rewards.dof_pos_limits.weight = 0.0
        self.rewards.joint_deviation_l1 = None
        self.rewards.energy_new_actual = None


@configclass
class GO2RobotRoughHeightTransferEnvCfg_PLAY(GO2RobotRoughHeightTransferEnvCfg):
    """Deterministic evaluation configuration for height-transfer policies."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = 2

        terrain_cfg = self.scene.terrain.terrain_generator
        if terrain_cfg is not None:
            terrain_cfg.num_rows = 5
            terrain_cfg.num_cols = 5
            terrain_cfg.curriculum = False

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class GO2RobotStairsHeightTransferEnvCfg(GO2RobotRoughHeightTransferEnvCfg):
    """Stair-specialization curriculum initialized from the mixed-terrain policy."""

    def __post_init__(self) -> None:
        super().__post_init__()

        terrain_cfg = self.scene.terrain.terrain_generator
        if terrain_cfg is not None:
            terrain_cfg.sub_terrains["pyramid_stairs"].proportion = 0.5
            terrain_cfg.sub_terrains["pyramid_stairs_inv"].proportion = 0.5
            terrain_cfg.sub_terrains["boxes"].proportion = 0.0
            terrain_cfg.sub_terrains["random_rough"].proportion = 0.0
            terrain_cfg.sub_terrains["hf_pyramid_slope"].proportion = 0.0
            terrain_cfg.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.0
            terrain_cfg.sub_terrains["pyramid_stairs"].step_height_range = (0.02, 0.10)
            terrain_cfg.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.02, 0.10)

        self.scene.terrain.max_init_terrain_level = 1
        # Stair curriculum requires enough forward travel to cross half of an
        # 8 m terrain tile. Small symmetric commands can never promote levels.
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.15, 0.15)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.2, 0.2)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False
        self.rewards.feet_air_time.weight = 0.05
        self.rewards.flat_orientation_l2.weight = -0.5


@configclass
class GO2RobotStairsHeightTransferEnvCfg_PLAY(GO2RobotStairsHeightTransferEnvCfg):
    """Deterministic stair-only evaluation configuration."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = 3

        terrain_cfg = self.scene.terrain.terrain_generator
        if terrain_cfg is not None:
            terrain_cfg.num_rows = 5
            terrain_cfg.num_cols = 10
            terrain_cfg.curriculum = True

        # Freeze the spawn level during evaluation while preserving the
        # row-to-difficulty mapping in the generated stair grid.
        self.curriculum.terrain_levels = None

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class GO2RobotStairsHeightTransferDREnvCfg(GO2RobotStairsHeightTransferEnvCfg):
    """Stair env with dynamics domain randomization to close the MuJoCo sim2sim gap.

    The Unitree MuJoCo deployment model differs from the Isaac Lab USD model in
    PD damping, joint armature, joint Coulomb friction and contact friction. The
    baseline stair policy is trained without these terms and therefore collapses
    in MuJoCo. Randomizing them here (per-env, at startup) makes the policy robust
    to the exact dynamics it meets in :mod:`sim2sim_mujoco.deploy`.

    Randomized ranges are chosen to bracket the measured MuJoCo values:
      - PD damping: Isaac 0.5  ->  MuJoCo deploy 1.0 (+ MJCF joint damping 0.1)
      - joint armature: Isaac 0.0  ->  MJCF 0.01
      - joint friction: Isaac 0.0  ->  MJCF frictionloss 0.2
      - contact friction: Isaac 0.8/0.6  ->  MuJoCo ~0.6
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # 1) PD gains (DCMotor stiffness / damping). MuJoCo uses the same kp but
        #    roughly double the damping, so scale damping over a wide window.
        self.events.randomize_actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "stiffness_distribution_params": (0.9, 1.1),
                "damping_distribution_params": (0.5, 2.2),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

        # 2) Joint armature + Coulomb friction (covers MJCF armature / frictionloss).
        self.events.randomize_joint_parameters = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "armature_distribution_params": (0.0, 0.02),
                "friction_distribution_params": (0.0, 0.3),
                "operation": "abs",
                "distribution": "uniform",
            },
        )

        # 3) Widen body friction (base class fixes it at 0.8 / 0.6).
        self.events.physics_material.params["static_friction_range"] = (0.5, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.4, 0.8)

        # 4) Re-enable base COM randomization (disabled in the rough/stair lineage).
        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.02, 0.02)},
            },
        )

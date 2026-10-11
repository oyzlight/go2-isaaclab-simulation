"""Checkpoint-compatible training for a single 28 cm vertical ledge."""

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass

from . import mdp
from .go2_demo_rough import GO2RobotStairsHeightTransferEnvCfg
from .high_step_geometry import single_step_terrain


@configclass
class SingleStepTerrainCfg(SubTerrainBaseCfg):
    function = single_step_terrain
    height_levels: tuple[float, ...] = (
        0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28
    )
    approach_length: float = 2.0
    ledge_distance: float = 0.9


@configclass
class GO2RobotHighStepEnvCfg(GO2RobotStairsHeightTransferEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1024
        self.episode_length_s = 12.0
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(7.0, 4.0), border_width=5.0, num_rows=12, num_cols=16,
            curriculum=True, difficulty_range=(0.0, 1.0), use_cache=False,
            sub_terrains={"single_step": SingleStepTerrainCfg()},
        )
        self.scene.terrain.max_init_terrain_level = 2
        # Preserve 235 observation order, normalizers, 50 Hz control, .25 action
        # scale, PD gains, motor torque limits and the original additive manager.
        self.commands.base_velocity = UniformVelocityCommandCfg(
            class_type=mdp.HighStepVelocityCommand, asset_name="robot",
            resampling_time_range=(12.0, 12.0), heading_command=False,
            rel_standing_envs=0.0, rel_heading_envs=0.0, debug_vis=False,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(0.35, 0.55), lin_vel_y=(-0.15, 0.15), ang_vel_z=(-0.4, 0.4),
            ),
        )
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.15, 0.10), "y": (-0.12, 0.12), "yaw": (-0.08, 0.08)
        }
        self.events.reset_high_step = EventTerm(func=mdp.reset_high_step_state, mode="reset")
        self.events.base_external_force_torque = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.5, 0.5)

        # Climbing needs upward base motion and pitch; retain roll and joint
        # protection without forcing a flat walking posture onto a vertical ledge.
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.ang_vel_xy_l2.weight = -0.02
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.dof_pos_limits.weight = -0.2
        self.rewards.climb_roll = RewTerm(func=mdp.high_step_roll, weight=-1.0)
        self.rewards.climb_progress = RewTerm(func=mdp.high_step_progress, weight=4.0)
        self.rewards.climb_foot_landing = RewTerm(func=mdp.high_step_foot_landing, weight=1.0)
        self.rewards.climb_complete = RewTerm(func=mdp.high_step_completion, weight=15.0)
        self.rewards.climb_failure = RewTerm(func=mdp.high_step_failure, weight=-5.0)
        self.terminations.step_success = DoneTerm(func=mdp.high_step_success)
        self.terminations.out_of_lane = DoneTerm(func=mdp.high_step_out_of_lane)
        self.curriculum.terrain_levels = CurrTerm(func=mdp.high_step_curriculum)


@configclass
class GO2RobotHighStepEnvCfg_PLAY(GO2RobotHighStepEnvCfg):
    """Fixed target height, never a random mixture of low/high training rows."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        terrain = self.scene.terrain.terrain_generator
        terrain.num_rows = 1
        terrain.num_cols = 4
        terrain.sub_terrains["single_step"].height_levels = (0.28,)
        self.scene.terrain.max_init_terrain_level = 0
        self.curriculum.terrain_levels.params = {"promote": False}
        self.observations.policy.enable_corruption = False


@configclass
class GO2RobotHighStepStandEnvCfg(GO2RobotHighStepEnvCfg):
    """Fixed 28 cm climb + normal static stance; old climb tasks are unchanged."""

    def __post_init__(self):
        super().__post_init__()
        terrain = self.scene.terrain.terrain_generator
        terrain.num_rows = 1
        terrain.sub_terrains["single_step"].height_levels = (0.28,)
        self.scene.terrain.max_init_terrain_level = 0
        self.episode_length_s = 16.0
        self.commands.base_velocity.class_type = mdp.HighStepStandVelocityCommand
        self.events.reset_base.func = mdp.reset_high_step_stand_base
        self.events.reset_base.params["platform_probability"] = 0.25
        self.events.reset_high_step.func = mdp.reset_high_step_stand_state
        self.terminations.step_success = DoneTerm(
            func=mdp.high_step_stand_success,
            params={
                "hold_time": 2.0, "hip_tolerance": 0.12, "leg_tolerance": 0.35,
                "foot_tolerance": 0.05, "height_tolerance": 0.05,
                "linear_speed_max": 0.08, "angular_speed_max": 0.20,
                "tilt_max_deg": 10.0, "min_support_force": 5.0,
            },
        )
        self.curriculum.terrain_levels = CurrTerm(func=mdp.high_step_stand_curriculum)
        # Only the Stand task changes. Before settling, these have exactly the
        # original tracking values; four-foot settled support also retains them.
        self.rewards.track_lin_vel_xy_exp.func = mdp.stand_track_lin_vel_xy_exp
        self.rewards.track_lin_vel_xy_exp.params["unsupported_scale"] = 0.1
        self.rewards.track_ang_vel_z_exp.func = mdp.stand_track_ang_vel_z_exp
        self.rewards.track_ang_vel_z_exp.params["unsupported_scale"] = 0.1
        self.rewards.climb_complete.weight = 20.0
        self.rewards.stand_support = RewTerm(func=mdp.stand_support_reward, weight=2.0)
        self.rewards.stand_pose = RewTerm(func=mdp.stand_pose_penalty, weight=-3.0)
        self.rewards.stand_feet = RewTerm(func=mdp.stand_foot_placement_penalty, weight=-30.0)
        self.rewards.stand_orientation = RewTerm(func=mdp.stand_orientation_penalty, weight=-3.0)
        self.rewards.stand_height = RewTerm(func=mdp.stand_height_penalty, weight=-10.0)
        self.rewards.stand_motion = RewTerm(func=mdp.stand_motion_penalty, weight=-2.0)
        self.rewards.stand_quality = RewTerm(func=mdp.stand_quality_reward, weight=2.0)


@configclass
class GO2RobotHighStepStandEnvCfg_PLAY(GO2RobotHighStepStandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.terrain.terrain_generator.num_cols = 4
        # Evaluation must climb from below; never count a platform-start reset.
        self.events.reset_base.params["platform_probability"] = 0.0
        self.observations.policy.enable_corruption = False
        self.viewer.origin_type = "env"
        self.viewer.eye = (3.0, -3.0, 2.0)
        self.viewer.lookat = (1.0, 0.0, 0.35)
        # PLAY-only matte light-blue appearance; do not edit the source USD,
        # collision material, robot dynamics, or training appearance.
        self.scene.robot.spawn.visual_material_path = "clear_play_material"
        self.scene.robot.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.35, 0.60, 0.85),
            roughness=0.8,
            metallic=0.0,
            opacity=1.0,
        )
        self.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.9, 0.9, 0.9), roughness=1.0, metallic=0.0, opacity=1.0
        )
        self.viewer.resolution = (960, 540)

"""Fixed one-robot recording scenes. Does not modify any training task."""

import isaaclab.sim as sim_utils
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass

from .demo_terrain_geometry import demo_terrain
from .go2_demo_rough import GO2RobotRoughHeightTransferEnvCfg


@configclass
class DemoTerrainCfg(SubTerrainBaseCfg):
    function = demo_terrain
    kind: str = "flat"
    gravel_seed: int = 42
    rock_count: int = 600
    rock_height_range: tuple[float, float] = (0.04, 0.08)
    rock_radius_x_range: tuple[float, float] = (0.08, 0.16)
    rock_radius_y_range: tuple[float, float] = (0.07, 0.14)
    slope_degrees: float = 15.0
    ramp_length: float = 4.0
    step_height: float = 0.08
    step_depth: float = 0.45
    step_count: int = 4


@configclass
class GO2TerrainVideoEnvCfg(GO2RobotRoughHeightTransferEnvCfg):
    demo_kind: str = "flat"

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 20.0
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(12.0, 4.0), border_width=3.0, num_rows=1, num_cols=1,
            curriculum=False, use_cache=False,
            sub_terrains={"demo": DemoTerrainCfg(kind=self.demo_kind)},
        )
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.visual_material = None  # vertex colours, no tile textures
        self.curriculum.terrain_levels = None
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0), "y": (-0.03, 0.03), "yaw": (0.0, 0.0)
        }
        command = self.commands.base_velocity
        command.ranges.lin_vel_x = (0.5, 0.5)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (-0.4, 0.4)
        command.heading_command = True
        command.heading_control_stiffness = 1.5
        command.rel_heading_envs = 1.0
        command.rel_standing_envs = 0.0
        command.ranges.heading = (0.0, 0.0)
        command.resampling_time_range = (20.0, 20.0)
        command.debug_vis = False
        self.scene.robot.spawn.visual_material_path = "demo_robot_material"
        self.scene.robot.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.35, 0.60, 0.85), roughness=0.8, metallic=0.0, opacity=1.0
        )
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.eye = (2.5, -3.0, 1.8)
        self.viewer.lookat = (0.8, 0.0, 0.1)
        self.viewer.resolution = (1280, 720)
        self.sim.render.rendering_mode = "quality"
        self.sim.render.antialiasing_mode = "DLAA"
        self.sim.render.samples_per_pixel = 4


@configclass
class GO2FlatVideoEnvCfg(GO2TerrainVideoEnvCfg):
    demo_kind = "flat"


@configclass
class GO2RubbleVideoEnvCfg(GO2TerrainVideoEnvCfg):
    demo_kind = "rubble"


@configclass
class GO2UphillVideoEnvCfg(GO2TerrainVideoEnvCfg):
    demo_kind = "uphill"


@configclass
class GO2DownhillVideoEnvCfg(GO2TerrainVideoEnvCfg):
    demo_kind = "downhill"


@configclass
class GO2UpstairsVideoEnvCfg(GO2TerrainVideoEnvCfg):
    demo_kind = "stairs_up"


@configclass
class GO2DownstairsVideoEnvCfg(GO2TerrainVideoEnvCfg):
    demo_kind = "stairs_down"


@configclass
class GO2StepDown28cmVideoEnvCfg(GO2TerrainVideoEnvCfg):
    """One vertical 28 cm drop, not a flight of small stairs."""

    demo_kind = "stairs_down"

    def __post_init__(self):
        super().__post_init__()
        terrain = self.scene.terrain.terrain_generator.sub_terrains["demo"]
        terrain.step_count = 1
        terrain.step_height = 0.28

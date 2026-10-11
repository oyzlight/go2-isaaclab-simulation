"""High-quality coloured stair playback scenes, independent of the training tasks."""

import numpy as np
from pxr import Gf, UsdGeom, Vt

from isaaclab.managers import EventTermCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils import configclass

from .go2_demo_high_step import GO2RobotHighStepStandEnvCfg_PLAY, SingleStepTerrainCfg
from .high_step_geometry import single_step_terrain, two_step_terrain


@configclass
class TwoStepTerrainCfg(SingleStepTerrainCfg):
    function = two_step_terrain
    height_levels = (0.28,)
    step_count: int = 2
    step_depth: float = 0.70


@configclass
class ColoredSingleStepTerrainCfg(SingleStepTerrainCfg):
    function = single_step_terrain
    height_levels = (0.28,)
    step_count: int = 1


def color_two_step_terrain(env, env_ids=None):
    """Colour only visible faces; never add/move collision geometry."""
    stage = get_current_stage()
    cfg = env.cfg.scene.terrain.terrain_generator.sub_terrains["single_step"]
    riser_height = cfg.height_levels[0]
    coloured = 0
    for path in env.scene.terrain.terrain_prim_paths:
        mesh = UsdGeom.Mesh.Get(stage, f"{path}/mesh")
        if not mesh:
            raise RuntimeError(f"Expected terrain mesh at {path}/mesh")
        points = np.asarray(mesh.GetPointsAttr().Get())
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        if not np.all(counts == 3):
            raise RuntimeError("Two-step colouring expects triangular faces")
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        max_z = points[faces, 2].max(axis=1)
        colours = np.tile((0.90, 0.90, 0.90), (len(faces), 1))
        colours[max_z > 0.01] = (0.90, 0.38, 0.08)  # first tread/front: orange
        colours[max_z > riser_height + 0.01] = (1.0, 0.60, 0.15)  # second: lighter orange
        colour = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "displayColor", mesh.GetDisplayColorPrimvar().GetTypeName(), UsdGeom.Tokens.uniform
        )
        colour.Set(Vt.Vec3fArray([Gf.Vec3f(*values) for values in colours.tolist()]))
        colour.SetIndices(Vt.IntArray())
        opacity = mesh.GetDisplayOpacityPrimvar()
        opacity.SetInterpolation(UsdGeom.Tokens.uniform)
        opacity.Set(Vt.FloatArray([1.0] * len(faces)))
        opacity.SetIndices(Vt.IntArray())
        coloured += len(faces)
    count = getattr(cfg, "step_count", 1)
    print(f"[INFO]: Coloured stair scene: {count} riser(s), {riser_height:.2f} m each, {count*riser_height:.2f} m total; coloured {coloured} faces.")


@configclass
class GO2RobotTwoStepEnvCfg_PLAY(GO2RobotHighStepStandEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.terrain.terrain_generator.num_cols = 1
        self.scene.terrain.terrain_generator.sub_terrains = {"single_step": TwoStepTerrainCfg()}
        # No bound whole-mesh material: use per-face colours for white ground
        # and contrasting stairs. Physics material stays unchanged.
        self.scene.terrain.visual_material = None
        self.events.colour_stairs = EventTermCfg(func=color_two_step_terrain, mode="startup")
        self.episode_length_s = 24.0
        self.sim.render.rendering_mode = "quality"
        self.sim.render.antialiasing_mode = "DLAA"
        self.sim.render.dlss_mode = 2
        self.sim.render.samples_per_pixel = 4
        self.sim.render.enable_reflections = True
        self.sim.render.enable_global_illumination = True
        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_dl_denoiser = True
        self.viewer.resolution = (1280, 720)
        self.viewer.eye = (3.6, -3.2, 2.3)
        self.viewer.lookat = (1.4, 0.0, 0.45)


@configclass
class GO2RobotColoredSingleStepEnvCfg_PLAY(GO2RobotTwoStepEnvCfg_PLAY):
    """Same high-quality colours, but one 28 cm riser and the original goal."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.sub_terrains = {"single_step": ColoredSingleStepTerrainCfg()}
        self.episode_length_s = 16.0
        self.viewer.eye = (3.0, -3.0, 2.0)
        self.viewer.lookat = (1.0, 0.0, 0.35)

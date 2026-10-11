"""Deterministic straight corridors for recorded locomotion demonstrations."""

import numpy as np
import trimesh


def demo_terrain(difficulty, cfg):
    length, width = cfg.size
    origin_x, edge = 1.5, 2.5
    if length < 10 or width < 3:
        raise ValueError("Demo corridors need size >= (10, 3) metres")
    floor_colour = (230, 230, 230, 255)
    stair_colour = (230, 125, 35, 255)
    ramp_colour = (100, 160, 110, 255)

    def colour(mesh, rgba):
        mesh.visual.vertex_colors = np.tile(rgba, (len(mesh.vertices), 1)).astype(np.uint8)
        return mesh

    def box(start, end, top, rgba=floor_colour):
        mesh = trimesh.creation.box(
            (end - start, width, top + 0.15),
            transform=trimesh.transformations.translation_matrix(
                ((start + end) / 2, width / 2, (top - 0.15) / 2)
            ),
        )
        return colour(mesh, rgba)

    if cfg.kind == "flat":
        return [box(0, length, 0)], np.array((origin_x, width / 2, 0.0))
    if cfg.kind == "rubble":
        meshes = [box(0, length, 0)]
        rng = np.random.default_rng(cfg.gravel_seed)
        for _ in range(cfg.rock_count):
            # Fixed irregular rocks, not a simulated loose granular bed.
            height = rng.uniform(0.02, 0.06)
            mesh = trimesh.creation.icosphere(subdivisions=0, radius=1.0)
            mesh.apply_scale((rng.uniform(0.06, 0.13), rng.uniform(0.05, 0.11), height * 0.7))
            mesh.apply_transform(trimesh.transformations.rotation_matrix(rng.uniform(-np.pi, np.pi), (0, 0, 1)))
            mesh.apply_translation((rng.uniform(edge, 9.0), rng.uniform(width/2 - 1, width/2 + 1), height * 0.3))
            meshes.append(colour(mesh, (110, 125, 140, 255)))
        return meshes, np.array((origin_x, width / 2, 0.0))
    if cfg.kind in ("uphill", "downhill"):
        end = edge + cfg.ramp_length
        rise = np.tan(np.deg2rad(cfg.slope_degrees)) * cfg.ramp_length
        if not 0 < cfg.slope_degrees < 30 or end > length - 1:
            raise ValueError("Invalid demo ramp dimensions")
        start_z, end_z = (0.0, rise) if cfg.kind == "uphill" else (rise, 0.0)
        vertices = np.array([
            (edge, 0, -0.15), (end, 0, -0.15), (end, width, -0.15), (edge, width, -0.15),
            (edge, 0, start_z), (end, 0, end_z), (end, width, end_z), (edge, width, start_z),
        ])
        faces = np.array([
            (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
            (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
            (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
        ])
        ramp = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        ramp.fix_normals()
        return [box(0, edge, start_z), colour(ramp, ramp_colour), box(end, length, end_z)], np.array((origin_x, width/2, start_z))
    if cfg.kind in ("stairs_up", "stairs_down"):
        if cfg.step_count < 1 or cfg.step_height <= 0 or cfg.step_depth <= 0:
            raise ValueError("Invalid demo stairs")
        if edge + (cfg.step_count - 1) * cfg.step_depth > length - 1:
            raise ValueError("Demo stairs need a final platform")
        up = cfg.kind == "stairs_up"
        start_z = 0.0 if up else cfg.step_count * cfg.step_height
        meshes = [box(0, edge, start_z)]
        for index in range(cfg.step_count):
            start = edge + index * cfg.step_depth
            end = start + cfg.step_depth if index < cfg.step_count - 1 else length
            height = (index + 1 if up else cfg.step_count - index - 1) * cfg.step_height
            meshes.append(box(start, end, height, stair_colour))
        return meshes, np.array((origin_x, width/2, start_z))
    raise ValueError(f"Unknown demo terrain: {cfg.kind}")

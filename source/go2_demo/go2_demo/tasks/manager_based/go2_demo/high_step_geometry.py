"""Single vertical ledge geometry, independent of the simulator for testing."""

import numpy as np
import trimesh


def single_step_terrain(difficulty, cfg):
    """Lower approach floor + broad upper platform; origin is on the lower floor."""
    length, width = cfg.size
    heights = cfg.height_levels
    if not heights or any(height <= 0 for height in heights):
        raise ValueError("height_levels must contain positive heights in metres")
    if any(a > b for a, b in zip(heights, heights[1:])):
        raise ValueError("height_levels must be nondecreasing")
    origin_x = cfg.approach_length
    edge_x = origin_x + cfg.ledge_distance
    if not (0.5 < origin_x < edge_x < length - 1.5 and width >= 3.0):
        raise ValueError("Single-step tile needs an approach, a broad platform and width >= 3 m")
    level = min(max(int(difficulty * len(heights)), 0), len(heights) - 1)
    height = heights[level]

    def box(size, center):
        return trimesh.creation.box(
            size, transform=trimesh.transformations.translation_matrix(center)
        )

    # No overlapping coplanar top faces: the lower floor ends exactly at the ledge.
    lower = box((edge_x, width, 0.1), (edge_x / 2, width / 2, -0.05))
    upper = box(
        (length - edge_x, width, height + 0.1),
        ((length + edge_x) / 2, width / 2, (height - 0.1) / 2),
    )
    return [lower, upper], np.array((origin_x, width / 2, 0.0))


def two_step_terrain(difficulty, cfg):
    """Two real vertical risers; the first tread is step_depth metres long."""
    # Reuse the single-step validation and difficulty-to-riser-height mapping.
    meshes, origin = single_step_terrain(difficulty, cfg)
    if cfg.step_count != 2 or cfg.step_depth <= 0:
        raise ValueError("Two-step terrain requires step_count=2 and positive step_depth")
    length, width = cfg.size
    edge = cfg.approach_length + cfg.ledge_distance
    second_edge = edge + cfg.step_depth
    if second_edge > length - 1.5:
        raise ValueError("Two-step terrain needs at least 1.5 m of final platform")
    height = float(meshes[1].bounds[1, 2])

    def box(start, end, top):
        return trimesh.creation.box(
            (end - start, width, top + 0.1),
            transform=trimesh.transformations.translation_matrix(
                ((start + end) / 2, width / 2, (top - 0.1) / 2)
            ),
        )

    # Disjoint tops avoid coplanar flicker; the second riser is not a ramp.
    return [meshes[0], box(edge, second_edge, height), box(second_edge, length, 2 * height)], origin

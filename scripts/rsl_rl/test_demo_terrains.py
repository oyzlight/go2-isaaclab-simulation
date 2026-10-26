"""Check corridor heights, ramp directions, closed meshes and deterministic rocks."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

path = Path(__file__).resolve().parents[2] / "source/go2_demo/go2_demo/tasks/manager_based/go2_demo/demo_terrain_geometry.py"
spec = importlib.util.spec_from_file_location("demo_geometry_test", path)
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)


class DemoTerrainTests(unittest.TestCase):
    @staticmethod
    def cfg(kind):
        return SimpleNamespace(
            kind=kind, size=(12., 4.), gravel_seed=42, rock_count=600,
            rock_height_range=(.04, .08), rock_radius_x_range=(.08, .16),
            rock_radius_y_range=(.07, .14),
            slope_degrees=10., ramp_length=4., step_height=.08, step_depth=.45, step_count=4,
        )

    def test_all_corridors_have_closed_collision_meshes(self):
        for kind in ("flat", "rubble", "uphill", "downhill", "stairs_up", "stairs_down"):
            with self.subTest(kind=kind):
                meshes, origin = geometry.demo_terrain(.5, self.cfg(kind))
                self.assertEqual(origin[0], 1.5)
                for mesh in meshes:
                    self.assertTrue(mesh.is_watertight)
                    self.assertGreater(mesh.volume, 0)

    def test_stairs_exactly_eight_centimetres(self):
        for kind, expected in (("stairs_up", (0., .08, .16, .24, .32)),
                               ("stairs_down", (.32, .24, .16, .08, 0.))):
            meshes, origin = geometry.demo_terrain(.5, self.cfg(kind))
            np.testing.assert_allclose([m.bounds[1, 2] for m in meshes], expected)
            self.assertAlmostEqual(origin[2], expected[0])
            for a, b in zip(meshes, meshes[1:]):
                self.assertAlmostEqual(a.bounds[1, 0], b.bounds[0, 0])

    def test_ramp_slope_and_spawn_height(self):
        for degrees in (10, 15, 20):
            height = np.tan(np.deg2rad(degrees)) * 4
            for kind, expected_start in (("uphill", 0), ("downhill", height)):
                cfg = self.cfg(kind)
                cfg.slope_degrees = degrees
                meshes, origin = geometry.demo_terrain(.5, cfg)
                self.assertAlmostEqual(origin[2], expected_start)
                top = meshes[1].vertices[4:, 2]
                self.assertAlmostEqual(top[1] - top[0], height if kind == "uphill" else -height)

    def test_rocks_are_repeatable_and_in_height_range(self):
        a, _ = geometry.demo_terrain(.5, self.cfg("rubble"))
        b, _ = geometry.demo_terrain(.5, self.cfg("rubble"))
        self.assertEqual(len(a), 601)
        for x, y in zip(a[1:], b[1:]):
            np.testing.assert_allclose(x.vertices, y.vertices)
            self.assertGreaterEqual(x.bounds[1, 2], .04)
            self.assertLessEqual(x.bounds[1, 2], .08)
            self.assertLess(x.bounds[0, 2], 0.0)

    def test_dense_rubble_covers_the_route_and_leaves_spawn_clear(self):
        meshes, origin = geometry.demo_terrain(.5, self.cfg("rubble"))
        centres = np.array([mesh.bounds.mean(axis=0)[:2] for mesh in meshes[1:]])
        counts, _, _ = np.histogram2d(centres[:, 0], centres[:, 1],
                                      bins=(10, 5), range=((2.5, 9.0), (1.0, 3.0)))
        self.assertGreaterEqual(counts.min(), 4)
        self.assertTrue(all(mesh.bounds[0, 0] > origin[0] + .5 for mesh in meshes[1:]))

    def test_rubble_rejects_invalid_parameters(self):
        for key, value in (("rock_count", 0), ("rock_height_range", (.08, .04)),
                           ("rock_radius_x_range", (0., .1))):
            cfg = self.cfg("rubble")
            setattr(cfg, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                geometry.demo_terrain(.5, cfg)

    def test_single_step_down_has_one_exact_28cm_drop(self):
        cfg = self.cfg("stairs_down")
        cfg.step_count = 1
        cfg.step_height = .28
        meshes, origin = geometry.demo_terrain(.5, cfg)
        self.assertEqual(len(meshes), 2)
        self.assertAlmostEqual(origin[2], .28)
        self.assertAlmostEqual(meshes[0].bounds[1, 2], .28)
        self.assertAlmostEqual(meshes[1].bounds[1, 2], 0)
        self.assertAlmostEqual(meshes[0].bounds[1, 0], meshes[1].bounds[0, 0])


if __name__ == "__main__":
    unittest.main()

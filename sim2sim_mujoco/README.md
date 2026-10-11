# Go2 Isaac Lab → MuJoCo Sim2Sim

This runner executes the exported Isaac Lab TorchScript policy in Unitree's official Go2 MuJoCo model.

## Current validation status

Updated 2026-10-07: the selected **235-D height-aware policy** now walks on flat ground and traverses a complete
5 cm up/down stair course in MuJoCo, using its original training gains and action scale. No policy retraining was needed.

The previous conclusion that the policy had necessarily overfit PhysX was premature. The main faults were in the
deployment adapter: wrong velocity reference axes, and a last-action observation delayed by an extra policy interval.

| Evaluation | Duration | Observed result |
|---|---:|---|
| Flat ground, vx=0.5 | 20 s | 8.31 m displacement, no fall; actual body-frame vx about 0.417 m/s |
| Forward walking with vy=0.15 | 20 s | No fall; body-frame vy about 0.096 m/s |
| Forward walking with wz=0.15 | 20 s | No fall; yaw rate about 0.089 rad/s |
| 5 cm up/down stairs | 30 s | Crossed ascent, platform, and descent; x about 12.81 m, no fall |
| 8 cm up/down stairs | 30 s | Not cleared; stalled on ascent near x=1.93 m, without falling |

These are single rollouts from fixed initial conditions, not success-rate estimates. Arbitrary terrain, varying friction,
and real hardware have not been validated. Velocity tracking still has a measurable gap.

## Adapter fixes

- Read MuJoCo BODY COM velocity in world orientation, then rotate into the root **link** frame. BODY local velocity
  uses the principal inertia axes; Go2's inertial orientation is different from its link orientation.
- Feed the most recently executed raw action into `last_action`, removing the old extra 20 ms delay.
- Remove the arbitrary `[-5,5]` action clip, which the training configuration does not use.
- Refresh kinematics with `mj_forward` before inference so base and joint observations refer to the same instant.
- Match Isaac Lab's DCMotor torque-speed saturation (23.5 Nm, 30 rad/s) instead of constant torque limits only.
- Match disabled robot self-collisions and zero additional passive joint damping/friction/armature by default.

The interface regression suite checks velocity reference axes, action history, DC motor clipping, and height scanning.

```bat
python sim2sim_mujoco\test_interfaces.py
```

## Included interface

- Isaac policy observation order: 235 values.
- Explicit Isaac-to-MuJoCo joint and actuator mapping.
- 200 Hz MuJoCo physics and 50 Hz policy inference.
- Isaac joint-position action conversion: `q_target = q_default + 0.25 * action`.
- Torque PD controller: `kp=25`, `kd=0.5`, speed-dependent torque envelope up to 23.5 Nm.
- Yaw-aligned `17 x 11` height scan using MuJoCo rays.
- Immediate policy takeover after reset, matching the Isaac Lab environment.

## Policy

The default policy is:

```text
../logs/rsl_rl/go2_velocity_stairs_height_transfer/
2026-10-01_10-58-57_stairs_lateral_speed/exported/policy.pt
```

The exported policy contains the Isaac Lab observation normalizer. Do not normalize the 235-D observation again.
The runner detects both legacy 48-D policies and height-aware 235-D policies automatically.
The default scene is a custom obstacle-free plane. Unitree's stock `scene.xml` contains obstacles beginning near
`x=1.2 m` and should not be used for the first parity test.

## Headless smoke test

From `go2_demo`:

```bat
python sim2sim_mujoco\deploy.py --headless --duration 3 --vx 0.5
```

Validated legacy-policy flat test:

```bat
python sim2sim_mujoco\deploy.py --headless --duration 20 --vx 0.5 ^
  --policy "logs\rsl_rl\go2_velocity\2026-09-28_15-19-37\exported\policy.pt"
```

## Interactive viewer

### Arrow-key playground (Windows hold-to-move controls)

```bat
python sim2sim_mujoco\deploy.py --scene playground --keyboard
```

Click inside the MuJoCo viewer before controlling. On Windows hold keys to move, release to ramp smoothly to zero;
switching focus to another application immediately clears the command. The robot starts with a zero command.

| Key | Action |
|---|---|
| Up / Down | Forward / backward in the robot body frame |
| Left / Right | Move left / right sideways |
| Q / E | Turn left / right |
| Space | Immediately set the command to zero |
| R | Reset the robot and action history to the start pad |
| M | Switch between the overview map and following camera |
| + / - | Adjust forward/back speed by 0.1 m/s, bounded to 0.1--1.0 m/s |

Arrow keys and Q/E can be combined. Default command magnitudes are `vx=0.5 m/s`, `vy=0.15 m/s`, `wz=0.2 rad/s`.
These are policy commands, not direct joint controls. Actual speed may differ because the current policy has tracking
error. Use `--move-speed`, `--strafe-speed`, `--turn-speed` to configure keyboard magnitudes; `--vx/--vy/--wz` apply
only when keyboard control is off. On other platforms the MuJoCo press callback uses press-to-toggle controls because
it does not report key release; Space clears all latched commands. No extra keyboard package is required.

The on-screen overlay shows active commands and focus state (supported by the installed MuJoCo viewer).

### Playground layout

The route goes along world +X; the infinite ground around it remains walkable.

| Region | World x range | Terrain |
|---|---|---|
| Blue start pad | about -1.4 to 1.4 m | Flat ground, robot spawns at (0,0) |
| Green ascent | 1.5--3.3 m | Six 5 cm risers, 30 cm treads, 4 m wide |
| Blue top platform | 3.3--5.1 m | Flat platform 30 cm above the floor |
| Purple descent | 5.1--6.6 m | Descending stairs back to ground level |
| Blue transition pad | 6.6--8.0 m | Flat ground |
| Orange/brown raised ground | 8.0--12.0 m | 1 cm pad, ridges and scattered blocks with tops 3--5 cm above the floor |
| Blue finish pad | about 12.0--14.2 m | Flat ground |

Two additional parallel routes are now available. Use M for an overview, and Q/E plus forward movement to reach
them from the start pad:

| Route | Location | Terrain |
|---|---|---|
| Teal ramp / rubble | y=+4.5 m (left of the robot at spawn) | About 5.7 degree up/down ramp to a 20 cm platform, then 2--4 cm scattered blocks |
| Purple narrow stairs | y=-4.5 m (right of the robot at spawn) | Four 3 cm risers, a platform, and descent; path is 1.2 m wide |

The added routes are test terrain, not a guarantee that every approach angle will be handled by the current policy.

The ridges/blocks are 2--4 cm above the raised pad. Ground color markings are visual-only; collision and height rays
only see the real terrain. The complete route was crossed without falling in a 35 s fixed-command test at `vx=0.5`:
see `results/playground_235.json`. This is a single route test, not a guarantee for every manually chosen path.

### Ground flicker fix

The original colored pad boxes had top surfaces at z=0, exactly coincident with the ground plane. This caused
z-fighting when the camera moved. Their top surfaces are now 3 mm above the plane, with contact masks disabled
and geom group 2 still excluded from height rays. Ground checker repetition was reduced from 12x12 to 2x2 to
reduce distant texture aliasing. The scene's far clipping range was shortened to improve depth precision. These
rendering changes do not add a physical step to the flat pads.

Run it automatically instead of using the keyboard:

```bat
python sim2sim_mujoco\deploy.py --scene playground --vx 0.5
```

Change only stair riser height (8 cm is still not validated):

```bat
python sim2sim_mujoco\deploy.py --scene playground --keyboard --step-height 0.05
```

### Fixed command viewer

```bat
python sim2sim_mujoco\deploy.py --vx 0.5 --vy 0.0 --wz 0.0
```

Close the MuJoCo viewer to stop.

5 cm up/down stair course:

```bat
python sim2sim_mujoco\deploy.py --scene stairs --step-height 0.05 --vx 0.5
```

The course has six 5 cm risers, 30 cm treads, a 1.8 m platform, and a descending section. Use `--duration 30`
to stop automatically. Higher risers may be inspected with `--step-height 0.08`, but that case is not currently passed.

Save quantitative rollout results:

```bat
python sim2sim_mujoco\deploy.py --headless --duration 20 --vx 0.5 --report sim2sim_mujoco\results\flat_235.json
python sim2sim_mujoco\deploy.py --headless --duration 30 --scene stairs --vx 0.5 --report sim2sim_mujoco\results\stairs_5cm_235.json
```

Reports distinguish `first_fall_s` from `stairs_course_cleared`. No fall alone does not imply stair traversal success.
The diagnostic fall rule is root height below 0.15 m or root z axis tilted beyond 60 degrees, not Isaac's contact-based
termination rule. Reports also contain displacement, body velocities, and posture metrics.

## Select a different scene or policy

```bat
python sim2sim_mujoco\deploy.py ^
  --model "sim2sim_mujoco\unitree_mujoco\unitree_robots\go2\scene_terrain.xml" ^
  --policy "PATH\TO\exported\policy.pt" ^
  --vx 0.5
```

## Important parity checks

1. Verify every single joint moves in the same direction in Isaac Lab and MuJoCo.
2. Compare the 48 proprioceptive observations in a static default pose.
3. Compare action, target joint position, and PD torque logs.
4. Validate flat ground before testing stairs or height fields.
5. If the robot oscillates, check joint sign/order and PD parameters before tuning the policy.

The default `--joint-dynamics isaac` removes additional passive joint damping/friction/armature to match the local
Isaac asset and applies explicit PD on the torque actuators. `--joint-dynamics official` retains the original Unitree
passive parameters for comparison. Terrain/robot collisions are retained, robot self-collisions are disabled.

Official resources are from https://github.com/unitreerobotics/unitree_mujoco at commit
`1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d`. Keep the repository LICENSE. The adjacent `scene_policy_flat.xml` and
`scene_policy_stairs.xml` and `scene_policy_playground.xml` files are this project's test scenes, not upstream scenes.

Keyboard sequence checks:

```bat
python sim2sim_mujoco\test_keyboard_control.py
```

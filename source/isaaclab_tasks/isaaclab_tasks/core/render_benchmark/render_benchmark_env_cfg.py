# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG


@configclass
class _RenderBenchmarkPhysicsCfg(PresetCfg):
    """Physics backend presets — pick via ``presets=newton_mjwarp`` (default)
    or ``presets=physx`` (requires Isaac Sim)."""

    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(solver="newton", integrator="implicitfast", njmax=200, nconmax=70),
        num_substeps=2,
        bvh_constructor_geometry=os.getenv("NEWTON_BVH_GEOMETRY", "cubql"),
        bvh_constructor_gaussian=os.getenv("NEWTON_BVH_GAUSSIAN", "cubql"),
        bvh_constructor_scene=os.getenv("NEWTON_BVH_SCENE", "sah"),
        use_cuda_graph=os.getenv("NEWTON_USE_CUDA_GRAPH", "0") == "1",
    )
    physx: PhysxCfg = PhysxCfg()
    default = newton_mjwarp


def _perspective_camera(
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float],
    focal_length: float = 18.0,
    width: int = 256,
    height: int = 256,
    warp_enable_shadows: bool = False,
) -> CameraCfg:
    """Build a perspective camera with explicit quaternion orientation.

    ``rot`` is ``(qx, qy, qz, qw)`` in ``convention="world"`` (+X camera-forward).
    Pick ``pos`` and ``rot`` to aim the camera at the workspace from a 3/4 angle."""
    return CameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 50.0),
        ),
        width=width,
        height=height,
        renderer_cfg=MultiBackendRendererCfg(
            newton_renderer=NewtonWarpRendererCfg(enable_shadows=warp_enable_shadows),
        ),
    )


@configclass
class _RenderBenchmarkTiledCameraCfg(PresetCfg):
    default: CameraCfg | None = None
    rgb: CameraCfg | None = None
    albedo: CameraCfg | None = None
    depth: CameraCfg | None = None
    simple_shading_constant_diffuse: CameraCfg | None = None
    simple_shading_diffuse_mdl: CameraCfg | None = None
    simple_shading_full_mdl: CameraCfg | None = None


def _tiled_with_camera(camera_base: CameraCfg) -> _RenderBenchmarkTiledCameraCfg:
    return _RenderBenchmarkTiledCameraCfg(
        default=camera_base.replace(data_types=["rgb"]),
        rgb=camera_base.replace(data_types=["rgb"]),
        albedo=camera_base.replace(data_types=["albedo"]),
        depth=camera_base.replace(data_types=["depth"]),
        simple_shading_constant_diffuse=camera_base.replace(data_types=["simple_shading_constant_diffuse"]),
        simple_shading_diffuse_mdl=camera_base.replace(data_types=["simple_shading_diffuse_mdl"]),
        simple_shading_full_mdl=camera_base.replace(data_types=["simple_shading_full_mdl"]),
    )


@configclass
class RenderBenchmarkFrankaCabinetEnvCfg(DirectRLEnvCfg):
    """Franka Panda + Sektion cabinet (4 articulated joints: 2 drawers, 2
    doors). Sinusoidal animation drives all joints so the rendered frames
    show drawers sliding in/out and doors swinging — the cabinet adds
    significant articulated-geometry complexity vs. the existing variants.

    Poses match the canonical ``Isaac-Franka-Cabinet-Direct-v0`` task:
      Franka: pos=(1.0, 0, 0), rot=180° around Y (faces -X toward cabinet)
      Cabinet: pos=(0.0, 0, 0.4), rot=180° around Z (opens toward -X)
    """

    decimation: int = 2
    episode_length_s: float = 60.0

    action_space: int = 1
    observation_space: int = 1
    state_space: int = 0

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=2, physics=_RenderBenchmarkPhysicsCfg())

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=3.0, replicate_physics=True)

    # Front view with no yaw / no roll — pitch-only camera. Sits in front of
    # the cabinet (+X side, the drawer-facing side), elevated, pitched ~35°
    # down so the workspace reads as a slight bird's-eye. Tighter focal so
    # the Franka + cabinet fill most of the frame.
    #   pos        = (2.0, 0.0, 1.5)
    #   look-at    = (0.4, 0, 0.4)  →  forward = (-0.824, 0, -0.566)
    #   pitch      = ~34.5° down,  yaw = 0,  roll = 0
    # Quaternion (qxyzw) = (-0.296, 0, 0.955, 0) — qw=0, qx and qz only,
    # corresponds to a 180° rotation around the (-X, 0, +Z) axis.
    tiled_camera: _RenderBenchmarkTiledCameraCfg = _tiled_with_camera(
        _perspective_camera(
            pos=(2.0, 0.0, 1.5),
            rot=(-0.296, 0.0, 0.955, 0.0),
            focal_length=24.0,
            warp_enable_shadows=True,
            width=int(os.getenv("BENCHMARK_RENDER_RESOLUTION", "256")),
            height=int(os.getenv("BENCHMARK_RENDER_RESOLUTION", "256")),
        )
    )

    # Static, non-rigid USD references (e.g. a SeattleLabTable). Each entry's
    # spawn cfg is invoked at ``/World/envs/env_.*/<name>`` with the configured
    # init pose. These prims are visual-only — Newton sees them only via the
    # USD's intrinsic collision (if any), and they are not exposed as
    # RigidObject handles to the env code.
    static_assets: dict = {}
    terrain: TerrainImporterCfg | None = None
    # Override default ground-plane z. Lift-task scenes drop the floor below
    # the table's foot (table USD origin is at its surface).
    ground_plane_z: float = 0.0
    # When True, replace the USD GroundPlaneCfg (which Warp's renderer doesn't
    # see — it only knows about simulation meshes) with a large flat
    # CuboidCfg routed through the standard RigidObject path. Both OVRTX
    # and Warp render the same geometry, giving an apple-to-apple
    # comparison (without this, Warp's background is just the clear color,
    # so primary-ray-miss + shadow-ray costs differ).
    use_flat_ground: bool = False
    flat_ground_size: tuple = (50.0, 50.0)  # XY extent (m); single per-env cuboid
    flat_ground_thickness: float = 0.1
    flat_ground_color: tuple = (0.5, 0.5, 0.5)
    dome_light_intensity: float = 2000.0

    # USD DistantLight matching Warp's hardcoded directional
    # ``(-0.57735, 0.57735, -0.57735)``. Orientation is the quaternion that
    # rotates USD ``DistantLight``'s default ``-Z`` to that direction.
    light_cfg: sim_utils.LightCfg | None = sim_utils.DistantLightCfg(
        intensity=200.0,
        exposure=0.0,
        angle=0.0,
        color=(1.0, 1.0, 1.0),
        normalize=True,
    )
    light_orientation: tuple = (0.3251, 0.3251, 0.0, 0.8881)

    articulations: dict = {
        # Franka mounted at +X facing -X (toward cabinet). HighPD config so
        # the joints track sinusoidal commands smoothly.
        "robot": FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="/World/envs/env_.*/Robot",
            init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(
                pos=(1.0, 0.0, 0.0),
                rot=(0.0, 0.0, 1.0, 0.0),  # 180° around Y
            ),
        ),
        # Sektion cabinet: 4 articulated joints (door_left, door_right,
        # drawer_top, drawer_bottom). Loaded as ArticulationCfg so it clones
        # reliably to every env via the standard IsaacLab path.
        "cabinet": ArticulationCfg(
            prim_path="/World/envs/env_.*/Cabinet",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
                activate_contact_sensors=False,
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.4),
                rot=(0.0, 0.0, 0.0, 1.0),  # 180° around Z
                joint_pos={
                    "door_left_joint": 0.0,
                    "door_right_joint": 0.0,
                    "drawer_bottom_joint": 0.0,
                    "drawer_top_joint": 0.0,
                },
            ),
            actuators={
                "drawers": ImplicitActuatorCfg(
                    joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                    effort_limit_sim=87.0,
                    stiffness=10.0,
                    damping=1.0,
                ),
                "doors": ImplicitActuatorCfg(
                    joint_names_expr=["door_left_joint", "door_right_joint"],
                    effort_limit_sim=87.0,
                    stiffness=10.0,
                    damping=2.5,
                ),
            },
        ),
    }
    rigid_objects: dict = {}

    # Slightly slower / smaller-amplitude joint animation than the default so
    # the drawers and doors move at a clearly-visible-but-not-frantic rate.
    joint_animation_amplitude: float = 0.4
    joint_animation_freq_hz: float = 0.35

    # Use the flat-color CuboidCfg ground (visible in BOTH OVRTX and Warp)
    # instead of the USD GroundPlaneCfg (which only OVRTX sees). Necessary
    # for fair OVRTX-vs-Warp profile comparison — without this, Warp gets
    # extra primary-ray-misses + skipped shadow rays.
    use_flat_ground: bool = True

    write_image_to_file: bool = os.getenv("BENCHMARK_SAVE_IMAGE", "0") == "1"

    # IK pick controller (optional). When ``ik_articulation`` is non-empty the
    # named articulation is driven by a DifferentialIKController cycling
    # through ``ik_goals``; the sinusoidal animation is skipped for that
    # articulation but still applies to others (e.g. ANYmal in kitchen-sink).
    ik_articulation: str | None = None
    ik_target_body: str = "panda_hand"
    ik_joint_pattern: str = "panda_joint.*"
    # Each goal is [x, y, z, qx, qy, qz, qw] in the robot's root frame.
    ik_goals: list = []
    # Steps per goal (env-steps, not sim-steps); envs are offset by phase
    # so they don't all sit at the same waypoint at the same time.
    ik_cycle_steps: int = 40

    # Per-env color randomization for the simple PreviewSurface props. Each
    # listed prop name gets a fresh random RGB on every env at scene setup.
    # DexCube and other USD-loaded assets keep their baked colors.
    randomize_prop_colors: list = []
    # Per-env random initial rotation for the DexCube (so top-down views see
    # different colored faces per env). Set per-variant in subclasses.
    randomize_dexcube_rotation: bool = False

    # Lift-cube task hooks. When ``lift_cube_task=True`` and a rigid object
    # named ``dex_cube`` exists, the env samples a per-env target pose at init
    # time (in the configured ranges), writes the ``target_marker`` rigid
    # object to that pose every frame, substitutes the goal's XYZ with the
    # per-env target for IK goal indices listed in ``target_goal_indices``,
    # and teleports the DexCube to follow the gripper for goal indices in
    # ``grasped_goal_indices``.
    lift_cube_task: bool = False
    target_pose_range_x: tuple = (0.4, 0.6)
    target_pose_range_y: tuple = (-0.25, 0.25)
    target_pose_range_z: tuple = (0.25, 0.5)
    target_goal_indices: tuple = ()
    grasped_goal_indices: tuple = ()

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Env configs for the render-only benchmark tasks. Adds a TiledCamera with
``MultiBackendRendererCfg`` + the standard ``simple_shading_*`` preset
infrastructure so existing benchmark/visualize tooling drives these scenes
the same way it drives ShadowHand Vision.

Camera convention notes:
    All variants use ``convention="world"`` which is ``+X forward, +Z up``.
    The quaternion ``(0, 0.7071, 0, 0.7071)`` is a 90° rotation around +Y
    that sends camera-forward from +X to -Z, producing a strict top-down view.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils.configclass import configclass

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG, FRANKA_PANDA_HIGH_PD_CFG
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab_tasks.utils import PresetCfg
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg


# ----------------------------------------------------------------------
# Shared physics + camera preset infrastructure
# ----------------------------------------------------------------------


@configclass
class _RenderBenchPhysicsCfg(PresetCfg):
    """Physics backend presets — pick via ``presets=newton_mjwarp`` (default)
    or ``presets=physx`` (requires Isaac Sim)."""

    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(solver="newton", integrator="implicitfast", njmax=200, nconmax=70),
        num_substeps=2,
    )
    physx: PhysxCfg = PhysxCfg()
    default = newton_mjwarp


# Top-down quaternion: 90° around +Y → camera-forward +X→-Z (look straight down).
_TOP_DOWN_ROT = (0.0, 0.7071, 0.0, 0.7071)


def _top_down_camera(
    pos: tuple[float, float, float],
    focal_length: float = 15.0,
    width: int = 256,
    height: int = 256,
) -> CameraCfg:
    """Build a top-down camera at ``pos`` with the given focal length.
    Smaller focal_length = wider FOV. Default 256×256 per tile."""
    return CameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=CameraCfg.OffsetCfg(pos=pos, rot=_TOP_DOWN_ROT, convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 50.0),
        ),
        width=width,
        height=height,
        renderer_cfg=MultiBackendRendererCfg(),
    )


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


def _tiled_with_camera(camera_base: CameraCfg) -> "_RenderBenchTiledCameraCfg":
    """Wrap a CameraCfg in the data-type PresetCfg shell so all standard
    ``simple_shading_*`` / ``rgb`` / ``albedo`` / ``depth`` presets are available."""
    return _RenderBenchTiledCameraCfg(
        default=camera_base.replace(data_types=["rgb"]),
        rgb=camera_base.replace(data_types=["rgb"]),
        albedo=camera_base.replace(data_types=["albedo"]),
        depth=camera_base.replace(data_types=["depth"]),
        simple_shading_constant_diffuse=camera_base.replace(data_types=["simple_shading_constant_diffuse"]),
        simple_shading_diffuse_mdl=camera_base.replace(data_types=["simple_shading_diffuse_mdl"]),
        simple_shading_full_mdl=camera_base.replace(data_types=["simple_shading_full_mdl"]),
    )


# Default top-down camera (overridden per-variant for framing).
_BASE_CAMERA = _top_down_camera(pos=(0.5, 0.0, 3.5), focal_length=15.0)


@configclass
class _RenderBenchTiledCameraCfg(PresetCfg):
    """Data-type presets — pick via ``presets=simple_shading_full_mdl``, etc."""

    default: CameraCfg = _BASE_CAMERA.replace(data_types=["rgb"])
    rgb: CameraCfg = _BASE_CAMERA.replace(data_types=["rgb"])
    albedo: CameraCfg = _BASE_CAMERA.replace(data_types=["albedo"])
    depth: CameraCfg = _BASE_CAMERA.replace(data_types=["depth"])
    simple_shading_constant_diffuse: CameraCfg = _BASE_CAMERA.replace(
        data_types=["simple_shading_constant_diffuse"]
    )
    simple_shading_diffuse_mdl: CameraCfg = _BASE_CAMERA.replace(data_types=["simple_shading_diffuse_mdl"])
    simple_shading_full_mdl: CameraCfg = _BASE_CAMERA.replace(data_types=["simple_shading_full_mdl"])


# ----------------------------------------------------------------------
# Reusable prop helpers
# ----------------------------------------------------------------------


def _cube(name: str, pos, color=(0.7, 0.2, 0.2), size=(0.3, 0.3, 0.3), metallic=0.2) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=metallic),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


def _sphere(name: str, pos, color=(0.2, 0.7, 0.3), radius=0.2, metallic=0.4) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{name}",
        spawn=sim_utils.SphereCfg(
            radius=radius,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=metallic),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


def _kinematic_box(
    name: str,
    pos,
    size,
    color=(0.15, 0.15, 0.15),
    metallic: float = 0.3,
) -> RigidObjectCfg:
    """A static (kinematic) box — useful for pedestals/tables. Doesn't move
    under gravity or contact; we rely on Newton's kinematic flag to pin it."""
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=metallic),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, kinematic_enabled=True
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


def _cylinder(name: str, pos, color=(0.2, 0.3, 0.8), radius=0.2, height=0.4, metallic=0.6) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{name}",
        spawn=sim_utils.CylinderCfg(
            radius=radius,
            height=height,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=metallic),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


# ----------------------------------------------------------------------
# Per-env color-randomized props via MultiAssetSpawnerCfg.
# Each env picks a random variant from the assets_cfg list.
# Cloned envs use USD instancing → can't override material on each instance,
# so we pre-generate N colored variants and let the spawner randomize.
# ----------------------------------------------------------------------


def _hsv_colors(n: int, saturation: float = 0.7, value: float = 0.9) -> list:
    import colorsys
    return [colorsys.hsv_to_rgb(i / n, saturation, value) for i in range(n)]


def _multicolor_cube(name: str, pos, size=(0.08, 0.08, 0.08), n_variants: int = 8, metallic: float = 0.2) -> RigidObjectCfg:
    # MultiAssetSpawnerCfg requires `.*` in the leaf path so per-env distinct
    # asset paths can be generated (e.g. ``CubeB_0``, ``CubeB_1``, …).
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{name}_.*",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.CuboidCfg(
                    size=size,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c, metallic=metallic),
                )
                for c in _hsv_colors(n_variants)
            ],
            random_choice=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


def _multicolor_sphere(name: str, pos, radius=0.06, n_variants: int = 8, metallic: float = 0.6) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{name}_.*",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.SphereCfg(
                    radius=radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c, metallic=metallic),
                )
                for c in _hsv_colors(n_variants)
            ],
            random_choice=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


def _multicolor_cylinder(name: str, pos, radius=0.05, height=0.14, n_variants: int = 8, metallic: float = 0.5) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{name}_.*",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.CylinderCfg(
                    radius=radius,
                    height=height,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c, metallic=metallic),
                )
                for c in _hsv_colors(n_variants)
            ],
            random_choice=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


# ----------------------------------------------------------------------
# Base env cfg shared by all variants.
# ----------------------------------------------------------------------


@configclass
class RenderBenchBaseEnvCfg(DirectRLEnvCfg):
    decimation: int = 2
    episode_length_s: float = 60.0

    action_space: int = 1
    observation_space: int = 1
    state_space: int = 0

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=2, physics=_RenderBenchPhysicsCfg())
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=4.0, replicate_physics=True)
    tiled_camera: _RenderBenchTiledCameraCfg = _RenderBenchTiledCameraCfg()

    articulations: dict = {}
    rigid_objects: dict = {}
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
    # Optional override: if set, use this light cfg instead of the default
    # DomeLight. Useful for DistantLight (directional sun-like) scenes.
    light_cfg: sim_utils.LightCfg | None = None
    light_orientation: tuple = (0.0, 0.0, 0.0, 1.0)

    write_image_to_file: bool = False

    # Per-joint sinusoidal animation amplitude (radians). 0 = no animation;
    # each articulation's joints oscillate around their default pose with a
    # random per-(env, joint) phase, clamped to the joint's soft limits.
    joint_animation_amplitude: float = 0.6
    joint_animation_freq_hz: float = 0.5

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


# ----------------------------------------------------------------------
# Variant 1: Anymal-C quadruped + 3 small props.
# ----------------------------------------------------------------------


@configclass
class RenderBenchAnymalTabletopEnvCfg(RenderBenchBaseEnvCfg):
    # Anymal is ~1m long × 0.7m wide × 0.7m tall, centered at env origin.
    # Props at (1.0, ±0.5, 0.3). Scene span ~2m along +X. Camera centered
    # at (0.5, 0, 3.5) with focal=15 frames everything top-down.
    tiled_camera: _RenderBenchTiledCameraCfg = _tiled_with_camera(
        _top_down_camera(pos=(0.5, 0.0, 3.5), focal_length=15.0)
    )

    articulations: dict = {"robot": ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot")}
    rigid_objects: dict = {
        "cube": _cube("Cube", pos=(1.0, 0.5, 0.3)),
        "sphere": _sphere("Sphere", pos=(1.0, -0.5, 0.3)),
        "cylinder": _cylinder("Cylinder", pos=(1.5, 0.0, 0.3)),
    }


# ----------------------------------------------------------------------
# Variant 2: Franka Panda arm + tabletop props.
# ----------------------------------------------------------------------


@configclass
class RenderBenchFrankaTabletopEnvCfg(RenderBenchBaseEnvCfg):
    """Franka Panda arm anchored at env origin with five small props in
    front. Tighter scene — top-down camera lower for closer framing."""

    # Franka base at origin, arm reaches ~1m. Props at (0.5–0.7, ±0.2, ~0.05).
    # Scene span ~1m. Camera at (0.3, 0, 1.8) with focal=12 (wider FOV) gives
    # close top-down view of the workspace.
    tiled_camera: _RenderBenchTiledCameraCfg = _tiled_with_camera(
        _top_down_camera(pos=(0.3, 0.0, 1.8), focal_length=12.0)
    )

    # Default scene clone-from-source (replicate_physics=True) — re-enabled
    # since per-env material override isn't feasible here anyway, and we get
    # cheaper setup. Per-env variety comes from IK pose phase + DexCube
    # rotation randomization (see render_bench_env.py).
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=2.5, replicate_physics=True)

    # FRANKA_PANDA_HIGH_PD_CFG = higher PD gains for precise IK tracking. Also
    # disables gravity on the arm rigid_props so it doesn't sag under load.
    articulations: dict = {"robot": FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot")}
    rigid_objects: dict = {
        # Colored DexCube — multi-face-color cube. The IK controller cycles
        # the EE through poses above/at/lifting it so each env's arm visually
        # picks at the cube.
        "dex_cube": RigidObjectCfg(
            prim_path="/World/envs/env_.*/DexCube",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.05)),
        ),
        # Other PreviewSurface props. Note: per-env color randomization via
        # MultiAssetSpawnerCfg+PreviewSurfaceCfg requires Isaac Sim Kit for the
        # material binding step; without Kit (Newton-only setup), the colors
        # fall back to default gray. The DexCube above keeps its baked colors.
        "cube_b": _cube("CubeB", pos=(0.5, 0.2, 0.05), color=(0.1, 0.7, 0.4), size=(0.08, 0.08, 0.08)),
        "cube_c": _cube("CubeC", pos=(0.5, -0.2, 0.05), color=(0.1, 0.3, 0.9), size=(0.08, 0.08, 0.08), metallic=0.6),
        "sphere": _sphere("Sphere", pos=(0.7, 0.15, 0.06), radius=0.06, color=(0.95, 0.85, 0.2), metallic=0.8),
        "cylinder": _cylinder("Cylinder", pos=(0.65, -0.15, 0.07), radius=0.05, height=0.14, color=(0.7, 0.3, 0.8)),
    }

    # Drive the Franka arm with a DifferentialIK pick cycle: hover above the
    # DexCube → descend to it → lift up → carry sideways → repeat. Quaternion
    # (0, 1, 0, 0) is 180° around +Y so the gripper points straight down.
    ik_articulation: str | None = "robot"
    ik_target_body: str = "panda_hand"
    ik_joint_pattern: str = "panda_joint.*"
    ik_goals: list = [
        [0.5, 0.0, 0.30, 0.0, 1.0, 0.0, 0.0],  # hover above cube, gripper down
        [0.5, 0.0, 0.12, 0.0, 1.0, 0.0, 0.0],  # descend toward cube
        [0.5, 0.0, 0.40, 0.0, 1.0, 0.0, 0.0],  # lift up
        [0.3, 0.3, 0.40, 0.0, 1.0, 0.0, 0.0],  # carry to side
    ]
    ik_cycle_steps: int = 40

    # Per-env material color randomization is not supported in our Newton-only
    # setup (PreviewSurfaceCfg → Kit dependency; DexCube colors are baked into
    # a shared external USD). Instead, randomize the DexCube's initial
    # orientation per env so each tile shows a different colored face from
    # the top-down view.
    randomize_prop_colors: list = []
    randomize_dexcube_rotation: bool = True


# ----------------------------------------------------------------------
# Variant 3: Multi-robot "kitchen sink".
# ----------------------------------------------------------------------


@configclass
class RenderBenchKitchenSinkEnvCfg(RenderBenchBaseEnvCfg):
    """Heaviest variant: Anymal-C + Franka Panda + several props in the same
    env. Largest spatial extent so camera is highest and widest."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=6.0, replicate_physics=True)

    # Anymal at (0,0,0) area, Franka at (2,0,0). Props spread between them
    # and out to (~2.8, ±1.0). Scene span ~3m × 2m. Camera at (1.0, 0, 5.0)
    # with focal=10 (very wide) frames both robots + props.
    tiled_camera: _RenderBenchTiledCameraCfg = _tiled_with_camera(
        _top_down_camera(pos=(1.0, 0.0, 5.0), focal_length=10.0)
    )

    articulations: dict = {
        "anymal": ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Anymal"),
        "franka": FRANKA_PANDA_CFG.replace(
            prim_path="/World/envs/env_.*/Franka",
            init_state=FRANKA_PANDA_CFG.init_state.replace(pos=(2.0, 0.0, 0.0)),
        ),
    }
    rigid_objects: dict = {
        "cube_red": _cube("CubeRed", pos=(1.0, 1.0, 0.15), color=(0.95, 0.15, 0.15), size=(0.3, 0.3, 0.3)),
        "cube_green": _cube("CubeGreen", pos=(1.0, -1.0, 0.15), color=(0.15, 0.95, 0.3), size=(0.3, 0.3, 0.3)),
        "cube_blue": _cube("CubeBlue", pos=(2.0, 1.2, 0.15), color=(0.15, 0.3, 0.95), size=(0.3, 0.3, 0.3), metallic=0.6),
        "sphere_yellow": _sphere("SphereYellow", pos=(1.5, 0.5, 0.2), color=(0.95, 0.85, 0.2), radius=0.2, metallic=0.8),
        "sphere_purple": _sphere("SpherePurple", pos=(2.5, -0.5, 0.2), color=(0.7, 0.3, 0.85), radius=0.2),
        "cylinder_orange": _cylinder("Cyl1", pos=(0.8, -0.4, 0.2), color=(0.95, 0.55, 0.15), radius=0.18, height=0.4),
        "cylinder_teal": _cylinder("Cyl2", pos=(2.8, 0.3, 0.2), color=(0.15, 0.7, 0.75), radius=0.18, height=0.4),
    }
    dome_light_intensity: float = 2500.0


# ----------------------------------------------------------------------
# Variant 4: Anymal-C on procedural rough terrain.
# ----------------------------------------------------------------------


@configclass
class RenderBenchAnymalTerrainEnvCfg(RenderBenchBaseEnvCfg):
    """Anymal-C standing on procedurally-generated rough terrain (heightfield
    mesh). The terrain is heavy geometry compared to a flat plane — good for
    stressing ray-triangle intersection cost."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=4.0, replicate_physics=True)

    # Camera framed close to the robot: 2m above env origin with focal=24
    # (~47° FOV → ~1.7m ground coverage) so the ~1m Anymal fills the majority
    # of the tile with a sliver of terrain showing around it.
    tiled_camera: _RenderBenchTiledCameraCfg = _tiled_with_camera(
        _top_down_camera(pos=(0.0, 0.0, 2.0), focal_length=24.0)
    )

    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    articulations: dict = {
        "robot": ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
            init_state=ANYMAL_C_CFG.init_state.replace(pos=(0.0, 0.0, 0.7))
        ),
    }
    rigid_objects: dict = {}


# ----------------------------------------------------------------------
# Variant 5: Franka Lift-Cube (mirrors Isaac-Lift-Cube-Franka-v0).
# ----------------------------------------------------------------------


@configclass
class _LiftCubeSceneCfg(InteractiveSceneCfg):
    """Scene cfg for the Lift-Cube variant. ALL env-scoped entities (table,
    robot, dex_cube, target_marker) are scene-managed so IsaacLab's standard
    cloning plan handles them uniformly. The custom env's ``_setup_scene``
    grabs handles by name from ``self.scene.articulations`` /
    ``self.scene.rigid_objects`` rather than constructing them manually.

    Poses match /home/horde/Downloads/stage.usda env_0:
      Table: translate=(0.5, 0, 0), orient=(qw=0.707, 0, 0, qz=0.707)
      Robot: at env origin (default Franka init)
      Object (DexCube): on table at translate=(0.5, 0, 0.055), scale 0.8
    """

    # Table as a kinematic RigidObject using built-in CuboidCfg geometry.
    # The actual SeattleLabTable USD reference doesn't propagate to env_N
    # under Newton-only execution (confirmed empirically: only env_0 renders
    # the USD; cloned envs have the prim+reference at the layer level but
    # the Hydra/Newton render pipeline doesn't compose it). A CuboidCfg
    # clones reliably through the standard RigidObject path. Dimensions
    # (1.0 × 0.5 × 1.0 m) approximate the SeattleLabTable's bounding box.
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 1.0, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.20, 0.20, 0.22), metallic=0.25
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, kinematic_enabled=True
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, -0.5)),
    )
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    dex_cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/DexCube",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.055)),
    )
    target_marker: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetMarker",
        spawn=sim_utils.SphereCfg(
            radius=0.06,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.95, 0.15, 0.15), metallic=0.1
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, kinematic_enabled=True
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.35)),
    )


@configclass
class RenderBenchFrankaLiftCubeEnvCfg(RenderBenchBaseEnvCfg):
    """Franka picks a DexCube off a Seattle-Lab table and carries it to a
    per-env sampled target pose (visualized with a red sphere marker).
    Models the scene of ``Isaac-Lift-Cube-Franka-v0``.

    The "grasp" is faked by teleporting the cube to the gripper during the
    carry phases — no real contact-based grasping is needed since this is a
    render benchmark, not a manipulation benchmark."""

    # SeattleLabTable USD has its origin at the table SURFACE; legs extend
    # down ~1m. Drop the floor to -1.05 so the table sits flush on the floor
    # — matches the canonical Isaac-Lift-Cube-Franka scene layout (see
    # /home/horde/Downloads/robothand_64env/Stage.usd env_0/Table transform).
    ground_plane_z: float = -1.05

    # Clean 3/4 top-down camera (no image roll). Positioned in front-right and
    # above the workspace at (1.2, -1.2, 1.3), looking at the workspace center
    # (0.3, 0, 0.1). Resulting orientation:
    #   pitch ≈ 39° down from horizontal
    #   yaw   ≈ 127° from +X  (i.e. looks back-and-left)
    #   roll  = 0 (image-up stays world-up)
    # Quaternion (qx,qy,qz,qw) = (-0.296, 0.148, 0.844, 0.422), focal 18.15.
    # Note: the user-supplied isaac_anim2.usda's saved viewport had a 16° Z
    # roll plus zero pitch, which produced a tilted-floor image — that pose
    # is not used here.
    tiled_camera: _RenderBenchTiledCameraCfg = _tiled_with_camera(
        _perspective_camera(
            pos=(1.2, -1.2, 1.3),
            rot=(-0.2961, 0.1480, 0.8440, 0.4220),
            focal_length=18.147562,
        )
    )

    # Lighting: just a bright ambient DomeLight (no DistantLight). The
    # directional sun-light from the USDA was producing long shadow streaks
    # on the ground in Newton's renderer that read as wireframe artifacts;
    # ambient-only gives a cleaner image that's closer to the user's
    # reference rendering.
    light_cfg: sim_utils.LightCfg | None = None
    dome_light_intensity: float = 3000.0

    # All env-scoped entities live in the scene cfg so IsaacLab's standard
    # clone plan handles them uniformly (matching the ManagerBasedRLEnv
    # pattern). The env's _setup_scene fetches handles by name from the
    # already-constructed scene rather than re-constructing them.
    scene: InteractiveSceneCfg = _LiftCubeSceneCfg(
        num_envs=4, env_spacing=2.5, replicate_physics=True
    )

    # Manual entity dicts are empty — everything is in the scene cfg above.
    articulations: dict = {}

    static_assets: dict = {}
    rigid_objects: dict = {}

    # IK pick-and-place cycle. The waypoint XYZ at indices in
    # ``target_goal_indices`` is REPLACED at runtime with the per-env sampled
    # target. The cube is teleported to follow the gripper for indices in
    # ``grasped_goal_indices``.
    ik_articulation: str | None = "robot"
    ik_target_body: str = "panda_hand"
    ik_joint_pattern: str = "panda_joint.*"
    ik_goals: list = [
        [0.5, 0.0, 0.30, 0.0, 1.0, 0.0, 0.0],  # 0: hover above cube
        [0.5, 0.0, 0.13, 0.0, 1.0, 0.0, 0.0],  # 1: descend to grasp
        [0.5, 0.0, 0.30, 0.0, 1.0, 0.0, 0.0],  # 2: lift up (grasped)
        [0.5, 0.0, 0.40, 0.0, 1.0, 0.0, 0.0],  # 3: carry-to-target (XYZ substituted; grasped)
        [0.5, 0.0, 0.40, 0.0, 1.0, 0.0, 0.0],  # 4: hold at target (XYZ substituted; grasped)
        [0.5, 0.0, 0.55, 0.0, 1.0, 0.0, 0.0],  # 5: release / retract (XY substituted, Z raised)
    ]
    ik_cycle_steps: int = 35

    lift_cube_task: bool = True
    target_pose_range_x: tuple = (0.4, 0.6)
    target_pose_range_y: tuple = (-0.25, 0.25)
    target_pose_range_z: tuple = (0.25, 0.5)
    # Goals 3-5 use the sampled target XY. Goal 5 raises Z to retract above.
    target_goal_indices: tuple = (3, 4, 5)
    grasped_goal_indices: tuple = (2, 3, 4)


# ----------------------------------------------------------------------
# Variant 6: Franka + Sektion cabinet (mirrors Isaac-Franka-Cabinet-Direct-v0).
# ----------------------------------------------------------------------


@configclass
class RenderBenchFrankaCabinetEnvCfg(RenderBenchBaseEnvCfg):
    """Franka Panda + Sektion cabinet (4 articulated joints: 2 drawers, 2
    doors). Sinusoidal animation drives all joints so the rendered frames
    show drawers sliding in/out and doors swinging — the cabinet adds
    significant articulated-geometry complexity vs. the existing variants.

    Poses match the canonical ``Isaac-Franka-Cabinet-Direct-v0`` task:
      Franka: pos=(1.0, 0, 0), rot=180° around Y (faces -X toward cabinet)
      Cabinet: pos=(0.0, 0, 0.4), rot=180° around Z (opens toward -X)
    """

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4, env_spacing=3.0, replicate_physics=True
    )

    # Front view with no yaw / no roll — pitch-only camera. Sits in front of
    # the cabinet (+X side, the drawer-facing side), elevated, pitched ~35°
    # down so the workspace reads as a slight bird's-eye. Tighter focal so
    # the Franka + cabinet fill most of the frame.
    #   pos        = (2.0, 0.0, 1.5)
    #   look-at    = (0.4, 0, 0.4)  →  forward = (-0.824, 0, -0.566)
    #   pitch      = ~34.5° down,  yaw = 0,  roll = 0
    # Quaternion (qxyzw) = (-0.296, 0, 0.955, 0) — qw=0, qx and qz only,
    # corresponds to a 180° rotation around the (-X, 0, +Z) axis.
    tiled_camera: _RenderBenchTiledCameraCfg = _tiled_with_camera(
        _perspective_camera(
            pos=(2.0, 0.0, 1.5),
            rot=(-0.296, 0.0, 0.955, 0.0),
            focal_length=24.0,
            warp_enable_shadows=True,
        )
    )

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

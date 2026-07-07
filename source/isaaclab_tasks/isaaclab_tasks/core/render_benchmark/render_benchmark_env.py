# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct env for render-only benchmarking.

Loops the cfg's ``articulations`` and ``rigid_objects`` dicts to populate the
scene, then drives each articulation either with a sinusoidal joint animation
(default) or a DifferentialIK pick cycle (if ``cfg.ik_articulation`` is set).
Optional per-env diffuse-color randomization for PreviewSurface props is
applied at scene setup.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import torch

from pxr import Gf

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera, save_images_to_file
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from .render_benchmark_env_cfg import RenderBenchmarkFrankaCabinetEnvCfg


def _randomize_prop_colors(prim_path_expr: str) -> int:
    """Write random ``diffuseColor`` overrides on every prim matching the regex.

    Returns the number of prims actually touched. Walks each matched prim's
    descendants for any ``Shader`` prim with a ``diffuseColor`` input rather
    than assuming a fixed sub-path layout."""
    stage = get_current_stage()
    prim_paths = sim_utils.find_matching_prim_paths(prim_path_expr)
    touched = 0
    if not prim_paths:
        print(f"[render_benchmark]   (no prims matched {prim_path_expr})")
        return 0
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            continue
        # Walk descendants for a Shader prim with a diffuseColor input.
        shader_attr = None
        for descendant in prim.GetAllChildren() + [prim]:
            for sub in descendant.GetAllChildren():
                if sub.GetTypeName() == "Shader":
                    attr = sub.GetAttribute("inputs:diffuseColor")
                    if attr and attr.IsValid():
                        shader_attr = attr
                        break
                # one more level
                for subsub in sub.GetAllChildren():
                    if subsub.GetTypeName() == "Shader":
                        attr = subsub.GetAttribute("inputs:diffuseColor")
                        if attr and attr.IsValid():
                            shader_attr = attr
                            break
                if shader_attr:
                    break
            if shader_attr:
                break
        if not shader_attr:
            if touched == 0:  # log once per expr
                print(f"[render_benchmark]   no diffuseColor Shader under {prim_path}; children:")
                for c in prim.GetAllChildren():
                    print(f"    {c.GetPath()} (type={c.GetTypeName()})")
            continue
        shader_attr.Set(Gf.Vec3f(random.random(), random.random(), random.random()))
        touched += 1
    return touched


class RenderBenchmarkEnv(DirectRLEnv):
    cfg: RenderBenchmarkFrankaCabinetEnvCfg

    # --- scene construction --------------------------------------------------

    def _setup_scene(self):
        # If the scene cfg already manages entities (cfg-driven path, e.g.
        # ``_LiftCubeSceneCfg``), grab handles by name from the scene that
        # InteractiveScene.__init__ already constructed. Otherwise fall back
        # to the manual construction path used by the other variants.
        cfg_driven = self.scene._is_scene_setup_from_cfg()
        if cfg_driven:
            self._articulations = dict(self.scene.articulations)
            self._rigid_objects = dict(self.scene.rigid_objects)
        else:
            self._articulations: dict[str, Articulation] = {}
            for name, art_cfg in (self.cfg.articulations or {}).items():
                self._articulations[name] = Articulation(art_cfg)

            self._rigid_objects: dict[str, RigidObject] = {}
            for name, ro_cfg in (self.cfg.rigid_objects or {}).items():
                self._rigid_objects[name] = RigidObject(ro_cfg)

        self._tiled_camera = Camera(self.cfg.tiled_camera)

        if self.cfg.terrain is not None:
            TerrainImporter(self.cfg.terrain)
        elif self.cfg.use_flat_ground:
            # Flat-color ground via CuboidCfg routed through RigidObject so
            # both OVRTX and Warp see the same geometry (apple-to-apple
            # comparison). The USD GroundPlaneCfg path produces a USD-only
            # plane that Warp's render_megakernel doesn't see.
            sx, sy = self.cfg.flat_ground_size
            sz = self.cfg.flat_ground_thickness
            top_at = self.cfg.ground_plane_z
            ground_rigid_cfg = RigidObjectCfg(
                prim_path="/World/envs/env_.*/Floor",
                spawn=sim_utils.CuboidCfg(
                    size=(sx, sy, sz),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=self.cfg.flat_ground_color, metallic=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, top_at - sz / 2.0)),
            )
            self._rigid_objects["__floor"] = RigidObject(ground_rigid_cfg)
        else:
            # Default: USD-loaded grid floor. ``color=(1,1,1)`` overrides the
            # cfg default (0,0,0) so the USD's base diffuse texture color
            # shows through under Newton; Warp's renderer doesn't see this
            # plane at all (clear color shows through instead).
            ground_cfg = sim_utils.GroundPlaneCfg(color=(1.0, 1.0, 1.0))
            ground_cfg.func(
                "/World/defaultGroundPlane",
                ground_cfg,
                translation=(0.0, 0.0, self.cfg.ground_plane_z),
            )

        # Always spawn a dim ambient DomeLight so non-lit surfaces don't go
        # pure black under Newton's renderer (which lacks Kit's tone mapping
        # and ambient defaults).
        dome_cfg = sim_utils.DomeLightCfg(intensity=self.cfg.dome_light_intensity, color=(0.75, 0.75, 0.75))
        dome_cfg.func("/World/Light", dome_cfg)
        # Optional directional light (e.g. DistantLight) on top of the ambient.
        if self.cfg.light_cfg is not None:
            self.cfg.light_cfg.func(
                "/World/LightDirectional",
                self.cfg.light_cfg,
                orientation=self.cfg.light_orientation,
            )

        # Skip the manual clone for the cfg-driven path — InteractiveScene
        # has already run the clone, and re-running it would destructively
        # overwrite env_N with stale env_0 spec.
        if not cfg_driven:
            src, dest = "/World/envs/env_0", "/World/envs/env_{}"
            pos = cloner.grid_transforms(self.scene.num_envs, self.scene.cfg.env_spacing, device=self.device)[0]
            plan = cloner.ClonePlan.from_env_0(src, dest, self.scene.num_envs, self.device, pos)
            cloner.replicate(plan, stage=self.scene.stage)

            # Static USD references (e.g. SeattleLabTable) — manual per-env
            # spawn for non-cfg-driven variants. The quaternion in
            # ``AssetBaseCfg.InitialStateCfg.rot`` is ``(qw, qx, qy, qz)``;
            # ``spawn_from_usd``'s ``orientation`` arg expects ``(qx, qy, qz, qw)``.
            if self.cfg.static_assets:
                for name, asset_cfg in self.cfg.static_assets.items():
                    qw, qx, qy, qz = asset_cfg.init_state.rot
                    orient_xyzw = (qx, qy, qz, qw)
                    for env_id in range(self.num_envs):
                        asset_cfg.spawn.func(
                            f"/World/envs/env_{env_id}/{name}",
                            asset_cfg.spawn,
                            translation=asset_cfg.init_state.pos,
                            orientation=orient_xyzw,
                        )

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        if not cfg_driven:
            for name, art in self._articulations.items():
                self.scene.articulations[name] = art
            for name, ro in self._rigid_objects.items():
                self.scene.rigid_objects[name] = ro
        self.scene.sensors["tiled_camera"] = self._tiled_camera

        # Per-env color randomization for the listed PreviewSurface props.
        for prop_name in self.cfg.randomize_prop_colors or []:
            touched = _randomize_prop_colors(f"/World/envs/env_.*/{prop_name}")
            print(f"[render_benchmark] randomized {touched} {prop_name} prop colors")

    # --- physics step --------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._lazy_init_motion()
        # One-time per-env DexCube rotation randomization (after assets are
        # initialized; can't run in _setup_scene because the sim isn't ready).
        if (
            getattr(self.cfg, "randomize_dexcube_rotation", False)
            and not getattr(self, "_dexcube_randomized", False)
            and "dex_cube" in self._rigid_objects
        ):
            self._randomize_dexcube_orientation()
            self._dexcube_randomized = True
        # Pin the target marker to the per-env sampled lift target.
        if self._lift_active:
            self._write_target_marker()
        if self._ik_active:
            self._ik_step()
        if self.cfg.joint_animation_amplitude > 0.0:
            self._sin_animate(skip=self.cfg.ik_articulation)

    def _randomize_dexcube_orientation(self):
        """Sample one random unit quaternion per env and write it to the
        DexCube's root pose, keeping the configured position."""
        dex = self._rigid_objects["dex_cube"]
        # Sample uniform random unit quaternions in (w, x, y, z) order then
        # interleave to write the (env, 7)=(px,py,pz,qw,qx,qy,qz) state.
        u1 = torch.rand(self.num_envs, device=self.device)
        u2 = torch.rand(self.num_envs, device=self.device) * (2.0 * math.pi)
        u3 = torch.rand(self.num_envs, device=self.device) * (2.0 * math.pi)
        s1 = torch.sqrt(1.0 - u1)
        s2 = torch.sqrt(u1)
        # Shoemake's algorithm — uniform random unit quaternions.
        qw = s1 * torch.sin(u2)
        qx = s1 * torch.cos(u2)
        qy = s2 * torch.sin(u3)
        qz = s2 * torch.cos(u3)
        # Default position from cfg (keep all envs at the same spot).
        cur_pose = dex.data.root_pose_w.torch.clone()  # (envs, 7) = pos+quat
        cur_pose[:, 3] = qw
        cur_pose[:, 4] = qx
        cur_pose[:, 5] = qy
        cur_pose[:, 6] = qz
        dex.write_root_pose_to_sim_index(root_pose=cur_pose)
        print(f"[render_benchmark] randomized DexCube rotation for {self.num_envs} envs")

    def _apply_action(self) -> None:
        pass

    # --- IK pick cycle -------------------------------------------------------

    def _lazy_init_motion(self):
        if hasattr(self, "_motion_init_done"):
            return
        self._motion_init_done = True

        # Sinusoidal phases per (env, joint) for every articulation. The IK
        # articulation's phases are unused but cheap to compute and harmless.
        # Use a DETERMINISTIC hash of (env_idx, joint_idx) so the simulation
        # is reproducible across runs and rendering backends — otherwise the
        # default torch RNG state varies process-to-process and OVRTX vs.
        # Warp runs end up at different joint poses at the same frame even
        # though the physics solver itself is deterministic.
        self._anim_phases: dict[str, torch.Tensor] = {}
        for name, art in self._articulations.items():
            default_pos = art.data.default_joint_pos.torch
            n_envs, n_joints = default_pos.shape
            env_idx = torch.arange(n_envs, device=self.device, dtype=default_pos.dtype).unsqueeze(1)
            joint_idx = torch.arange(n_joints, device=self.device, dtype=default_pos.dtype).unsqueeze(0)
            # Coprime primes spread phases pseudo-uniformly across (env, joint).
            phase_int = (env_idx * 7919.0 + joint_idx * 6553.0) % 10007.0
            self._anim_phases[name] = phase_int * (2.0 * math.pi / 10007.0)
        self._anim_t = 0.0

        # IK setup — only if a target articulation is specified.
        self._ik_active = False
        self._lift_active = False
        if not self.cfg.ik_articulation:
            return
        art = self._articulations.get(self.cfg.ik_articulation)
        if art is None or not self.cfg.ik_goals:
            return

        self._ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=self.num_envs,
            device=self.device,
        )
        entity_cfg = SceneEntityCfg(
            self.cfg.ik_articulation,
            joint_names=[self.cfg.ik_joint_pattern],
            body_names=[self.cfg.ik_target_body],
        )
        entity_cfg.resolve(self.scene)
        self._ik_entity_cfg = entity_cfg
        # Jacobian body-id offset: fixed base → drop 1 for the root link;
        # floating base → 0.
        self._ik_ee_jacobi_idx = entity_cfg.body_ids[0] - (1 if art.is_fixed_base else 0)

        self._ik_goals = torch.tensor(self.cfg.ik_goals, device=self.device)  # (G, 7)
        # Per-env phase offset so each env is at a different point in the cycle.
        n_goals = self._ik_goals.shape[0]
        cycle_total = n_goals * self.cfg.ik_cycle_steps
        self._ik_env_phase = torch.arange(self.num_envs, device=self.device) * (cycle_total // max(self.num_envs, 1))
        self._ik_step_counter = 0
        self._ik_active = True

        # ----- Lift-cube task setup -----
        self._lift_active = False
        if (
            getattr(self.cfg, "lift_cube_task", False)
            and "dex_cube" in self._rigid_objects
            and "target_marker" in self._rigid_objects
        ):
            # Sample per-env target XYZ in the configured ranges.
            def _sample(low_hi):
                lo, hi = low_hi
                return torch.rand(self.num_envs, device=self.device) * (hi - lo) + lo

            tx = _sample(self.cfg.target_pose_range_x)
            ty = _sample(self.cfg.target_pose_range_y)
            tz = _sample(self.cfg.target_pose_range_z)
            self._lift_target_xyz = torch.stack([tx, ty, tz], dim=-1)  # (num_envs, 3)
            self._lift_target_goal_indices = torch.tensor(
                list(self.cfg.target_goal_indices), device=self.device, dtype=torch.long
            )
            self._lift_grasped_goal_indices = set(int(i) for i in self.cfg.grasped_goal_indices)
            self._lift_release_goal_indices = set(  # goals using sampled XY but with raised Z (release-above)
                int(i) for i in self.cfg.target_goal_indices if int(i) not in self._lift_grasped_goal_indices
            )
            # Goal index → sampled-XY z-offset table (release waypoint sits above
            # the target; carry/hold waypoints sit at the target).
            self._lift_z_override_above = 0.15
            # Cache the dex cube's default Z so we can hold it in place during
            # non-grasped phases.
            # Read the cube's rest Z from whichever source carries it (manual
            # rigid_objects dict for the legacy variants, or the scene cfg
            # for the cfg-driven Lift-Cube variant).
            if self.cfg.rigid_objects and "dex_cube" in self.cfg.rigid_objects:
                cube_init_z = self.cfg.rigid_objects["dex_cube"].init_state.pos[2]
            else:
                cube_init_z = self.cfg.scene.dex_cube.init_state.pos[2]
            self._lift_cube_rest_z = float(cube_init_z)
            self._lift_active = True
            print(
                f"[render_benchmark] lift_cube_task: sampled {self.num_envs} targets in "
                f"x={tuple(self.cfg.target_pose_range_x)} y={tuple(self.cfg.target_pose_range_y)} "
                f"z={tuple(self.cfg.target_pose_range_z)}"
            )

    def _ik_step(self):
        art = self._articulations[self.cfg.ik_articulation]
        entity = self._ik_entity_cfg

        # Per-env current goal index
        n_goals = self._ik_goals.shape[0]
        env_step = self._ik_step_counter + self._ik_env_phase
        goal_idx = (env_step // self.cfg.ik_cycle_steps) % n_goals
        ik_command = self._ik_goals[goal_idx].clone()  # (num_envs, 7)

        # Lift task: substitute the per-env sampled target XYZ for designated
        # goal indices. Release-above goals raise Z by _lift_z_override_above.
        if self._lift_active:
            for g in self.cfg.target_goal_indices:
                mask = goal_idx == g
                if not mask.any():
                    continue
                ik_command[mask, 0] = self._lift_target_xyz[mask, 0]
                ik_command[mask, 1] = self._lift_target_xyz[mask, 1]
                z_off = self._lift_z_override_above if g in self._lift_release_goal_indices else 0.0
                ik_command[mask, 2] = self._lift_target_xyz[mask, 2] + z_off

        self._ik.set_command(ik_command)
        self._ik_step_counter += 1

        # Read kinematic quantities. ``num_base_dofs`` shifts joint ids past
        # the floating-base columns in the Jacobian (0 for fixed-base Franka).
        jacobi_joint_ids = [j + art.num_base_dofs for j in entity.joint_ids]
        jacobian = art.data.body_link_jacobian_w.torch[:, self._ik_ee_jacobi_idx, :, jacobi_joint_ids]
        ee_pose_w = art.data.body_pose_w.torch[:, entity.body_ids[0]]
        root_pose_w = art.data.root_pose_w.torch
        joint_pos = art.data.joint_pos.torch[:, entity.joint_ids]

        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        joint_pos_des = self._ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        art.set_joint_position_target_index(target=joint_pos_des, joint_ids=entity.joint_ids)

        # Lift task: during grasped phases, teleport the cube to follow the
        # gripper (offset slightly below the panda_hand body). Approximates a
        # grasp without contact dynamics — fine for a render benchmark.
        if self._lift_active and self._lift_grasped_goal_indices:
            grasp_mask = torch.zeros_like(goal_idx, dtype=torch.bool)
            for g in self._lift_grasped_goal_indices:
                grasp_mask |= goal_idx == g
            if grasp_mask.any():
                dex = self._rigid_objects["dex_cube"]
                cur_pose = dex.data.root_pose_w.torch.clone()
                # Cube sits ~10cm below the hand origin (between fingers).
                cur_pose[grasp_mask, 0] = ee_pose_w[grasp_mask, 0]
                cur_pose[grasp_mask, 1] = ee_pose_w[grasp_mask, 1]
                cur_pose[grasp_mask, 2] = ee_pose_w[grasp_mask, 2] - 0.10
                dex.write_root_pose_to_sim_index(root_pose=cur_pose)

    def _write_target_marker(self):
        """Write the target marker rigid body's root pose to the per-env
        sampled lift target. Called every step; the body is kinematic."""
        marker = self._rigid_objects["target_marker"]
        pose = marker.data.root_pose_w.torch.clone()
        pose[:, 0:3] = self._lift_target_xyz
        pose[:, 3] = 1.0  # qw
        pose[:, 4:7] = 0.0
        marker.write_root_pose_to_sim_index(root_pose=pose)

    # --- Sinusoidal joint animation (applied to non-IK articulations) -------

    def _sin_animate(self, skip: str | None = None):
        self._anim_t += self.cfg.sim.dt * self.cfg.decimation
        omega = 2.0 * math.pi * self.cfg.joint_animation_freq_hz
        for name, art in self._articulations.items():
            if name == skip:
                continue
            default_pos = art.data.default_joint_pos.torch
            phase = self._anim_phases[name]
            target = default_pos + self.cfg.joint_animation_amplitude * torch.sin(omega * self._anim_t + phase)
            soft_limits = art.data.soft_joint_pos_limits.torch
            target = torch.clamp(target, soft_limits[..., 0], soft_limits[..., 1])
            art.set_joint_position_target_index(target=target)

    # --- DirectRLEnv plumbing ------------------------------------------------

    def _get_observations(self) -> dict:
        data_type = self.cfg.tiled_camera.data_types[0]
        out = self._tiled_camera.data.output
        if isinstance(out, dict) and data_type in out and self.cfg.write_image_to_file:
            try:
                cam = out[data_type]
                if not torch.is_tensor(cam):
                    cam = torch.from_dlpack(cam)
                img = cam.float()
                if img.max() > 1.5:
                    img = img / 255.0
                save_images_to_file(img[:, ..., :3], f"render_benchmark_{data_type}.png")
            except Exception as e:
                print(f"[render_benchmark] write_image_to_file failed: {e}")
        return {"policy": torch.zeros((self.num_envs, 1), device=self.device)}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros_like(time_out), time_out

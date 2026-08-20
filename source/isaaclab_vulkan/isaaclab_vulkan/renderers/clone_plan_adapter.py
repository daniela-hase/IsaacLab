# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ClonePlan-driven scene adapter for the Vulkan ray tracing renderer.

Under kit-less physics backends (e.g. Newton), the USD stage only authors the *prototype*
environment(s); the remaining environments are replicated in the physics model, not in USD. Walking
the stage per-environment therefore only finds one complete environment. This adapter instead follows
the authoritative :class:`~isaaclab.cloner.clone_plan.ClonePlan`:

* geometry is loaded **once** per prototype source root (which *is* authored in USD),
* one Vulkan render world is built per environment by replicating the source assigned to it (the
  ``clone_mask`` selects the variant, so heterogeneous multi-asset scenes are supported),
* environment origins come from :attr:`ClonePlan.positions`, and
* per-frame motion is driven from the :class:`~isaaclab.scene_data.SceneDataProvider`'s world-space
  rigid-body transforms, mapped to each replicated world through the clone plan's destination
  templates.

Mesh geometry is shared across worlds (the underlying :class:`vulkan_renderer.VulkanRenderer` dedups
mesh resources by identity), so replication costs one TLAS instance per placement rather than a full
copy of the geometry.

The adapter uses the public ``vulkan_renderer`` primitives plus its ``build_usd_scene`` loader; all
ClonePlan-specific logic lives here in Isaac Lab.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import warp as wp

logger = logging.getLogger(__name__)

# Env-slot placeholder used by ClonePlan destination templates.
_ENV_SLOT = "{}"

# Sentinel: a visual material is bound but textured (albedo comes from a texture we do not apply yet).
_TEXTURED = "textured"


class ClonePlanAdapter:
    """Builds and drives a replicated multi-world Vulkan scene from a :class:`ClonePlan`.

    Args:
        stage: USD stage that authors the prototype geometry.
        plan: Active clone plan from :meth:`SimulationContext.get_clone_plan`.
        transform_paths: Prim paths of the tracked rigid bodies, aligned with the transforms the
            :class:`~isaaclab.scene_data.SceneDataProvider` returns (``provider.backend.transform_paths``).
        device: Warp device for the render buffers and per-frame kernels (e.g. ``"cuda:0"``).
        num_envs: Number of environments (render worlds) to build.
        include_guides: Also render ``guide``/``proxy`` purpose prims (e.g. collision shapes).
        verbose: Print the geometry-load summary.
        lights: Optional Vulkan lights list.
        ambient: Constant ambient term.
        shadows_enabled: Enable shadow rays.
    """

    def __init__(
        self,
        stage: Any,
        plan: Any,
        transform_paths: list[str],
        *,
        device: str,
        num_envs: int,
        include_guides: bool = False,
        verbose: bool = True,
        lights: Any = None,
        ambient: float = 0.3,
        shadows_enabled: bool = True,
    ) -> None:
        try:
            from vulkan_renderer import VulkanRenderer, create_vulkan_context
            from vulkan_renderer.renderer import (
                OUTPUT_ALBEDO,
                OUTPUT_DEPTH,
                OUTPUT_FORWARD_DEPTH,
                OUTPUT_HDR,
                OUTPUT_NORMAL,
            )
            from vulkan_renderer.usd_adapter import build_usd_scene
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("The 'vulkan-renderer' package is required for the Vulkan renderer backend.") from exc
        from isaaclab.scene_data.scene_data_backend import SceneDataFormat

        # Cached once so the per-frame render/update paths avoid re-importing and re-looking-up.
        self._Matrix44 = SceneDataFormat.Matrix44
        self._aux_specs = (
            ("hdr", OUTPUT_HDR),
            ("depth", OUTPUT_DEPTH),
            ("forward_depth", OUTPUT_FORWARD_DEPTH),
            ("normal", OUTPUT_NORMAL),
            ("albedo", OUTPUT_ALBEDO),
        )

        self._stage = stage
        self._device = device
        self._num_envs = int(num_envs)

        from pxr import UsdGeom

        self._xform_cache = UsdGeom.XformCache()

        # ------------------------------------------------------------------ #
        # 1. Load ALL authored geometry from the stage as a single flat world.
        # ------------------------------------------------------------------ #
        # world_regex=None keeps everything in world space with absolute per-instance transforms and
        # full ancestor metadata; we partition it into prototypes / globals ourselves below.
        worlds, transforms, meta = build_usd_scene(
            stage, time=None, up_axis="Z", world_regex=None, include_guides=include_guides, verbose=verbose
        )
        instances = worlds[0]
        instance_world = transforms[0]
        instance_paths = [m[0] for m in meta]

        # ``build_usd_scene`` only reads ``displayColor`` / ``UsdPreviewSurface``; robot assets use MDL
        # ``OmniPBR`` materials (and are instanceable), so their meshes come back grey. Re-resolve the
        # bound-material albedo and recolor the instances so they match the Newton renderer.
        if len(instances) == len(instance_paths):
            instances = _recolor_instances(instances, instance_paths, stage, verbose=verbose)

        # ------------------------------------------------------------------ #
        # 2. Resolve clone-plan structure.
        # ------------------------------------------------------------------ #
        sources = [s.rstrip("/") for s in plan.sources]
        destinations = list(plan.destinations)
        clone_mask = plan.clone_mask.detach().cpu().numpy().astype(bool)  # [num_sources, num_clones]
        env_ids = (
            plan.env_ids.detach().cpu().numpy().astype(np.int64)
            if plan.env_ids is not None
            else np.arange(clone_mask.shape[1], dtype=np.int64)
        )
        positions = (
            plan.positions.detach().cpu().numpy().astype(np.float32)
            if plan.positions is not None
            else np.zeros((clone_mask.shape[1], 3), dtype=np.float32)
        )
        num_clones = clone_mask.shape[1]

        # Bodies that belong to a prototype (source-side); nearest such ancestor drives a mesh.
        source_body_set = {b for b in transform_paths if _under_any(b, sources)}

        # Cache each source's env-root world transform (for re-centering prototype geometry env-local).
        # A source prim may sit below its env root (e.g. ``/World/envs/env_0/Robot``); re-centering uses
        # the env root, not the source prim, so the geometry keeps its placement within the env.
        source_env_world = {}
        for s in sources:
            env_root = _env_root(s)
            source_env_world[s] = self._prim_world(env_root) if env_root else np.eye(4, dtype=np.float32)

        # ------------------------------------------------------------------ #
        # 3. Classify each loaded instance: which prototype row, or global, or junk.
        # ------------------------------------------------------------------ #
        # proto_rows[i] = source row index if instance i is prototype geometry, else -1 (global) or
        # None (partial-clone junk under a non-source env root -> dropped).
        proto_instances: dict[int, list[int]] = {r: [] for r in range(len(sources))}
        global_instances: list[int] = []
        for i, path in enumerate(instance_paths):
            row = _source_row(path, sources)
            if row is not None:
                proto_instances[row].append(i)
            elif not _under_env(path):
                global_instances.append(i)
            # else: geometry under a non-source env clone (partial authoring) — skip.

        # Precompute, per prototype instance, its controlling body suffix + mesh-relative-to-body.
        # body_suffix maps into a source's destination template; mesh_rel_body is constant.
        inst_body_suffix: dict[int, str | None] = {}
        inst_mesh_rel_body: dict[int, np.ndarray] = {}
        for row, idxs in proto_instances.items():
            src_root = sources[row]
            for i in idxs:
                body = _nearest_ancestor(instance_paths[i], source_body_set)
                if body is None:
                    inst_body_suffix[i] = None
                    continue
                inst_body_suffix[i] = body[len(src_root) :]  # e.g. "/Robot/link3"
                w_body = self._prim_world(body)
                inst_mesh_rel_body[i] = (np.linalg.inv(w_body) @ instance_world[i]).astype(np.float32)

        # ------------------------------------------------------------------ #
        # 4. Build one render world per clone; collect dynamic-driver metadata.
        # ------------------------------------------------------------------ #
        body_index = {p: k for k, p in enumerate(transform_paths)}
        out_worlds: list[list[Any]] = []
        out_transforms: list[list[np.ndarray]] = []
        # Flat (world-major) dynamic-instance records, aligned with the TLAS instance buffer.
        dyn_record: list[int] = []
        dyn_env_inv: list[np.ndarray] = []
        dyn_body_idx: list[int] = []
        dyn_mesh_rel: list[np.ndarray] = []

        record_offset = 0
        missing_bodies: set[str] = set()
        for j in range(num_clones):
            env_id = int(env_ids[j])
            env_inv = _translation_inverse(positions[j])  # env origins are pure translations
            world_insts: list[Any] = []
            world_xforms: list[np.ndarray] = []

            # An env can be populated by several plan rows (e.g. the robot row plus the object's variant
            # row in a multi-asset scene); include the geometry of every row that covers this env.
            for row in np.nonzero(clone_mask[:, j])[0]:
                row = int(row)
                dest = destinations[row]
                src_env_inv = np.linalg.inv(source_env_world[sources[row]]).astype(np.float32)
                for i in proto_instances[row]:
                    local_index = len(world_insts)
                    world_insts.append(instances[i])
                    # Static construction pose: prototype geometry re-centered to its env origin.
                    world_xforms.append((src_env_inv @ instance_world[i]).astype(np.float32))
                    suffix = inst_body_suffix.get(i)
                    if suffix is None:
                        continue  # prototype-static (e.g. table) — keep the construction pose.
                    clone_body = _apply_template(dest, env_id) + suffix
                    bidx = body_index.get(clone_body)
                    if bidx is None:
                        missing_bodies.add(clone_body)
                        continue
                    dyn_record.append(record_offset + local_index)
                    dyn_env_inv.append(env_inv)
                    dyn_body_idx.append(bidx)
                    dyn_mesh_rel.append(inst_mesh_rel_body[i])

            for i in global_instances:
                world_insts.append(instances[i])
                # Global geometry (e.g. ground) is world-space; express it in this env's local frame.
                world_xforms.append((env_inv @ instance_world[i]).astype(np.float32))

            out_worlds.append(world_insts)
            out_transforms.append(world_xforms)
            record_offset += len(world_insts)

        if missing_bodies:
            logger.warning(
                "ClonePlanAdapter: %d driver bodies had no matching provider transform (e.g. %s);"
                " those meshes will not animate.",
                len(missing_bodies),
                next(iter(sorted(missing_bodies))),
            )
        logger.info(
            "ClonePlanAdapter: %d worlds, %d prototype rows, %d dynamic instances, %d global instances",
            len(out_worlds),
            len(sources),
            len(dyn_record),
            len(global_instances),
        )

        # ------------------------------------------------------------------ #
        # 5. Construct the renderer and upload driver kernel inputs.
        # ------------------------------------------------------------------ #
        self._context = create_vulkan_context(application_name="IsaacLab Vulkan ClonePlan")
        self._renderer = VulkanRenderer(
            self._context,
            out_worlds,
            out_transforms,
            export_instance_buffers=True,
            lights=lights,
            ambient=ambient,
            shadows_enabled=shadows_enabled,
        )
        self.world_count = len(out_worlds)
        self._num_bodies = len(transform_paths)

        self._num_dynamic = len(dyn_record)
        if self._num_dynamic:
            self._dyn_record = wp.array(np.asarray(dyn_record, dtype=np.int32), dtype=wp.int32, device=device)
            self._dyn_env_inv = wp.array(np.stack(dyn_env_inv), dtype=wp.mat44f, device=device)
            self._dyn_body_idx = wp.array(np.asarray(dyn_body_idx, dtype=np.int32), dtype=wp.int32, device=device)
            self._dyn_mesh_rel = wp.array(np.stack(dyn_mesh_rel), dtype=wp.mat44f, device=device)
        # Per-frame transform state, built once on the first update (stable buffers, reused each frame).
        self._body_world: wp.array | None = None  # (num_bodies,) mat44f, refreshed each frame
        self._transform_out: Any = None  # cached SceneDataFormat.Matrix44 wrapping ``_body_world``
        self._records: wp.array | None = None  # cached Vulkan-backed TLAS instance-record view

        # Per-camera env origins, indexed by env id (for expressing camera poses env-local).
        env_origin_by_env = np.zeros((self._num_envs, 3), dtype=np.float32)
        for j in range(num_clones):
            e = int(env_ids[j])
            if 0 <= e < self._num_envs:
                env_origin_by_env[e] = positions[j]
        self._env_origin_by_env = env_origin_by_env

    # ---------------------------------------------------------------------- #
    # Public interface (mirrors the parts of vulkan_renderer.UsdAdapter used by the renderer)
    # ---------------------------------------------------------------------- #

    def env_origins_for(self, env_indices: np.ndarray) -> np.ndarray:
        """World-space origin [m] of each requested environment, shape ``(len(env_indices), 3)``."""
        return self._env_origin_by_env[np.clip(env_indices, 0, self._num_envs - 1)]

    def create_camera_arrays(self, *, width: int, height: int, device=None):
        return self._renderer.create_camera_arrays(
            camera_count=self.world_count, width=width, height=height, device=device
        )

    def update_transforms(self, provider: Any) -> None:
        """Refresh rigid-body transforms from the provider and rewrite the dynamic TLAS records."""
        if self._num_dynamic == 0:
            return
        if self._transform_out is None:
            # Buffers are stable across frames: the body-world array, its provider output wrapper, and
            # the Vulkan-backed instance-record view are all allocated once here and reused.
            self._body_world = wp.empty(self._num_bodies, dtype=wp.mat44f, device=self._device)
            self._transform_out = self._Matrix44()
            self._transform_out.matrices = self._body_world
            self._records = self._renderer.instance_record_array(device=self._device)

        provider.get_transforms(self._transform_out, mapping=None, allow_passthrough=False)
        wp.launch(
            kernel=_compose_clone_records,
            dim=self._num_dynamic,
            inputs=[
                self._body_world,
                self._dyn_env_inv,
                self._dyn_body_idx,
                self._dyn_mesh_rel,
                self._dyn_record,
                self._records,
            ],
            device=self._device,
        )

    def render(
        self,
        *,
        camera_transforms: wp.array,
        camera_rays: wp.array,
        color_image: wp.array | None = None,
        hdr_color_image: wp.array | None = None,
        depth_image: wp.array | None = None,
        forward_depth_image: wp.array | None = None,
        normal_image: wp.array | None = None,
        albedo_image: wp.array | None = None,
        shadows_enabled: bool | None = None,
    ) -> wp.array:
        """Render every world and copy the native output plus requested G-buffers into caller buffers."""
        # ``camera_rays`` has shape (H, W, 2); the output follows the caller buffer's device.
        height, width = int(camera_rays.shape[0]), int(camera_rays.shape[1])
        if color_image is not None:
            device = color_image.device
        else:
            device = camera_rays.device if str(camera_rays.device).startswith("cuda") else None

        buffers = (hdr_color_image, depth_image, forward_depth_image, normal_image, albedo_image)
        output_flags = 0
        for image, (_name, flag) in zip(buffers, self._aux_specs):
            if image is not None:
                output_flags |= flag

        pixels = self._renderer.render(
            camera_transforms,
            camera_rays,
            width=width,
            height=height,
            device=device,
            output_flags=output_flags,
            shadows_enabled=shadows_enabled,
        )
        if color_image is not None:
            # Fused copy + sRGB encode in one pass: the renderer outputs linear color, and Newton's
            # display output is sRGB-encoded.
            wp.launch(_srgb_encode_rgba8, dim=color_image.shape, inputs=[pixels, color_image], device=device)
        for image, (name, _flag) in zip(buffers, self._aux_specs):
            if image is None:
                continue
            source = self._renderer.aux_array(name, device=device)
            if source is not None:
                wp.copy(image, source)
        return pixels

    def close(self) -> None:
        renderer = getattr(self, "_renderer", None)
        if renderer is not None:
            renderer.close()
            self._renderer = None
        context = getattr(self, "_context", None)
        if context is not None:
            context.close()
            self._context = None

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #

    def _prim_world(self, prim_path: str) -> np.ndarray:
        """Static world transform of ``prim_path`` as a column-vector 4x4 (render convention)."""
        prim = self._stage.GetPrimAtPath(prim_path)
        gf = self._xform_cache.GetLocalToWorldTransform(prim)
        return np.array([[gf[r][c] for c in range(4)] for r in range(4)], dtype=np.float64).T.astype(np.float32)


# --------------------------------------------------------------------------- #
# Warp kernel
# --------------------------------------------------------------------------- #


@wp.kernel(enable_backward=False)
def _compose_clone_records(
    body_world: wp.array(dtype=wp.mat44f),
    env_inv: wp.array(dtype=wp.mat44f),
    body_idx: wp.array(dtype=int),
    mesh_rel_body: wp.array(dtype=wp.mat44f),
    record_idx: wp.array(dtype=int),
    instance_records: wp.array2d(dtype=wp.float32),
):
    """Write ``record = env_inv @ body_world @ mesh_rel_body`` (3x4) for each dynamic mesh."""
    i = wp.tid()
    m = wp.mul(wp.mul(env_inv[i], body_world[body_idx[i]]), mesh_rel_body[i])
    r = record_idx[i]
    instance_records[r, 0] = m[0, 0]
    instance_records[r, 1] = m[0, 1]
    instance_records[r, 2] = m[0, 2]
    instance_records[r, 3] = m[0, 3]
    instance_records[r, 4] = m[1, 0]
    instance_records[r, 5] = m[1, 1]
    instance_records[r, 6] = m[1, 2]
    instance_records[r, 7] = m[1, 3]
    instance_records[r, 8] = m[2, 0]
    instance_records[r, 9] = m[2, 1]
    instance_records[r, 10] = m[2, 2]
    instance_records[r, 11] = m[2, 3]


@wp.func
def _oetf(c: float):
    """sRGB opto-electronic transfer function (linear -> display)."""
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * wp.pow(c, 1.0 / 2.4) - 0.055


@wp.kernel(enable_backward=False)
def _srgb_encode_rgba8(src: wp.array3d(dtype=wp.uint32), dst: wp.array3d(dtype=wp.uint32)):
    """Copy packed little-endian RGBA8 pixels from ``src`` to ``dst`` with an sRGB encode (linear -> display)."""
    w, y, x = wp.tid()
    px = src[w, y, x]
    r = _oetf(wp.float32(px & wp.uint32(0xFF)) / 255.0)
    g = _oetf(wp.float32((px >> wp.uint32(8)) & wp.uint32(0xFF)) / 255.0)
    b = _oetf(wp.float32((px >> wp.uint32(16)) & wp.uint32(0xFF)) / 255.0)
    a = (px >> wp.uint32(24)) & wp.uint32(0xFF)
    ri = wp.uint32(wp.clamp(r * 255.0 + 0.5, 0.0, 255.0))
    gi = wp.uint32(wp.clamp(g * 255.0 + 0.5, 0.0, 255.0))
    bi = wp.uint32(wp.clamp(b * 255.0 + 0.5, 0.0, 255.0))
    dst[w, y, x] = ri | (gi << wp.uint32(8)) | (bi << wp.uint32(16)) | (a << wp.uint32(24))


# --------------------------------------------------------------------------- #
# Path / transform helpers
# --------------------------------------------------------------------------- #


def _recolor_instances(instances: list[Any], instance_paths: list[str], stage: Any, *, verbose: bool) -> list[Any]:
    """Override instance albedo from the bound material (OmniPBR MDL / UsdPreviewSurface), proxy-aware.

    ``build_usd_scene`` only reads ``displayColor`` / ``UsdPreviewSurface`` and does not follow
    instance-proxy prototypes, so Isaac robot visuals (instanceable, MDL-shaded) come back grey.
    :func:`_material_color` resolves the bound-material albedo directly. Shapes with no visual material
    at all (e.g. collision-only rigid objects) fall back to :data:`_DEFAULT_OBJECT_COLOR`. Textured
    surfaces keep the color ``build_usd_scene`` already loaded.
    """
    import dataclasses

    from vulkan_renderer import MeshInstance

    out = list(instances)
    recolored = 0
    for i, path in enumerate(instance_paths):
        try:
            rgb = _material_color(stage.GetPrimAtPath(path), stage)
        except Exception:
            rgb = None
        if rgb is _TEXTURED:
            continue  # textured — keep the loaded color (texture support is separate)
        if rgb is None:
            rgb = _DEFAULT_OBJECT_COLOR  # no visual material bound (e.g. collision-only object)
        inst = out[i]
        if isinstance(inst, MeshInstance):
            out[i] = MeshInstance(inst.mesh, rgb)
        else:  # PrimitiveInstance — frozen dataclass with a ``color`` field
            try:
                out[i] = dataclasses.replace(inst, color=rgb)
            except Exception:
                continue
        recolored += 1
    if verbose:
        logger.info("ClonePlanAdapter: resolved material colors for %d/%d instances", recolored, len(out))
    return out


def _material_color(prim: Any, stage: Any) -> tuple[float, float, float] | str | None:
    """Resolve a prim's albedo (sRGB) from its bound surface material, following instance proxies.

    Handles MDL ``OmniPBR`` (``diffuse_color_constant`` x ``diffuse_tint``) and ``UsdPreviewSurface``
    (``diffuseColor`` / ``baseColor``). Returns an RGB tuple for a solid color, :data:`_TEXTURED` when a
    material is bound but textured, or ``None`` when no visual material is bound at all.
    """
    if prim is None:
        return None
    try:
        from pxr import UsdShade
    except Exception:  # pragma: no cover - pxr always present at runtime
        return None
    result = None
    for material in _bound_materials(prim, stage, UsdShade):
        shader = _surface_shader(material, stage)
        if shader is None:
            continue
        color = _shader_albedo(shader, UsdShade)
        if isinstance(color, tuple):
            return color
        if color is _TEXTURED:
            result = _TEXTURED  # bound but textured; remember unless a solid color turns up
    return result


def _bound_materials(prim: Any, stage: Any, UsdShade: Any):
    """Yield the directly-bound material prims for ``prim`` (following an instance proxy to its prototype)."""
    candidates = [prim]
    try:
        if prim.IsInstanceProxy():
            candidates.append(prim.GetPrimInPrototype())
    except Exception:
        pass
    for p in candidates:
        if p is None or not p.IsValid():
            continue
        try:
            rel = UsdShade.MaterialBindingAPI(p).GetDirectBindingRel()
            targets = rel.GetTargets() if rel else []
        except Exception:
            targets = []
        for target in targets:
            material = stage.GetPrimAtPath(target)
            if material and material.IsValid():
                yield material


def _surface_shader(material: Any, stage: Any) -> Any:
    """The material's surface shader prim (MDL preferred, then the universal surface output)."""
    for output_name in ("outputs:mdl:surface", "outputs:surface"):
        attr = material.GetAttribute(output_name)
        try:
            conns = attr.GetConnections() if attr and attr.HasAuthoredConnections() else []
        except Exception:
            conns = []
        if conns:
            shader = stage.GetPrimAtPath(conns[0].GetPrimPath())
            if shader and shader.IsValid():
                return shader
    return None


def _shader_albedo(shader: Any, UsdShade: Any) -> tuple[float, float, float] | str | None:
    """Albedo (sRGB) from an ``OmniPBR`` MDL or ``UsdPreviewSurface`` shader.

    Returns an RGB tuple for a solid color, :data:`_TEXTURED` when the albedo comes from a texture
    (not applied yet), or ``None`` for an unrecognized shader.
    """
    is_mdl = bool(_attr_value(shader, "info:mdl:sourceAsset:subIdentifier")) or (
        _attr_value(shader, "info:implementationSource") == "sourceAsset"
    )
    # Albedo is returned in linear space; the renderer outputs linear and the backend sRGB-encodes the
    # final color (see ``_srgb_encode_rgba8``), matching the Newton renderer's display output.
    if is_mdl:
        texture = shader.GetAttribute("inputs:diffuse_texture")
        if texture and texture.Get() is not None:
            return _TEXTURED
        const = _attr_color(shader, "inputs:diffuse_color_constant", (0.2, 0.2, 0.2))
        tint = _attr_color(shader, "inputs:diffuse_tint", (1.0, 1.0, 1.0))
        return (const[0] * tint[0], const[1] * tint[1], const[2] * tint[2])
    if _attr_value(shader, "info:id") == "UsdPreviewSurface":
        wrapped = UsdShade.Shader(shader)
        inp = wrapped.GetInput("diffuseColor") or wrapped.GetInput("baseColor")
        try:
            connected = inp.HasConnectedSource() if inp else False
        except Exception:
            connected = False
        if inp and connected:
            return _TEXTURED
        if inp:
            value = inp.Get()
            if value is not None:
                return (float(value[0]), float(value[1]), float(value[2]))
    return None


def _attr_value(prim: Any, name: str) -> Any:
    """Authored value of ``prim``'s attribute ``name``, or ``None``."""
    attr = prim.GetAttribute(name)
    return attr.Get() if attr and attr.HasAuthoredValue() else None


def _attr_color(prim: Any, name: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    """Authored RGB value of ``prim``'s attribute ``name``, or ``default``."""
    value = _attr_value(prim, name)
    if value is None:
        return default
    return (float(value[0]), float(value[1]), float(value[2]))


def _srgb_to_linear(color: tuple[float, float, float]) -> tuple[float, float, float]:
    """Convert an sRGB display color to linear space."""
    out = []
    for v in color:
        v = max(0.0, float(v))
        out.append(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4)
    return (out[0], out[1], out[2])


# Fallback albedo (linear) for shapes with no authored visual color; sRGB-encodes to Newton's default
# object blue ``(68, 119, 170)``.
_DEFAULT_OBJECT_COLOR = _srgb_to_linear((68.0 / 255.0, 119.0 / 255.0, 170.0 / 255.0))


def _under(path: str, root: str) -> bool:
    """True if ``path`` is ``root`` or a descendant of it."""
    return path == root or path.startswith(root + "/")


def _under_any(path: str, roots: list[str]) -> bool:
    return any(_under(path, r) for r in roots)


def _source_row(path: str, sources: list[str]) -> int | None:
    """Index of the source root that owns ``path`` (deepest match), or ``None``."""
    best_row, best_len = None, -1
    for row, root in enumerate(sources):
        if _under(path, root) and len(root) > best_len:
            best_row, best_len = row, len(root)
    return best_row


_ENV_RE = re.compile(r"^/World/envs/env_[^/]+")


def _under_env(path: str) -> bool:
    """True if ``path`` lives under any ``/World/envs/env_*`` (a per-env clone slot)."""
    return _ENV_RE.match(path) is not None


def _nearest_ancestor(path: str, candidates: set[str]) -> str | None:
    """Deepest ancestor of (or) ``path`` present in ``candidates``."""
    parts = path.strip("/").split("/")
    for k in range(len(parts), 0, -1):
        prefix = "/" + "/".join(parts[:k])
        if prefix in candidates:
            return prefix
    return None


def _env_root(path: str) -> str | None:
    """The ``/World/envs/env_N`` ancestor of ``path`` (a source may sit below its env root)."""
    match = _ENV_RE.match(path)
    return match.group(0) if match else None


def _apply_template(destination: str, env_id: int) -> str:
    """Fill a ClonePlan destination template's env slot, tolerating ``{}`` or a bare root."""
    if _ENV_SLOT in destination:
        return destination.format(env_id)
    return destination.rstrip("/") + f"/env_{env_id}" if not destination.endswith(f"env_{env_id}") else destination


def _translation_inverse(position: np.ndarray) -> np.ndarray:
    """Inverse of a pure-translation 4x4 (env origins are translations)."""
    m = np.eye(4, dtype=np.float32)
    m[:3, 3] = -np.asarray(position, dtype=np.float32)
    return m

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone Vulkan ray tracing renderer for tiled camera rendering.

This backend is physics-agnostic. Rather than trusting the USD stage to carry per-environment
geometry (kit-less backends such as Newton only author the prototype environment in USD and replicate
the rest in the physics model), it follows the authoritative
:class:`~isaaclab.cloner.clone_plan.ClonePlan` via :class:`~isaaclab_vulkan.renderers.ClonePlanAdapter`:

* prototype geometry is loaded once and replicated into one Vulkan render world per environment,
* environment origins come from :attr:`ClonePlan.positions`, and
* per-frame motion is driven from the :class:`~isaaclab.scene_data.SceneDataProvider`'s world-space
  rigid-body transforms.

.. note::
    The backend assumes a Z-up stage (matching Isaac Lab) and pure-translation environment origins
    (matching :mod:`isaaclab.cloner`).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from isaaclab.renderers import BaseRenderer, RenderBufferKind, RenderBufferSpec
from isaaclab.renderers.camera_render_spec import CameraRenderSpec
from isaaclab.sim import SimulationContext
from isaaclab.utils.warp.warp_math import convert_camera_frame_orientation_convention_wp

from .clone_plan_adapter import ClonePlanAdapter
from .vulkan_warp_renderer_cfg import VulkanWarpRendererCfg

if TYPE_CHECKING:
    from isaaclab.sensors.camera.camera_data import CameraData
    from isaaclab.utils.warp import ProxyArray

logger = logging.getLogger(__name__)

# Maps each supported Isaac Lab output name to the adapter render keyword it feeds and the warp dtype
# used to reinterpret the caller's (N, H, W, C) buffer as the adapter's (N, H, W) layout.
# ``rgba`` -> packed uint32; single-channel depth -> float32; 3-channel HDR/normal -> vec3f.
_OUTPUT_TARGETS: dict[str, tuple[str, type]] = {
    str(RenderBufferKind.RGBA): ("color_image", wp.uint32),
    str(RenderBufferKind.RGB_HDR): ("hdr_color_image", wp.vec3f),
    str(RenderBufferKind.ALBEDO): ("albedo_image", wp.uint32),
    str(RenderBufferKind.NORMALS): ("normal_image", wp.vec3f),
    # Newton-style ray-hit (euclidean) distance from the optical center.
    str(RenderBufferKind.DISTANCE_TO_CAMERA): ("depth_image", wp.float32),
    # Planar depth (distance along the camera forward axis); ``depth`` aliases ``distance_to_image_plane``.
    str(RenderBufferKind.DEPTH): ("forward_depth_image", wp.float32),
    str(RenderBufferKind.DISTANCE_TO_IMAGE_PLANE): ("forward_depth_image", wp.float32),
}


class VulkanRenderData:
    """Per-camera render state for :class:`VulkanWarpRenderer`."""

    def __init__(self, width: int, height: int, device: str):
        self.width = width
        self.height = height
        self.device = device
        self.num_worlds = 0
        self.num_cameras = 0

        # Vulkan-backed camera arrays (filled in place in :meth:`VulkanWarpRenderer.update_camera`).
        self.camera_transforms: wp.array | None = None  # (num_worlds,) wp.transformf
        self.camera_rays: wp.array | None = None  # (H, W, 2) wp.vec3f
        self._rays_ready = False
        self._quat_scratch: wp.array | None = None  # (num_cameras,) wp.quatf, world->opengl buffer
        # Per-camera environment origin translation [m], used to express the camera pose env-local.
        self.cam_env_origin: wp.array | None = None  # (num_cameras,) wp.vec3f

        # Adapter render targets keyed by render keyword; each aliases a caller output buffer.
        self.targets: dict[str, wp.array | None] = {
            "color_image": None,
            "hdr_color_image": None,
            "depth_image": None,
            "forward_depth_image": None,
            "normal_image": None,
            "albedo_image": None,
        }
        # Extra (source_target_buffer, dest_view) copies for outputs that share one adapter target
        # (e.g. both ``depth`` and ``distance_to_image_plane`` map to ``forward_depth_image``).
        self.extra_copies: list[tuple[wp.array, wp.array]] = []


class VulkanWarpRenderer(BaseRenderer):
    """Vulkan ray tracing backend for tiled camera rendering."""

    RenderData = VulkanRenderData

    def __init__(self, cfg: VulkanWarpRendererCfg):
        """Pre-physics initialization: declare the scene-data requirement for this backend."""
        from isaaclab.physics.scene_data_requirements import aggregate_requirements, requirement_for_renderer_type

        self.cfg = cfg
        self._sim = SimulationContext.instance()

        # Captured in ``prepare_stage``; the scene is built lazily from these.
        self._stage: Any = None
        self._num_envs: int = 0
        self._device: str = str(self._sim.device)

        self.adapter: ClonePlanAdapter | None = None
        self._provider: Any = None

        current_req = self._sim.get_scene_data_requirements()
        renderer_req = requirement_for_renderer_type(cfg.renderer_type)
        merged = aggregate_requirements([current_req, renderer_req])
        if merged != current_req:
            self._sim.update_scene_data_requirements(merged)

    # ------------------------------------------------------------------ #
    # Capability / lifecycle
    # ------------------------------------------------------------------ #

    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        """See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.supported_output_types`."""
        return {
            RenderBufferKind.RGBA: RenderBufferSpec(4, wp.uint8),
            RenderBufferKind.RGB: RenderBufferSpec(3, wp.uint8),
            RenderBufferKind.RGB_HDR: RenderBufferSpec(3, wp.float32),
            RenderBufferKind.ALBEDO: RenderBufferSpec(3, wp.uint8),
            RenderBufferKind.DEPTH: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.DISTANCE_TO_CAMERA: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.DISTANCE_TO_IMAGE_PLANE: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.NORMALS: RenderBufferSpec(3, wp.float32),
        }

    def prepare_stage(self, stage: Any, num_envs: int) -> None:
        """Capture the stage and environment count. See :meth:`BaseRenderer.prepare_stage`."""
        self._stage = stage
        self._num_envs = num_envs

    def initialize(self) -> None:
        """Post-physics setup: resolve the scene data provider. See :meth:`BaseRenderer.initialize`."""
        self._provider = self._sim.get_scene_data_provider()

    # ------------------------------------------------------------------ #
    # Lazy scene construction (ClonePlan-driven)
    # ------------------------------------------------------------------ #

    def _ensure_scene(self) -> None:
        """Build the replicated multi-world scene from the clone plan, once."""
        if self.adapter is not None:
            return
        provider = self._provider if self._provider is not None else self._sim.get_scene_data_provider()
        self._provider = provider
        if provider is None:
            raise RuntimeError("VulkanWarpRenderer requires a SceneDataProvider but none is available.")

        plan = self._sim.get_clone_plan()
        if plan is None:
            raise RuntimeError(
                "VulkanWarpRenderer requires a clone plan (SimulationContext.get_clone_plan()) to replicate"
                " environment geometry, but none was set. This backend targets cloned multi-env scenes."
            )
        stage = self._stage if self._stage is not None else provider.usd_stage
        if stage is None:
            raise RuntimeError("VulkanWarpRenderer requires a USD stage but none was available.")

        transform_paths = list(provider.backend.transform_paths or [])
        self.adapter = ClonePlanAdapter(
            stage,
            plan,
            transform_paths,
            device=self._device,
            num_envs=self._num_envs,
            include_guides=self.cfg.include_guides,
            verbose=self.cfg.verbose,
            ambient=self.cfg.ambient,
            shadows_enabled=self.cfg.enable_shadows,
        )
        logger.info(
            "VulkanWarpRenderer: built %d render worlds for %d environments", self.adapter.world_count, self._num_envs
        )

    # ------------------------------------------------------------------ #
    # Per-camera setup
    # ------------------------------------------------------------------ #

    def create_render_data(self, spec: CameraRenderSpec) -> VulkanRenderData:
        """See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.create_render_data`."""
        self._ensure_scene()
        width = int(getattr(spec.cfg, "width", 100))
        height = int(getattr(spec.cfg, "height", 100))
        device = spec.device
        rd = VulkanRenderData(width, height, device)
        rd.num_worlds = self.adapter.world_count
        rd.num_cameras = spec.num_instances
        rd.camera_transforms, rd.camera_rays = self.adapter.create_camera_arrays(
            width=width, height=height, device=device
        )
        # One camera pose per environment; env origins come from the clone plan (the camera prim only
        # exists in the prototype env, so the stage cannot supply per-env origins here).
        origins = self.adapter.env_origins_for(np.arange(spec.num_instances))
        rd.cam_env_origin = wp.array(origins, dtype=wp.vec3f, device=device)
        if rd.num_cameras != rd.num_worlds:
            logger.warning(
                "VulkanWarpRenderer: camera instances (%d) != world count (%d); tiles will misalign.",
                rd.num_cameras,
                rd.num_worlds,
            )
        return rd

    def set_outputs(self, render_data: VulkanRenderData, output_data: dict[str, ProxyArray]) -> None:
        """Bind caller output buffers as adapter render targets. See :meth:`BaseRenderer.set_outputs`."""
        rd = render_data
        for key in rd.targets:
            rd.targets[key] = None
        rd.extra_copies = []
        shape = (rd.num_worlds, rd.height, rd.width)
        for output_name, proxy in output_data.items():
            if output_name == str(RenderBufferKind.RGB):
                # ``rgb`` is derived by the camera from the ``rgba`` buffer; nothing to bind here.
                continue
            target = _OUTPUT_TARGETS.get(output_name)
            if target is None:
                logger.warning("VulkanWarpRenderer - output type %s is not yet supported", output_name)
                continue
            target_kw, dtype = target
            view = self._view(proxy, dtype, shape)
            if rd.targets[target_kw] is None:
                rd.targets[target_kw] = view
            else:
                # Another output already claimed this adapter target (e.g. depth + distance_to_image_plane);
                # fill it once and copy into the extra destination in ``read_output``.
                rd.extra_copies.append((rd.targets[target_kw], view))

    @staticmethod
    def _view(proxy: ProxyArray, dtype: type, shape: tuple[int, ...]) -> wp.array:
        """Reinterpret the caller's ``(N, H, W, C)`` buffer as a ``(N, H, W)`` array of ``dtype`` (no copy)."""
        arr = proxy.warp
        return wp.array(ptr=arr.ptr, dtype=dtype, shape=shape, device=arr.device, copy=False)

    # ------------------------------------------------------------------ #
    # Per-frame update / render
    # ------------------------------------------------------------------ #

    def update_transforms(self) -> None:
        """Drive rendered geometry from the scene data provider. See :meth:`BaseRenderer.update_transforms`."""
        self._ensure_scene()
        self.adapter.update_transforms(self._provider)

    def update_geometries(self) -> None:
        """No-op: the Vulkan backend only supports rigid transforms. See :meth:`BaseRenderer.update_geometries`."""
        return

    def update_camera(
        self,
        render_data: VulkanRenderData,
        positions: ProxyArray,
        orientations: ProxyArray,
        intrinsics: ProxyArray,
    ) -> None:
        """See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.update_camera`."""
        rd = render_data
        n = rd.num_cameras
        if rd._quat_scratch is None:
            rd._quat_scratch = wp.empty(n, dtype=wp.quatf, device=rd.device)
        # The adapter's camera-local rays use the OpenGL convention (forward -Z, up +Y).
        convert_camera_frame_orientation_convention_wp(
            src=positions_or_warp(orientations),
            dst=rd._quat_scratch,
            origin="world",
            target="opengl",
            device=rd.device,
        )
        wp.launch(
            kernel=_pack_camera_transforms,
            dim=n,
            inputs=[positions_or_warp(positions), rd._quat_scratch, rd.cam_env_origin, rd.camera_transforms],
            device=rd.device,
        )
        if not rd._rays_ready:
            # Pinhole camera rays share one set across worlds (identical intrinsics assumed).
            fy = float(intrinsics.torch[0, 1, 1].item())
            fov_y = 2.0 * math.atan(rd.height / (2.0 * fy))
            rd.camera_rays.assign(_build_camera_rays(rd.width, rd.height, fov_y))
            rd._rays_ready = True

    def render(self, render_data: VulkanRenderData) -> None:
        """See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.render`."""
        rd = render_data
        self.adapter.render(
            camera_transforms=rd.camera_transforms,
            camera_rays=rd.camera_rays,
            color_image=rd.targets["color_image"],
            hdr_color_image=rd.targets["hdr_color_image"],
            depth_image=rd.targets["depth_image"],
            forward_depth_image=rd.targets["forward_depth_image"],
            normal_image=rd.targets["normal_image"],
            albedo_image=rd.targets["albedo_image"],
            shadows_enabled=self.cfg.enable_shadows,
        )

    def read_output(self, render_data: VulkanRenderData, camera_data: CameraData) -> None:
        """Fan out shared adapter targets into their remaining destinations. See :meth:`BaseRenderer.read_output`.

        The adapter writes directly into the caller buffers aliased in :meth:`set_outputs`, so only
        outputs that shared a single adapter target still need a copy.
        """
        for source, dest in render_data.extra_copies:
            if source.ptr != dest.ptr:
                wp.copy(dest, source)

    def cleanup(self, render_data: VulkanRenderData | None) -> None:
        """See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.cleanup`."""
        return

    def close(self) -> None:
        """Release the adapter and its Vulkan context. See :meth:`BaseRenderer.close`."""
        if self.adapter is not None:
            self.adapter.close()
            self.adapter = None


# --------------------------------------------------------------------------- #
# Warp kernels
# --------------------------------------------------------------------------- #


@wp.kernel(enable_backward=False)
def _pack_camera_transforms(
    positions: wp.array(dtype=wp.vec3f),
    orientations: wp.array(dtype=wp.quatf),
    env_origin: wp.array(dtype=wp.vec3f),
    out_transforms: wp.array(dtype=wp.transformf),
):
    """Pack per-camera env-local position + OpenGL-convention orientation into a Vulkan camera transform."""
    i = wp.tid()
    out_transforms[i] = wp.transformf(positions[i] - env_origin[i], orientations[i])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def positions_or_warp(proxy: Any) -> wp.array:
    """Return the underlying warp array of a ``ProxyArray`` (or the value itself if already an array)."""
    return proxy.warp if hasattr(proxy, "warp") else proxy


def _build_camera_rays(width: int, height: int, fov_y: float) -> np.ndarray:
    """Camera-local pinhole ray directions ``(H, W, 2, 3)`` (origin at 0, forward -Z, up +Y)."""
    aspect = width / height
    tan_half_fov = np.float32(math.tan(fov_y * 0.5))
    px = (
        ((np.arange(width, dtype=np.float32) + 0.5) / np.float32(width) * 2.0 - 1.0) * np.float32(aspect) * tan_half_fov
    )
    py = (1.0 - (np.arange(height, dtype=np.float32) + 0.5) / np.float32(height) * 2.0) * tan_half_fov
    directions = (
        np.array((0.0, 0.0, -1.0), dtype=np.float32).reshape(1, 1, 3)
        + px.reshape(1, width, 1) * np.array((1.0, 0.0, 0.0), dtype=np.float32).reshape(1, 1, 3)
        + py.reshape(height, 1, 1) * np.array((0.0, 1.0, 0.0), dtype=np.float32).reshape(1, 1, 3)
    )
    directions /= np.linalg.norm(directions, axis=2, keepdims=True)
    rays = np.zeros((height, width, 2, 3), dtype=np.float32)
    rays[:, :, 1, :] = directions
    return rays

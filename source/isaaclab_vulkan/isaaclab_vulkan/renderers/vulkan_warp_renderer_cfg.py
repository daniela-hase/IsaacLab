# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Vulkan ray tracing renderer."""

from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.utils.configclass import configclass


@configclass
class VulkanWarpRendererCfg(RendererCfg):
    """Configuration for the Vulkan ray tracing renderer.

    This backend replicates prototype geometry into one render world per environment following the
    :class:`~isaaclab.cloner.clone_plan.ClonePlan`, and drives per-frame rigid-body motion from the
    :class:`~isaaclab.scene_data.SceneDataProvider`. It is physics-backend agnostic: it consumes
    world-space body transforms rather than a Newton model.
    """

    renderer_type: str = "vulkan_rt"
    """Type identifier for the Vulkan ray tracing renderer."""

    enable_shadows: bool = True
    """Enable shadow rays for directional lights."""

    ambient: float = 0.1
    """Constant ambient lighting term added to every surface."""

    include_guides: bool = False
    """Also render ``guide``/``proxy`` purpose prims (e.g. collision shapes)."""

    verbose: bool = True
    """Print a one-time summary of what was loaded from the stage when the adapter is built."""

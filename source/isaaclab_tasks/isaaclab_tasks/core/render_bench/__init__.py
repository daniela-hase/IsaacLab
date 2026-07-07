# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render-only benchmark scenes for OVRTX / Newton-Warp comparison."""

import gymnasium as gym

gym.register(
    id="Isaac-RenderBench-Anymal-Tabletop-v0",
    entry_point=f"{__name__}.render_bench_env:RenderBenchEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.render_bench_env_cfg:RenderBenchAnymalTabletopEnvCfg"},
)

gym.register(
    id="Isaac-RenderBench-Franka-Tabletop-v0",
    entry_point=f"{__name__}.render_bench_env:RenderBenchEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.render_bench_env_cfg:RenderBenchFrankaTabletopEnvCfg"},
)

gym.register(
    id="Isaac-RenderBench-Kitchen-Sink-v0",
    entry_point=f"{__name__}.render_bench_env:RenderBenchEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.render_bench_env_cfg:RenderBenchKitchenSinkEnvCfg"},
)

gym.register(
    id="Isaac-RenderBench-Anymal-Terrain-v0",
    entry_point=f"{__name__}.render_bench_env:RenderBenchEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.render_bench_env_cfg:RenderBenchAnymalTerrainEnvCfg"},
)

gym.register(
    id="Isaac-RenderBench-Franka-Lift-Cube-v0",
    entry_point=f"{__name__}.render_bench_env:RenderBenchEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.render_bench_env_cfg:RenderBenchFrankaLiftCubeEnvCfg"},
)

gym.register(
    id="Isaac-RenderBench-Franka-Cabinet-v0",
    entry_point=f"{__name__}.render_bench_env:RenderBenchEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.render_bench_env_cfg:RenderBenchFrankaCabinetEnvCfg"},
)

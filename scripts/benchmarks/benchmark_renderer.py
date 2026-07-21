# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import fnmatch
import os
import shutil
import site
import sqlite3
import statistics
import subprocess
import sys

PROFILES = [
    {
        "name": "ovrtx_constant_diffuse_oldpipe",
        "preset": "ovrtx_renderer,simple_shading_constant_diffuse",
        "settings": {"min-pipe": False},
    },
    {
        "name": "ovrtx_constant_diffuse_newpipe",
        "preset": "ovrtx_renderer,simple_shading_constant_diffuse",
        "settings": {"min-pipe": True},
    },
    {
        "name": "ovrtx_diffuse_mdl_oldpipe",
        "preset": "ovrtx_renderer,simple_shading_diffuse_mdl",
        "settings": {"min-pipe": False},
    },
    {
        "name": "ovrtx_diffuse_mdl_newpipe",
        "preset": "ovrtx_renderer,simple_shading_diffuse_mdl",
        "settings": {"min-pipe": True},
    },
    {
        "name": "ovrtx_full_mdl_oldpipe",
        "preset": "ovrtx_renderer,simple_shading_full_mdl",
        "settings": {"min-pipe": False},
    },
    {
        "name": "ovrtx_full_mdl_newpipe",
        "preset": "ovrtx_renderer,simple_shading_full_mdl",
        "settings": {"min-pipe": True},
    },
    {"name": "newton_lbvh_lbvh", "preset": "newton_renderer,rgb", "settings": {"tlas": "lbvh", "blas": "lbvh"}},
    {"name": "newton_lbvh_sah", "preset": "newton_renderer,rgb", "settings": {"tlas": "lbvh", "blas": "sah"}},
    {"name": "newton_lbvh_cubql", "preset": "newton_renderer,rgb", "settings": {"tlas": "lbvh", "blas": "cubql"}},
    {"name": "newton_sah_lbvh", "preset": "newton_renderer,rgb", "settings": {"tlas": "sah", "blas": "lbvh"}},
    {"name": "newton_sah_sah", "preset": "newton_renderer,rgb", "settings": {"tlas": "sah", "blas": "sah"}},
    {"name": "newton_sah_cubql", "preset": "newton_renderer,rgb", "settings": {"tlas": "sah", "blas": "cubql"}},
]

TASK_NAME = "Isaac-RenderBenchmark-Franka-Cabinet"
OUTPUT_PATH = "benchmarks"
FRAME_PADDING = 5


def get_profile(name: str) -> dict | None:
    for profile in PROFILES:
        if profile["name"] == name:
            return profile
    return None


def parse_profile(filename: str, renderer: str, num_frames: int):
    cursor = sqlite3.connect(filename)

    if renderer == "ovrtx":
        steps = cursor.execute(
            "SELECT start,end FROM NVTX_EVENTS WHERE text='ovrtx_step_execute' ORDER BY start"
        ).fetchall()
        rid_row = cursor.execute("SELECT id FROM StringIds WHERE value='RTX Rendering'").fetchone()
        if not rid_row:
            return None

        out = []
        for s, e in cursor.execute(
            "SELECT start,end FROM VULKAN_WORKLOAD WHERE textId=? ORDER BY start", (rid_row[0],)
        ).fetchall():
            for i, (ss, se) in enumerate(steps):
                if ss <= s <= se:
                    if FRAME_PADDING < i + 1 <= (num_frames + FRAME_PADDING):
                        out.append((e - s) / 1e6)
                    break

    if renderer == "newton_renderer":
        ranges = cursor.execute("""
            SELECT n.start, n.end FROM NVTX_EVENTS n
            JOIN StringIds s ON s.id=n.textId
            WHERE s.value LIKE '%NewtonWarpRenderer.render%'
            ORDER BY n.start
        """).fetchall()

        out = []
        for i, (rs, re) in enumerate(ranges):
            next_rs = ranges[i + 1][0] if i + 1 < len(ranges) else float("inf")

            row = cursor.execute(
                """
                SELECT MIN(k.start), MAX(k.end) FROM CUPTI_ACTIVITY_KIND_KERNEL k
                JOIN StringIds s ON s.id=k.demangledName
                WHERE k.start>=? AND k.start<? AND s.value LIKE '%render_megakernel%'
            """,
                (rs, next_rs),
            ).fetchone()

            if row[0] is None:
                continue

            if FRAME_PADDING < i + 1 <= (num_frames + FRAME_PADDING):
                out.append((row[1] - row[0]) / 1e6)

    if out:
        return {
            "size": len(out),
            "median": statistics.median(out),
            "mean": statistics.mean(out),
            "min": min(out),
            "max": max(out),
            "stdev": statistics.stdev(out) if len(out) > 1 else 0,
        }
    return None


def run_profile(profile: dict, num_frames: int, num_envs: int, resolution: int, save_image: bool, verbose: bool):
    warp_cache_path = os.path.abspath(os.path.join(OUTPUT_PATH, "warp-cache"))

    env = {
        "NVTX_PROFILE_PYTHON": "1",
        "NVTX_PROFILE_INCLUDE": "isaaclab,newton,warp,rsl_rl",
        "NEWTON_USE_CUDA_GRAPH": "0",
        "BENCHMARK_SAVE_IMAGE": "1" if save_image else "0",
        "BENCHMARK_RENDER_RESOLUTION": f"{resolution}",
        "WARP_CACHE_PATH": warp_cache_path,
    }

    if os.path.exists(warp_cache_path):
        shutil.rmtree(warp_cache_path)

    profile_name: str = profile["name"]
    preset: str = profile["preset"]
    renderer: str = preset.split(",")[0]
    trace: str = "nvtx"

    if renderer == "ovrtx_renderer":
        trace = "nvtx,vulkan,vulkan-annotations"
        env["LD_PRELOAD"] = os.path.join(
            site.getsitepackages()[0], "ovrtx/bin/plugins/omni.client.lib/libomniclient.so"
        )
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["OMNI_KIT_ACCEPT_EULA"] = "YES"
        env["OVRTX_rtx_post_tonemap_op"] = "0"
        env["OVRTX_rtx_minimal_useMinimalPipeline"] = "1" if profile["settings"]["min-pipe"] else "0"
        env["OVRTX_app_profilerBackend"] = "nvtx"
        env["OVRTX_app_profileFromStart"] = "true"
        env["OVRTX_app_profilerMask"] = "1"

    if renderer == "newton_renderer":
        trace = "nvtx,cuda"
        env["NEWTON_BVH_SCENE"] = profile["settings"]["tlas"]
        env["NEWTON_BVH_GEOMETRY"] = profile["settings"]["blas"]

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    profile_filename = os.path.join(OUTPUT_PATH, profile_name + ".nsys-rep")
    log_filename = os.path.join(OUTPUT_PATH, profile["name"] + ".log")

    cmd = [
        "nsys",
        "profile",
        "--output",
        profile_filename,
        "--force-overwrite",
        "true",
        "--trace",
        trace,
        "--vulkan-gpu-workload",
        "individual",
        "--python-functions-trace",
        "scripts/benchmarks/nsys_trace.json",
        "./isaaclab.sh",
        "-p",
        "scripts/benchmarks/runtime.py",
        "--task",
        TASK_NAME,
        "--headless",
        "--enable_cameras",
        "--num_envs",
        f"{num_envs}",
        "--num_frames",
        f"{num_frames + FRAME_PADDING * 2}",
        "--output_path",
        OUTPUT_PATH,
        f"presets={preset}",
    ]

    with open(log_filename, "w") as file:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=dict(os.environ) | env, text=True
        )

        for line in process.stdout:
            file.write(line)
            if verbose:
                sys.stdout.write(f"\x1b[90m{line}\x1b[0m")
                sys.stdout.flush()

        return_code = process.wait()
        if return_code != 0:
            print(f"Failed with exit code {process.returncode}, see {log_filename} for details.")
            return False

    subprocess.run(
        [
            "nsys",
            "stats",
            "--force-export",
            "true",
            "--report",
            "cuda_kern_exec_sum",
            "--format",
            "csv",
            profile_filename,
        ],
        stdout=subprocess.PIPE,
    )
    return parse_profile(profile_filename.replace(".nsys-rep", ".sqlite"), renderer, num_frames)


parser = argparse.ArgumentParser("IsaacLab Benchmark: Sweep Franka Cabinet")
parser.add_argument("--num-frames", type=int, default="20", help="Number of frames to render")
parser.add_argument("--num-envs", type=int, default="1024", help="Number of environments to render")
parser.add_argument("--resolution", type=int, default="256", help="Render resolution")
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
parser.add_argument("-s", "--save-image", action="store_true", help="Save image for debugging purposes")
parser.add_argument("profile", nargs="*", help="Profiles to run")

args = parser.parse_args()

if not args.profile:
    print("Available profiles:")
    for profile in PROFILES:
        print("  " + profile["name"])
    exit(0)

profiles = set()

for profile_name in args.profile:
    found_match = False
    for profile in PROFILES:
        if fnmatch.fnmatch(profile["name"], profile_name):
            profiles.add(profile["name"])
            found_match = True

    if not found_match:
        print(f"Not profile found matching: {profile_name}")
        exit(-1)

all_results = {}
for profile_name in profiles:
    if profile := get_profile(profile_name):
        print(f"profile: {profile['name']}")
        print(f"  preset: {profile['preset']}")
        for key, value in profile["settings"].items():
            print(f"  {key}: {value}")

        try:
            all_results[profile_name] = run_profile(
                profile, args.num_frames, args.num_envs, args.resolution, args.save_image, args.verbose
            )
        except KeyboardInterrupt:
            break

        if results := all_results[profile_name]:
            print(f"    size: {results['size']}")
            for key, value in results.items():
                if key != "size":
                    print(f"    {key}: {value:.2f}ms")
        print("")

benchmark_faled = False
print("")
print(
    "| PROFILE                                  | SIZE |  PIXEL / SEC |    MEDIAN    |     MEAN     |     MIN      |     MAX      |    STDEV     |"  # noqa: E501
)
print(
    "|------------------------------------------|------|--------------|--------------|--------------|--------------|--------------|--------------|"  # noqa: E501
)
for profile_name, results in all_results.items():
    if results:
        gpxs = 1000 / results['median'] * (1024 * 256 * 256) / 1000000000
        print(
            f"| {profile_name:<40} | {results['size']:>4} | {gpxs:>6.2f} Gpx/s | {results['median']:>10.2f}ms | {results['mean']:>10.2f}ms | {results['min']:>10.2f}ms | {results['max']:>10.2f}ms | {results['stdev']:>10.2f}ms |"  # noqa: E501
        )
    else:
        print(
            f"| {profile_name:<40} |                                             FAILED                                            |"
        )
        benchmark_faled = True
print(
    "|------------------------------------------|------|--------------|--------------|--------------|--------------|--------------|--------------|"  # noqa: E501
)
print("")
if benchmark_faled:
    exit(-1)
exit(0)

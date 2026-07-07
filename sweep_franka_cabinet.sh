#!/usr/bin/env bash
# 8-profile sweep for Isaac-RenderBench-Franka-Cabinet-v0:
#   OVRTX (old + new pipeline) × {constant_diffuse, diffuse_mdl, full_mdl}  =  6
#   Newton-Warp + cuBQL BLAS × {TLAS=LBVH (default), TLAS=SAH}              =  2
#   --------------------------------------------------------------------------
# All at 1024 envs × 256×256 × 30 frames, median over steady-state #6-25.
#
# Key fixes vs sweep_franka_cabinet.sh:
#   1. --python-functions-trace=scripts/benchmarks/nsys_trace.json so the
#      NewtonWarpRenderer.render NVTX range is emitted and nvtx_gpu_proj_sum
#      can project it onto the GPU timeline for the Warp extraction.
#   2. TLAS=SAH variant: in-place sed of newton/_src/geometry/bvh.py to pass
#      ``constructor="sah"`` to ``wp.Bvh`` for the duration of that run,
#      then revert. ``constructor="lbvh"`` (the default) is the baseline.
set -e
source .venv/bin/activate

# Clean-checkout env fixes (md §4): EULA + OVRTX-bundled omni.client preload.
# sudo strips LD_PRELOAD, so it must be passed on the sudo command line (below).
LDP="$(python -c 'import site;print(site.getsitepackages()[0])')/ovrtx/bin/plugins/omni.client.lib/libomniclient.so"

OUTDIR=artifacts
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

TASK=Isaac-RenderBench-Franka-Cabinet-v0

BVH_TLAS_PY=".venv/lib/python3.12/site-packages/newton/_src/sim/model.py"
BVH_TLAS_SEARCH_LINE='.*self.bvh_shapes = wp.Bvh(.*'
BVH_TLAS_LBVH_LINE='        self.bvh_shapes = wp.Bvh(lowers, uppers, constructor="lbvh", groups=groups)'
BVH_TLAS_SAH_LINE='        self.bvh_shapes = wp.Bvh(lowers, uppers, constructor="sah", groups=groups)'

BVH_BLAS_PY=".venv/lib/python3.12/site-packages/newton/_src/geometry/types.py"
BVH_BLAS_SEARCH_LINE='.*self.mesh = wp.Mesh(.*'
BVH_BLAS_LBVH_LINE='            self.mesh = wp.Mesh(points=pos, velocities=vel, indices=indices, bvh_constructor="lbvh")'
BVH_BLAS_SAH_LINE='            self.mesh = wp.Mesh(points=pos, velocities=vel, indices=indices, bvh_constructor="sah")'
BVH_BLAS_CUBQL_LINE='            self.mesh = wp.Mesh(points=pos, velocities=vel, indices=indices, bvh_constructor="cubql")'

# Newton physics CUDA-graph toggle. On this 8x B200 box, nsys CUPTI (cuda trace,
# used only by the Warp profiles) DEADLOCKS while Warp captures the physics CUDA
# graph. Disabling the graph (eager physics) lets CUPTI trace the render
# megakernel; the kernel's GPU duration is unchanged. Reverted at exit.
NEWTON_CFG="source/isaaclab_newton/isaaclab_newton/physics/newton_manager_cfg.py"
graph_on()  { sed -i 's|^    use_cuda_graph: bool = False.*|    use_cuda_graph: bool = True|' "$NEWTON_CFG"; }
graph_off() { sed -i 's|^    use_cuda_graph: bool = True|    use_cuda_graph: bool = False  # TEMP (sweep: CUPTI trace)|' "$NEWTON_CFG"; }
trap graph_on EXIT

# Always start from LBVH (default) baseline; any SAH leftover from a prior
# interrupted run gets normalized back here.
sed -i "s|${BVH_TLAS_SEARCH_LINE}|${BVH_TLAS_LBVH_LINE}|" "$BVH_TLAS_PY"
sed -i "s|${BVH_BLAS_SEARCH_LINE}|${BVH_BLAS_LBVH_LINE}|" "$BVH_BLAS_PY"

# Profile table: tag, renderer-kind, shading-or-empty, use_min_pipe-or-empty, tlas-or-empty
PROFILES=(
#    "constant_diffuse_oldpipe ovrtx       simple_shading_constant_diffuse false ''"
#    "constant_diffuse_newpipe ovrtx       simple_shading_constant_diffuse true  ''"
#    "diffuse_mdl_oldpipe      ovrtx       simple_shading_diffuse_mdl      false ''"
#    "diffuse_mdl_newpipe      ovrtx       simple_shading_diffuse_mdl      true  ''"
#    "full_mdl_oldpipe         ovrtx       simple_shading_full_mdl         false ''"
#    "full_mdl_newpipe         ovrtx       simple_shading_full_mdl         true  ''"
    "warp_lbvh_lbvh            newton_warp ''                              ''    lbvh  lbvh"
    "warp_lbvh_sah             newton_warp ''                              ''    lbvh  sah"
    "warp_lbvh_cubql           newton_warp ''                              ''    lbvh  cubql"
    "warp_sah_lbvh             newton_warp ''                              ''    sah   lbvh"
    "warp_sah_sah              newton_warp ''                              ''    sah   sah"
    "warp_sah_cubql            newton_warp ''                              ''    sah   cubql"
)

presets_for() {
    case "$1" in
        ovrtx)       echo "ovrtx_renderer,${2}";;
        newton_warp) echo "newton_renderer,rgb";;
    esac
}

# --- Profiling sweep --------------------------------------------------------
echo "==============================================================="
echo "8-profile sweep (1024 envs × 256² × 30 frames, median #6-25)"
echo "==============================================================="
for row in "${PROFILES[@]}"; do
    set -- $row
    TAG=$1; KIND=$2; SHADING=$3; USE_MIN=$4; TLAS=$5; BLAS=$6
    PRESETS=$(presets_for "$KIND" "$SHADING")
    OUT="$OUTDIR/renderbench_${TAG}.nsys-rep"
    LOG="$OUTDIR/log_${TAG}.log"

    # Per-profile env + trace flags
    EXTRA_ENV="CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES"
    TRACE="nvtx,vulkan,vulkan-annotations"
    if [ "$KIND" = "ovrtx" ]; then
        EXTRA_ENV="CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES LD_PRELOAD=$LDP OVRTX_rtx_post_tonemap_op=0 OVRTX_rtx_minimal_useMinimalPipeline=$USE_MIN OVRTX_app_profilerBackend=nvtx OVRTX_app_profileFromStart=true OVRTX_app_profilerMask=1"
        graph_on
    else
        TRACE="nvtx,cuda"
        graph_off   # CUPTI cannot trace through Warp's physics graph capture
        # TLAS switch (in-place sed of bvh.py)
        if [ "$TLAS" = "sah" ]; then
            sed -i "s|${BVH_TLAS_SEARCH_LINE}|${BVH_TLAS_SAH_LINE}|" "$BVH_TLAS_PY"
        else
            sed -i "s|${BVH_TLAS_SEARCH_LINE}|${BVH_TLAS_LBVH_LINE}|" "$BVH_TLAS_PY"
        fi

        if [ "$BLAS" = "sah" ]; then
            sed -i "s|${BVH_BLAS_SEARCH_LINE}|${BVH_BLAS_SAH_LINE}|" "$BVH_BLAS_PY"
        elif [ "$BLAS" = "cubql" ]; then
            sed -i "s|${BVH_BLAS_SEARCH_LINE}|${BVH_BLAS_CUBQL_LINE}|" "$BVH_BLAS_PY"
        else
            sed -i "s|${BVH_BLAS_SEARCH_LINE}|${BVH_BLAS_LBVH_LINE}|" "$BVH_BLAS_PY"
        fi

        # Clear Warp kernel cache so the JIT picks up the bvh.py change
        rm -rf ~/.cache/warp 2>/dev/null || true
    fi

    echo "=== [$(date +%H:%M:%S)] $TAG  presets=$PRESETS  tlas=$TLAS,  blas=$BLAS  use_min=$USE_MIN ==="
    env PATH="$PATH" VIRTUAL_ENV="$VIRTUAL_ENV" \
        NVTX_PROFILE_PYTHON=1 NVTX_PROFILE_INCLUDE=isaaclab,newton,warp,rsl_rl \
        ${EXTRA_ENV} \
        nsys profile --output="$OUT" --force-overwrite=true \
            --trace="$TRACE" \
            --vulkan-gpu-workload=individual \
            --python-functions-trace=scripts/benchmarks/nsys_trace.json \
            ./isaaclab.sh -p scripts/benchmarks/benchmark_non_rl.py \
                --task="$TASK" \
                --headless --enable_cameras \
                --num_envs=1024 --num_frames=30 \
                --benchmark_backend=summary \
                "presets=$PRESETS" \
                > "$LOG" 2>&1 || true

    chown $USER:$USER "$OUT" 2>/dev/null || true
    rm -f "${OUT%.nsys-rep}.sqlite"
    # Export sqlite for the extraction step
    nsys stats --force-export=true --report cuda_kern_exec_sum --format csv "$OUT" >/dev/null 2>&1 || true

    # --- Extract median GPU frametime -------------------------------------
    python3 - "$TAG" "$OUT" "$KIND" << 'PYEOF'
import sqlite3, statistics, sys, os
tag, rep, kind = sys.argv[1], sys.argv[2], sys.argv[3]
sql = rep.replace('.nsys-rep', '.sqlite')
if not os.path.exists(sql):
    print(f"SWEEP_RESULT tag={tag} median=NA (no sqlite produced)"); sys.exit(0)
c = sqlite3.connect(sql)
if kind == "ovrtx":
    steps = c.execute("SELECT start,end FROM NVTX_EVENTS WHERE text='ovrtx_step_execute' ORDER BY start").fetchall()
    rid_row = c.execute("SELECT id FROM StringIds WHERE value='RTX Rendering'").fetchone()
    if not rid_row:
        print(f"SWEEP_RESULT tag={tag} median=NA (no RTX Rendering events; steps={len(steps)})"); sys.exit(0)
    rid = rid_row[0]
    out = []
    for s, e in c.execute("SELECT start,end FROM VULKAN_WORKLOAD WHERE textId=? ORDER BY start", (rid,)).fetchall():
        for i, (ss, se) in enumerate(steps):
            if ss <= s <= se:
                if 6 <= i+1 <= 25: out.append((e-s)/1e6)
                break
else:
    # Warp: project NewtonWarpRenderer.render NVTX onto its render_megakernel kernel
    ranges = c.execute("""
        SELECT n.start, n.end FROM NVTX_EVENTS n
        JOIN StringIds s ON s.id=n.textId
        WHERE s.value LIKE '%NewtonWarpRenderer.render%'
        ORDER BY n.start
    """).fetchall()
    if not ranges:
        # Fallback: raw render_megakernel kernel durations (no NVTX framing needed)
        rows = c.execute("""
            SELECT (k.end-k.start)/1e6 ms FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id=k.shortName
            WHERE s.value LIKE '%render_megakernel%'
            ORDER BY k.start
        """).fetchall()
        out = [r[0] for r in rows[5:25]] if rows else []
        suffix = " (fallback: raw kernel timings, no NVTX framing)"
    else:
        out = []
        for i, (rs, re) in enumerate(ranges):
            next_rs = ranges[i+1][0] if i+1 < len(ranges) else float('inf')
            row = c.execute("""
                SELECT MIN(k.start), MAX(k.end) FROM CUPTI_ACTIVITY_KIND_KERNEL k
                JOIN StringIds s ON s.id=k.shortName
                WHERE k.start>=? AND k.start<? AND s.value LIKE '%render_megakernel%'
            """, (rs, next_rs)).fetchone()
            if row[0] is None: continue
            if 6 <= i+1 <= 25: out.append((row[1]-row[0])/1e6)
        suffix = ""
if out:
    print(f"SWEEP_RESULT tag={tag} median={statistics.median(out):.3f}ms n={len(out)} "
          f"mean={statistics.mean(out):.3f}ms min={min(out):.3f}ms max={max(out):.3f}ms "
          f"stdev={(statistics.stdev(out) if len(out)>1 else 0):.3f}ms{suffix if kind=='newton_warp' else ''}")
else:
    print(f"SWEEP_RESULT tag={tag} median=NA (no steady-state hits)")
PYEOF
done

# Always restore LBVH default after the sweep
sed -i "s|${BVH_TLAS_SEARCH_LINE}|${BVH_TLAS_LBVH_LINE}|" "$BVH_TLAS_PY"
sed -i "s|${BVH_BLAS_SEARCH_LINE}|${BVH_BLAS_LBVH_LINE}|" "$BVH_BLAS_PY"

echo
echo "==============================================================="
echo "SWEEP COMPLETE — profiles in $OUTDIR"
echo "==============================================================="

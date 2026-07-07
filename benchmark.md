# Setup Benchmark

```bash
uv venv
uv pip install cmake==3.31.6
uv pip install nvtx ovrtx==0.4.0.319531 'usd-core==25.5.1' --index-url https://urm.nvidia.com/artifactory/api/pypi/ct-omniverse-pypi/simple --extra-index-url https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi/simple

source .venv/bin/activate
./isaaclab.sh -i
```

# Run Benchmark
```bash
./sweep_franka_cabinet.sh
```
# Setup Benchmark

```bash
git clone -b dev/benchmark-franka-cabinet https://github.com/daniela-hase/IsaacLab.git

uv venv --python 3.12
uv pip install cmake==3.31.6

source .venv/bin/activate
./isaaclab.sh -i
```

# Install nsys (Ubuntu)
```bash
sudo echo "deb [signed-by=/etc/apt/keyrings/nvidia-nsys.gpg] http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | sudo tee /etc/apt/sources.list.d/nvidia-devtools.list
curl -fsSL http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | sudo gpg --dearmor -o /etc/apt/keyrings/nvidia-nsys.gpg

sudo apt update
sudo apt install -y nsight-systems-cli
```

# Run Benchmark
```bash
source .venv/bin/activate
python scripts/benchmarks/benchmark_renderer.py newton_sah_cubql

# or

uv run --no-sync python scripts/benchmarks/benchmark_renderer.py newton_sah_cubql
```

# Newton SAH/cuBQL Benchmark Results (1024 envs, 256x256)

| GPU                                      | SIZE |  PIXEL / SEC |    MEDIAN    |     MEAN     |     MIN      |     MAX      |    STDEV     |
|------------------------------------------|------|--------------|--------------|--------------|--------------|--------------|--------------|
| NVIDIA RTX Pro 6000                      |   20 |   3.13 Gpx/s |      21.42ms |      21.48ms |      20.78ms |      22.09ms |       0.39ms |
| NVIDIA B200                              |   20 |   2.42 Gpx/s |      27.70ms |      27.79ms |      26.80ms |      28.57ms |       0.51ms |
| NVIDIA GH200                             |   20 |   1.26 Gpx/s |      53.25ms |      53.40ms |      51.13ms |      54.79ms |       0.98ms |
| NVIDIA RTX Pro 6000 1/4 MIG 1g.24gb      |   20 |   0.62 Gpx/s |     107.89ms |     108.16ms |     104.69ms |     111.32ms |       2.02ms |

<details>
    <summary>Full Sweep Benchmark Results</summary>

## NVIDIA B200

| PROFILE                                  | SIZE |    MEDIAN    |     MEAN     |     MIN      |     MAX      |    STDEV     |
|------------------------------------------|------|--------------|--------------|--------------|--------------|--------------|
| newton_lbvh_lbvh                         |   20 |      38.05ms |      38.07ms |      36.58ms |      38.86ms |       0.62ms |
| newton_lbvh_sah                          |   20 |      32.85ms |      32.78ms |      31.42ms |      33.37ms |       0.54ms |
| newton_lbvh_cubql                        |   20 |      30.44ms |      30.34ms |      28.99ms |      30.87ms |       0.53ms |
| newton_sah_lbvh                          |   20 |      35.00ms |      35.17ms |      34.16ms |      36.16ms |       0.63ms |
| newton_sah_sah                           |   20 |      30.17ms |      30.19ms |      29.18ms |      30.94ms |       0.51ms |
| newton_sah_cubql                         |   20 |      27.70ms |      27.79ms |      26.80ms |      28.57ms |       0.51ms |

## NVIDIA GH200

| PROFILE                                  | SIZE |    MEDIAN    |     MEAN     |     MIN      |     MAX      |    STDEV     |
|------------------------------------------|------|--------------|--------------|--------------|--------------|--------------|
| newton_lbvh_lbvh                         |   20 |      74.59ms |      74.52ms |      71.68ms |      76.08ms |       1.21ms |
| newton_lbvh_sah                          |   20 |      65.77ms |      65.61ms |      62.93ms |      66.89ms |       1.06ms |
| newton_lbvh_cubql                        |   20 |      59.00ms |      58.71ms |      55.95ms |      59.82ms |       1.05ms |
| newton_sah_lbvh                          |   20 |      68.24ms |      68.49ms |      66.29ms |      70.41ms |       1.19ms |
| newton_sah_sah                           |   20 |      60.00ms |      60.07ms |      57.87ms |      61.64ms |       1.02ms |
| newton_sah_cubql                         |   20 |      53.25ms |      53.40ms |      51.13ms |      54.79ms |       0.98ms |


## NVIDIA RTX Pro 6000

| PROFILE                                  | SIZE |    MEDIAN    |     MEAN     |     MIN      |     MAX      |    STDEV     |
|------------------------------------------|------|--------------|--------------|--------------|--------------|--------------|
| newton_lbvh_lbvh                         |   20 |      29.99ms |      30.08ms |      29.22ms |      30.77ms |       0.42ms |
| newton_lbvh_sah                          |   20 |      26.51ms |      26.50ms |      25.65ms |      26.99ms |       0.36ms |
| newton_lbvh_cubql                        |   20 |      23.56ms |      23.53ms |      22.61ms |      23.97ms |       0.35ms |
| newton_sah_lbvh                          |   20 |      27.49ms |      27.62ms |      26.98ms |      28.52ms |       0.54ms |
| newton_sah_sah                           |   20 |      24.31ms |      24.33ms |      23.74ms |      24.98ms |       0.43ms |
| newton_sah_cubql                         |   20 |      21.42ms |      21.48ms |      20.78ms |      22.09ms |       0.39ms |

## NVIDIA RTX Pro 6000 1/4 MIG 1g.24gb

| PROFILE                                  | SIZE |    MEDIAN    |     MEAN     |     MIN      |     MAX      |    STDEV     |
|------------------------------------------|------|--------------|--------------|--------------|--------------|--------------|
| newton_lbvh_lbvh                         |   20 |     151.55ms |     152.14ms |     147.77ms |     155.57ms |       2.21ms |
| newton_lbvh_sah                          |   20 |     133.81ms |     133.87ms |     129.58ms |     136.33ms |       1.85ms |
| newton_lbvh_cubql                        |   20 |     118.58ms |     118.53ms |     113.97ms |     120.91ms |       1.84ms |
| newton_sah_lbvh                          |   20 |     139.04ms |     139.65ms |     136.48ms |     144.15ms |       2.71ms |
| newton_sah_sah                           |   20 |     122.68ms |     122.90ms |     120.04ms |     126.18ms |       2.14ms |
| newton_sah_cubql                         |   20 |     107.89ms |     108.16ms |     104.69ms |     111.32ms |       2.02ms |

</details>
# Setup Benchmark

```bash
git clone https://github.com/daniela-hase/IsaacLab.git
git checkout dev/benchmark-franka-cabinet

uv venv
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
python benchmark.py newton_sah_cubql

# or

uv run --no-sync python benchmark.py newton_sah_cubql
```

# Benchmark Results

## B200

| PROFILE                                  | SIZE |    MEDIAN    |     MEAN     |     MIN      |     MAX      |    STDEV     |
|------------------------------------------|------|--------------|--------------|--------------|--------------|--------------|
| newton_lbvh_lbvh                         |   20 |      38.05ms |      38.07ms |      36.58ms |      38.86ms |       0.62ms |
| newton_lbvh_sah                          |   20 |      32.85ms |      32.78ms |      31.42ms |      33.37ms |       0.54ms |
| newton_lbvh_cubql                        |   20 |      30.44ms |      30.34ms |      28.99ms |      30.87ms |       0.53ms |
| newton_sah_lbvh                          |   20 |      35.00ms |      35.17ms |      34.16ms |      36.16ms |       0.63ms |
| newton_sah_sah                           |   20 |      30.17ms |      30.19ms |      29.18ms |      30.94ms |       0.51ms |
| newton_sah_cubql                         |   20 |      27.70ms |      27.79ms |      26.80ms |      28.57ms |       0.51ms |

## RTX Pro 6000

| PROFILE                                  | SIZE |    MEDIAN    |     MEAN     |     MIN      |     MAX      |    STDEV     |
|------------------------------------------|------|--------------|--------------|--------------|--------------|--------------|
| newton_lbvh_lbvh                         |   20 |     151.55ms |     152.14ms |     147.77ms |     155.57ms |       2.21ms |
| newton_lbvh_sah                          |   20 |     133.81ms |     133.87ms |     129.58ms |     136.33ms |       1.85ms |
| newton_lbvh_cubql                        |   20 |     118.58ms |     118.53ms |     113.97ms |     120.91ms |       1.84ms |
| newton_sah_lbvh                          |   20 |     139.04ms |     139.65ms |     136.48ms |     144.15ms |       2.71ms |
| newton_sah_sah                           |   20 |     122.68ms |     122.90ms |     120.04ms |     126.18ms |       2.14ms |
| newton_sah_cubql                         |   20 |     107.89ms |     108.16ms |     104.69ms |     111.32ms |       2.02ms |

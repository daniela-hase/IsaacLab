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
| newton_lbvh_lbvh                         |   20 |      83.66ms |      83.43ms |      79.67ms |      84.80ms |       1.26ms |
| newton_lbvh_sah                          |   20 |      78.37ms |      78.16ms |      74.71ms |      79.39ms |       1.13ms |
| newton_lbvh_cubql                        |   20 |      77.17ms |      77.02ms |      73.58ms |      78.34ms |       1.12ms |
| newton_sah_lbvh                          |   20 |      75.98ms |      76.24ms |      73.78ms |      78.11ms |       1.10ms |
| newton_sah_sah                           |   20 |      70.87ms |      71.16ms |      68.83ms |      72.85ms |       1.01ms |
| newton_sah_cubql                         |   20 |      69.62ms |      69.96ms |      67.62ms |      71.74ms |       1.07ms |

## H200

| PROFILE                                  | SIZE |    MEDIAN    |     MEAN     |     MIN      |     MAX      |    STDEV     |
|------------------------------------------|------|--------------|--------------|--------------|--------------|--------------|
| newton_lbvh_lbvh                         |   20 |     124.64ms |     124.04ms |     118.27ms |     126.26ms |       2.25ms |
| newton_lbvh_sah                          |   20 |     113.43ms |     112.75ms |     107.44ms |     114.52ms |       1.97ms |
| newton_lbvh_cubql                        |   20 |     103.17ms |     102.64ms |      97.78ms |     104.08ms |       1.72ms |
| newton_sah_lbvh                          |   20 |     113.62ms |     114.01ms |     110.54ms |     116.56ms |       1.69ms |
| newton_sah_sah                           |   20 |     103.08ms |     103.26ms |     100.09ms |     105.17ms |       1.38ms |
| newton_sah_cubql                         |   20 |      93.07ms |      93.16ms |      90.06ms |      95.15ms |       1.29ms |

## RTX Pro 6000

| PROFILE                                  | SIZE |    MEDIAN    |     MEAN     |     MIN      |     MAX      |    STDEV     |
|------------------------------------------|------|--------------|--------------|--------------|--------------|--------------|
| newton_lbvh_lbvh                         |   20 |     285.16ms |     284.27ms |     271.59ms |     289.44ms |       4.92ms |
| newton_lbvh_sah                          |   20 |     254.43ms |     252.80ms |     240.66ms |     256.87ms |       4.69ms |
| newton_lbvh_cubql                        |   20 |     236.22ms |     234.70ms |     222.42ms |     238.27ms |       4.44ms |
| newton_sah_lbvh                          |   20 |     261.71ms |     262.52ms |     255.15ms |     268.90ms |       3.98ms |
| newton_sah_sah                           |   20 |     232.60ms |     233.08ms |     225.63ms |     238.16ms |       3.53ms |
| newton_sah_cubql                         |   20 |     214.78ms |     215.15ms |     208.00ms |     220.03ms |       3.35ms |

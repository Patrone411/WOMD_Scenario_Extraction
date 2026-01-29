# waymo_osc_extractor

A small pipeline that downloads a Waymo TFRecord, processes one scene after another, and emits JSON used for later scenario filtering. It uses a slightly modified `TagsGenerator` from `external/waymo_motion_scenario_mining` (existing Master Thesis) to create actor tags, computes road segments using the`waymoScenarioMining` submodule, and then `data_stiching` filters the tagged data of each scene down to the actors present in the road segments. Results are written to S3.

---

## Prerequisites

- **Git** (with access to the private submodules)
- **Python 3.8.20** (tested)
- **Docker** (to build/run the image)
- **AWS credentials** with access to the target S3 bucket
- If required by your network: a trusted **CA bundle** for AWS endpoints

---

## Getting started (local dev)

Clone, pull, and initialize submodules:

```bash
git clone https://gitlab.iavgroup.local/I010444/waymo_osc_extractor.git
cd waymo_osc_extractor
git fetch
git pull
git submodule update --init --recursive
```

Create a virtual environment with Python **3.8.20** and install dependencies:
*Conda*

```bash
conda create -n waymo38 python=3.8.20
conda activate waymo38
pip install -r requirements.txt
```

```bash alternative
# venv (pure Python)
python3.8 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Build the Docker image

```bash
docker build -t waymo-miner:py38 .
```

---

## Usage (Docker)

The image runs the `kube_runner` Python module and expects these environment variables:

| Variable                | Description                                                                                 |
|-------------------------|---------------------------------------------------------------------------------------------|
| `AWS_ACCESS_KEY_ID`     | AWS key (or mount `~/.aws` and use a profile instead)                                      |
| `AWS_SECRET_ACCESS_KEY` | AWS secret                                                                                  |
| `AWS_REGION`            | AWS region (e.g., `eu-west-1`)                                                              |
| `AWS_CA_BUNDLE`         | Path **inside the container** to your CA bundle if your network requires a custom CA        |
| `BUCKET`                | S3 bucket name (e.g., `waymo`)                                                              |
| `INPUT_PREFIX`          | TFRecord key/prefix to process                                                              |
| `RESULT_PREFIX`         | Destination prefix for outputs (JSON)                                                       |
| `SHARD_COUNT`           | Number of shards to split processing into                                                   |
| `MAX_UPLOAD_THREADS`    | Parallel S3 upload threads                                                                  |

### Example

```bash
docker run --rm   -e AWS_CA_BUNDLE=/app/waymo_osc_extractor/certs/IAV-CA-Bundle.pem   -e AWS_ACCESS_KEY_ID="<your key>"   -e AWS_SECRET_ACCESS_KEY="<your key>"   -e AWS_REGION="eu-west-1"   -e BUCKET="waymo"   -e INPUT_PREFIX="tfrecords/training_tfexample.tfrecord-00000-of-01000"   -e RESULT_PREFIX="results/test-docker/"   -e SHARD_COUNT=1   -e MAX_UPLOAD_THREADS=4   waymo-miner:py38
```

## What the pipeline does

1. **Download** a single TFRecord from S3.
2. **Iterate** scene-by-scene.
3. **Tag actors** with the modified `TagsGenerator` from `external/waymo_motion_scenario_mining`.
4. **Compute road segments** using modules in `/waymoScenarioMining`.
5. **Filter** tagged actors with `data_stiching` to those relevant to each scene’s road segments.
6. **Write JSON outputs** (segment data + relevant actor data/tags) to:
   ```
   s3://<BUCKET>/<RESULT_PREFIX>  # e.g., s3://waymo/results/test-docker/
   ```


---

## Other Utils:
from the parent folder containing waymo_osc_extractor run:
python -m waymo_osc_extractor.blockmatcher


blockmatcher.py is a script that combines other functionalities available in this project:

it generates a pytree from an osc file:
config = MiniOSC2ScenarioConfig("/external/scenario_runner/srunner/examples/change_speed.osc")
visitor = ConfigInit(config)
visitor.visit(config.ast_tree)
py_tree = visitor.pytree

it gets map constraints from the scenario config:
map_constraints = {}
map_constraints["min_lanes"] = config.path.min_driving_lanes

it generates actor constraints from the pytree:

actor_constraints = pytree_to_actor_constraints(py_tree)
print("actor constraints: ", actor_constraints)

using the stitched data and the constraints,  run_scenario_matching() returns and saves filtered out scenarios.
due to currently limited functionality of the scenario matching filters, its using a hard coded example of constraints.
per default its using a minimal sample of stitched data for visualization and understanding purposes. tweak arguments of run_scenario_matching to use other stitched data if needed.

the results are plotted using:

plot_matches("blockmatch_results.json")
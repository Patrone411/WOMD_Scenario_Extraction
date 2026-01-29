
## Overview
this is a Python module for processing **Waymo Open Dataset** roadgraph information to find **lane segments matching specific geometric and connectivity criteria**.  
It is primarily designed to locate real-world road sections that match constraints defined in **OpenSCENARIO** descriptions, such as:

- Minimum number of lanes  
- Minimum continuous section length  

This allows you to automatically find driving situations from Waymo data that match OpenSCENARIO requirements, reducing the search space for scenario data mining.

---

## Features
- Extraction of **lane-level topology**:
  - Lane successors & predecessors
  - Left and right lane neighbors
- Search for continuous lane sections with:
  - Minimum section length
  - Minimum number of parallel/neighboring lanes
- Output results as **segment-level neighbor relationships** for targeted scenario identification.

---

## 🛠 Installation
tested using Python 3.8.20
in WSL running Ubuntu 24.04.1 LTS
quit
```bash
git clone https://gitlab.iavgroup.local/I010444/WaymoScenarioMining
cd WaymoScenarioMining
pip install -r requirements.txt
#add your s3 keys to env:
export S3_ACCESS_KEY="your_key"
export S3_SECRET_KEY="your_secret"
```

---

## Getting Started

### Reading scenarios from Waymo TFRecords
You can stream TFRecords from S3 using `tf_scenario_streamer` and construct a `Scenario` object:

```python
from scenario_handling import Scenario, features_description
from s3_handlers import tf_scenario_streamer

for i, example in enumerate(tf_scenario_streamer(features_description)):
    scenario = Scenario(example)
    print(f"Example {i}: scenario ID = {scenario.scenario_id}")
```

The `Scenario` class:
- Extracts and stores **map** and **actor** details.
- Provides plotting utilities for map, actors, and trajectories.
- Instantiates a **LaneGraph** object for lane neighborhood analysis.

---

## Lane Graph Analysis

Given Waymo **Lane Data**, the `LaneGraph` class can find and extract **Lane Segments** made up of Lane Chains and their neighborhood information.

### 1. Root and Branch Lane Chains
```python
sequences = lane_graph.sequences
root_seqs = [s['lane_ids'] for s in sequences if not s["is_branch_root"]]
```
Combines computed lane successor Information into **Lane Chains** which can be divided into Root and Branch Chains.
A successive Chain of Lanes is defined as Root Chain until a single Lane has multiple successor Lanes.
In case of multiple Sucessors, the one with the least angular deviation from the last lane Segment is added to the root chain.
For the other successor, a new Chain is started and marked as Branch Chain.

### 2. Build Neighborhood information
```python
road_segs = lane_graph.build_global_road_segments(all_chains=root_seqs, min_overlap=20)
```
The Chains can then be turned into Road Segments made up of 2 or 3 Lane neighboring lane chains using a geometric
constraint. The output format is close to OpenDRIVE Data in its presentation:

**Output format:**
```python
[
  'seg_idx': {'num_lanes': 2, 'num_segments': 52, 'chains': [{'id': 1, 'lane_ids': [101, 93, 96, 145, 175, 162, 153]}, {'id': 2, 'lane_ids': [117, 109, 95, 99, 146, 176, 168, 158]}]}
  ...
]
```

---
it contains the number of Invloved Lanes, the number of consecutive segments for
which the Neighborhood of the involved lanes persists. The Chain id encodes their relative positional info
just like OpenSCENARIO does. The Lane ids contain the Lanes involved in the respective lane chains.

---

## Example Use Cases
- Identify **multi-lane highway stretches** for overtaking scenarios.
- Extract road sections matching **OpenSCENARIO** constraints for automated testing.

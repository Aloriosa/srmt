from pathlib import Path

import yaml

map_files = [
    'bottlenecks-9-330',
    'test-bottlenecks-9-31000',
            ]

maps = {}
for file_name in map_files:
    with open(Path(__file__).parent / f"{file_name}.yaml", "r") as f:
        maps.update(**yaml.safe_load(f))

MAPS_REGISTRY = maps

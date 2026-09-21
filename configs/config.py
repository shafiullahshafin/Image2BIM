"""
Global configuration defaults for Image2BIM pipeline.
All paths are dynamically customizable via CLI flags in run_pipeline.py.
"""

import os
from pathlib import Path

# Project root: dynamically resolved to Image2BIM
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default directories
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

# Default Model Checkpoints
DEFAULT_HEAT_CHECKPOINT = CHECKPOINTS_DIR / "checkpoint.pth"

# Dataset Root Directory (holds dataset/<dataset_name>/<scene_id>/)
DATASET_DIR = PROJECT_ROOT / "dataset"

# Datasets Config
DATASETS_CONFIG = {
    "structured3d": {
        "raw_dir": str(DATASET_DIR / "structured3d"),
        "norm_dict_path": "d:/Scan2BIM_Experiments/heat/data/s3d_floorplan/exact_normalization_dicts.json",
        "normals_dir": "d:/Scan2BIM_Experiments/heat/data/s3d_floorplan/normals",
    },
    "matterport3d": {
        "raw_dir": str(DATASET_DIR / "matterport3d"),
        "norm_dict_path": None,
        "normals_dir": None,
    }
}

# Reconstruction Defaults
DEFAULT_IMAGE_SIZE = 256
DEFAULT_WALL_HEIGHT = 2.80     # meters
DEFAULT_WALL_THICKNESS = 0.20   # meters
DEFAULT_SLAB_THICKNESS = 0.15   # meters
DEFAULT_DOOR_THICKNESS = 0.05   # meters
DEFAULT_WINDOW_THICKNESS = 0.06 # meters
DEFAULT_CEILING_TRANSPARENCY = 0.80

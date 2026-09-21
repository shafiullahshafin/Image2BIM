"""
STAGE 5: 3D Door & Window Extraction.
Ingests exact doors and windows directly from dataset CAD annotations.
"""

import os
import math
import json
from typing import Dict, Any, List, Optional
import numpy as np

from configs.dataset_adapter import DatasetAdapter


def run_stage5_openings(adapter: DatasetAdapter, scene_id: str, out_dir: str,
                        pg_data: Dict[str, np.ndarray], norm_dict: Dict[str, Any],
                        source: str = "dataset") -> List[Dict[str, Any]]:
    """
    Execute Stage 5: Ingest exact 3D doors and windows from dataset CAD annotations.
    """
    print("\n" + "=" * 80)
    print("STAGE 5: 3D DOOR & WINDOW EXTRACTION (SOURCE: DATASET)")
    print("=" * 80)

    openings = adapter.get_openings(scene_id)
    if openings:
        print(f"  Ingested {len(openings)} openings directly from dataset CAD annotations")
    else:
        print("  No CAD openings found in dataset annotations.")
        openings = []

    out_json = os.path.join(out_dir, f"{scene_id}_openings.json")
    with open(out_json, "w") as f:
        json.dump(openings, f, indent=2)

    n_d = sum(1 for o in openings if o.get("type") == "door")
    n_w = sum(1 for o in openings if o.get("type") == "window")
    print(f"  Doors: {n_d}, Windows: {n_w}")
    print(f"  Saved Openings JSON: {out_json}")
    return openings

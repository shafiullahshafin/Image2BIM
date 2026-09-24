"""
STAGE 7: Benchmark & Evaluation Against CAD Ground Truth.
Compares reconstructed 2D floorplans and 3D BIM elements against CAD annotations.
"""

import os
import json
from typing import Dict, Any, List, Optional
import numpy as np

from configs.dataset_adapter import DatasetAdapter


def run_stage7_benchmark(adapter: DatasetAdapter, clean_id: str,
                         pg_data: Dict[str, np.ndarray], norm_dict: Dict[str, Any],
                         openings: List[Dict[str, Any]], out_dir: str) -> Dict[str, Any]:
    """
    Execute Stage 7: Evaluate reconstructed vector geometry against CAD ground truth.
    """
    print("\n" + "=" * 80)
    print("STAGE 7: BENCHMARKING RECONSTRUCTED BIM GEOMETRY AGAINST CAD GROUND TRUTH")
    print("=" * 80)

    anno = adapter.get_ground_truth(clean_id)
    if not anno or "junctions" not in anno:
        print("  No CAD ground truth annotations found for comparison.")
        return {"has_gt": False}

    junc_raw = [j["coordinate"] for j in anno.get("junctions", [])]
    if len(junc_raw) == 0:
        print("  No junction coordinates in annotations.")
        return {"has_gt": False}

    gt_junctions = np.array(junc_raw, dtype=float) / 1000.0

    min_c = np.array(norm_dict["min_coords"], dtype=float)
    max_c = np.array(norm_dict["max_coords"], dtype=float)
    x_span = max_c[0] - min_c[0]
    y_span = max_c[1] - min_c[1]

    pred_c_px = pg_data["corners"]
    pred_c_m = np.zeros_like(pred_c_px, dtype=float)
    pred_c_m[:, 0] = (pred_c_px[:, 0] / 256.0 * x_span + min_c[0]) / 1000.0
    pred_c_m[:, 1] = (pred_c_px[:, 1] / 256.0 * y_span + min_c[1]) / 1000.0

    metrics: Dict[str, Any] = {"has_gt": True}

    # 1. Corner Vertices Evaluation (< 0.50m tolerance)
    if len(pred_c_m) > 0 and len(gt_junctions) > 0:
        # Distance from each predicted corner to nearest GT junction
        dist_pred_to_gt = np.linalg.norm(pred_c_m[:, None, :2] - gt_junctions[None, :, :2], axis=-1)
        min_dists_pred = np.min(dist_pred_to_gt, axis=-1)
        matched_pred = np.sum(min_dists_pred < 0.50)
        precision = float(matched_pred / len(pred_c_m) * 100.0)

        # Distance from each GT junction to nearest predicted corner
        dist_gt_to_pred = np.linalg.norm(gt_junctions[:, None, :2] - pred_c_m[None, :, :2], axis=-1)
        min_dists_gt = np.min(dist_gt_to_pred, axis=-1)
        matched_gt = np.sum(min_dists_gt < 0.50)
        recall = float(matched_gt / len(gt_junctions) * 100.0)

        mean_dist_cm = float(np.mean(min_dists_pred) * 100.0)
    else:
        matched_pred = 0
        precision = 0.0
        recall = 0.0
        mean_dist_cm = 0.0

    # 2. Footprint Extents
    pred_span = np.max(pred_c_m, axis=0) - np.min(pred_c_m, axis=0) if len(pred_c_m) > 0 else np.array([0.0, 0.0])
    gt_span = np.max(gt_junctions, axis=0) - np.min(gt_junctions, axis=0)
    w_ratio = min(pred_span[0] / gt_span[0], gt_span[0] / pred_span[0]) if pred_span[0] > 0 and gt_span[0] > 0 else 0.0
    l_ratio = min(pred_span[1] / gt_span[1], gt_span[1] / pred_span[1]) if pred_span[1] > 0 and gt_span[1] > 0 else 0.0
    footprint_match = float((w_ratio + l_ratio) / 2.0 * 100.0)

    # 3. Openings
    doors_count = sum(1 for o in openings if o["type"] == "door")
    windows_count = sum(1 for o in openings if o["type"] == "window")

    print(f"  1. CORNER VERTICES ACCURACY:")
    print(f"     - Predicted Corners:        {len(pred_c_m)}")
    print(f"     - Ground Truth CAD Corners: {len(gt_junctions)}")
    print(f"     - Matched (< 0.5m):         {matched_pred} / {len(pred_c_m)} ({precision:.1f}% precision)")
    print(f"     - Recall (< 0.5m):          {matched_gt} / {len(gt_junctions)} ({recall:.1f}% recall)")
    print(f"     - Mean Corner Distance:     {mean_dist_cm:.1f} cm")

    print(f"\n  2. BUILDING FOOTPRINT MATCH:")
    print(f"     - Predicted Footprint Span: {pred_span[0]:.2f}m (W) x {pred_span[1]:.2f}m (L)")
    print(f"     - Ground Truth CAD Span:    {gt_span[0]:.2f}m (W) x {gt_span[1]:.2f}m (L)")
    print(f"     - Footprint Scale Match:    {footprint_match:.1f}%")

    print(f"\n  3. BIM MODEL RECONSTRUCTED ELEMENTS:")
    print(f"     - Reconstructed Walls:      {len(pg_data['edges'])}")
    print(f"     - Doors Extracted:          {doors_count}")
    print(f"     - Windows Extracted:        {windows_count}")
    print("=" * 80 + "\n")

    metrics.update({
        "pred_corners": len(pred_c_m),
        "gt_corners": len(gt_junctions),
        "precision": precision,
        "recall": recall,
        "mean_dist_cm": mean_dist_cm,
        "footprint_match": footprint_match,
        "walls": len(pg_data["edges"]),
        "doors": doors_count,
        "windows": windows_count
    })

    metrics_path = os.path.join(out_dir, f"{clean_id}_metrics.json")
    try:
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Benchmark Metrics JSON:        {metrics_path}")
    except Exception as e:
        print(f"  [WARNING] Could not save metrics.json: {e}")

    return metrics

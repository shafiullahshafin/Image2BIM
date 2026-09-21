"""
STAGE 2: Continuous 3D Point Cloud Synthesis & ASPRS LAS Export.
"""

import os
from typing import Dict, Any, Tuple
import numpy as np
import cv2

from utils.geometry import backproject_panorama_continuous
from utils.export_las import export_las


def run_stage2_pointcloud(sensor_data: Dict[str, Dict[str, Any]], out_dir: str,
                          scene_id: str, subsample: int = 2) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Execute Stage 2: Merge multi-station scans into continuous 3D point cloud and save LAS.
    """
    print("\n" + "=" * 80)
    print("STAGE 2: SYNTHESIZING CONTINUOUS 3D POINT CLOUD (LAS)")
    print("=" * 80)

    all_pts = []
    all_colors = []

    for st, data in sensor_data.items():
        depth_img = data["depth_mm"]
        rgb_bgr = data["rgb_img"]
        rgb_img = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        cam_xyz = data["cam_xyz"]

        pts, cols = backproject_panorama_continuous(depth_img, rgb_img, cam_xyz, subsample=subsample)
        all_pts.append(pts)
        all_colors.append(cols)

    pts = np.vstack(all_pts)
    colors = np.vstack(all_colors)

    min_c = np.min(pts, axis=0)
    max_c = np.max(pts, axis=0)
    span = max_c - min_c

    out_las = os.path.join(out_dir, f"{scene_id}_raw_scan.las")

    scan_rgb_uint16 = (colors.astype(np.uint32) * 257).astype(np.uint16)
    scan_cls = np.full(len(pts), 1, dtype=np.uint8)  # Class 1: Scanned Apartment
    export_las(out_las, pts, scan_rgb_uint16, scan_cls)

    print(f"  Total Raw 3D Points:     {len(pts):,}")
    print(f"  Physical Metric Extents: X=[{min_c[0]:.2f}, {max_c[0]:.2f}] Y=[{min_c[1]:.2f}, {max_c[1]:.2f}] Z=[{min_c[2]:.2f}, {max_c[2]:.2f}]")
    print(f"  Overall Span:            {span[0]:.2f}m (W) x {span[1]:.2f}m (L) x {span[2]:.2f}m (H)")
    print(f"  Saved Raw Scan LAS:      {out_las}")

    return pts, (min_c, max_c), colors

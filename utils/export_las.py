"""
LAS Point Cloud Exporter for CloudCompare and BIM visualization.
Facilitates both:
1. Raw Point Cloud (before processing): continuous unquantized 3D RGB scan.
2. Reconstructed Point Cloud (after processing): dense 3D walls, corner pillars, and classified overlay.
"""

import os
import math
from typing import Optional, Dict, Any, Tuple
import numpy as np
import laspy


def export_las(out_path: str, xyz: np.ndarray, rgb: Optional[np.ndarray] = None,
               classification: Optional[np.ndarray] = None):
    """
    Write 3D points and colors to an industry standard ASPRS LAS file (v1.2, Point Format 3).
    """
    if len(xyz) == 0:
        return

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.min(xyz, axis=0)
    las = laspy.LasData(header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]

    if rgb is not None:
        if rgb.dtype == np.uint8:
            rgb_uint16 = (rgb.astype(np.uint32) * 257).astype(np.uint16)
        else:
            rgb_uint16 = rgb.astype(np.uint16)
        las.red = rgb_uint16[:, 0]
        las.green = rgb_uint16[:, 1]
        las.blue = rgb_uint16[:, 2]

    if classification is not None:
        las.classification = classification

    las.write(out_path)


def sample_wall_points_from_pg(corners_m: np.ndarray, edges: np.ndarray,
                               z_floor: float = 0.0, wall_height: float = 2.80,
                               step: float = 0.03) -> np.ndarray:
    """
    Geometric realization of predicted planar graph edges into a dense 3D wall point cloud.
    """
    pts_list = []
    for edge in edges:
        p1 = corners_m[edge[0]]
        p2 = corners_m[edge[1]]
        length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if length < 0.02:
            continue

        n_s = max(int(length / step), 2)
        n_z = max(int(wall_height / step), 2)

        s_vals = np.linspace(0.0, 1.0, n_s)
        z_vals = np.linspace(z_floor, z_floor + wall_height, n_z)

        S, Z = np.meshgrid(s_vals, z_vals)
        X = p1[0] + S * (p2[0] - p1[0])
        Y = p1[1] + S * (p2[1] - p1[1])

        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
        pts_list.append(pts)

    if not pts_list:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(pts_list)


def sample_corner_pillars_from_pg(corners_m: np.ndarray, z_floor: float = 0.0,
                                  wall_height: float = 2.80, n_samples: int = 50) -> np.ndarray:
    """
    Sample discrete 3D vertical pillar points at each predicted corner junction.
    """
    corner_pts_list = []
    for c in corners_m:
        z_samples = np.linspace(z_floor, z_floor + wall_height, n_samples)
        c_xyz = np.column_stack([np.full_like(z_samples, c[0]), np.full_like(z_samples, c[1]), z_samples])
        corner_pts_list.append(c_xyz)
    if not corner_pts_list:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(corner_pts_list)


def export_reconstructed_pointclouds(pg_data: Dict[str, np.ndarray], norm_dict: Dict[str, Any],
                                     raw_pts: Optional[np.ndarray], raw_colors: Optional[np.ndarray],
                                     out_dir: str, scene_id: str,
                                     wall_height: float = 2.80, wall_step: float = 0.03) -> Dict[str, str]:
    """
    Export after-processing reconstructed point clouds:
    1. <scene_id>_predicted_walls.las (dense 3D walls in Coral Red)
    2. <scene_id>_graph_corners.las (corner junction pillars in Bright Yellow)
    3. <scene_id>_scan_and_walls_overlay.las (combined classified overlay for CloudCompare)
    """
    min_c = np.array(norm_dict["min_coords"], dtype=float)
    max_c = np.array(norm_dict["max_coords"], dtype=float)
    x_span = max_c[0] - min_c[0]
    y_span = max_c[1] - min_c[1]
    # Determine true physical floor level (z_floor):
    # If raw scan points exist, anchor to their 1st percentile elevation (floor surface);
    # otherwise fallback to 0.0m. (min_c[2] from norm_dict is a centered margin, not physical floor).
    if raw_pts is not None and len(raw_pts) > 0:
        z_floor = float(np.percentile(raw_pts[:, 2], 1))
    else:
        z_floor = 0.0

    c_px = pg_data["corners"]
    edges = pg_data["edges"]

    c_m = np.zeros_like(c_px, dtype=float)
    c_m[:, 0] = (c_px[:, 0] / 256.0 * x_span + min_c[0]) / 1000.0
    c_m[:, 1] = (c_px[:, 1] / 256.0 * y_span + min_c[1]) / 1000.0

    # 1. Sample Reconstructed 3D Walls (Coral Red, Class 2)
    wall_xyz = sample_wall_points_from_pg(c_m, edges, z_floor=z_floor, wall_height=wall_height, step=wall_step)
    wall_rgb = np.tile(np.array([65000, 14000, 14000], dtype=np.uint16), (len(wall_xyz), 1))
    wall_cls = np.full(len(wall_xyz), 2, dtype=np.uint8)

    out_walls_las = os.path.join(out_dir, f"{scene_id}_predicted_walls.las")
    export_las(out_walls_las, wall_xyz, wall_rgb, wall_cls)

    # 2. Sample Discrete Corner Pillars (Bright Yellow, Class 3)
    corners_xyz = sample_corner_pillars_from_pg(c_m, z_floor=z_floor, wall_height=wall_height, n_samples=50)
    corners_rgb = np.tile(np.array([65535, 65535, 0], dtype=np.uint16), (len(corners_xyz), 1))
    corners_cls = np.full(len(corners_xyz), 3, dtype=np.uint8)

    out_corners_las = os.path.join(out_dir, f"{scene_id}_graph_corners.las")
    export_las(out_corners_las, corners_xyz, corners_rgb, corners_cls)

    # 3. Merged Overlay LAS (Raw Scan + Reconstructed Walls + Corners)
    out_overlay_las = None
    if raw_pts is not None and len(raw_pts) > 0:
        if raw_colors is not None and raw_colors.dtype == np.uint8:
            scan_rgb = (raw_colors.astype(np.uint32) * 257).astype(np.uint16)
        else:
            scan_rgb = raw_colors.astype(np.uint16) if raw_colors is not None else np.tile(np.array([32000, 32000, 32000], dtype=np.uint16), (len(raw_pts), 1))
        scan_cls = np.full(len(raw_pts), 1, dtype=np.uint8)

        overlay_xyz = np.vstack([raw_pts, wall_xyz, corners_xyz])
        overlay_rgb = np.vstack([scan_rgb, wall_rgb, corners_rgb])
        overlay_cls = np.concatenate([scan_cls, wall_cls, corners_cls])

        out_overlay_las = os.path.join(out_dir, f"{scene_id}_scan_and_walls_overlay.las")
        export_las(out_overlay_las, overlay_xyz, overlay_rgb, overlay_cls)

    print(f"  Saved Reconstructed Walls LAS: {out_walls_las} ({len(wall_xyz):,} pts)")
    print(f"  Saved Discrete Corners LAS:    {out_corners_las} ({len(corners_xyz):,} pts)")
    if out_overlay_las:
        print(f"  Saved Multi-Class Overlay LAS: {out_overlay_las} ({len(overlay_xyz):,} pts)")

    return {
        "walls_las": out_walls_las,
        "corners_las": out_corners_las,
        "overlay_las": out_overlay_las
    }


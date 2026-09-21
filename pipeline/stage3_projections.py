"""
STAGE 3: 2D Top-Down Density & Normal Map Projections.
"""

import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2

from configs.dataset_adapter import DatasetAdapter


def run_stage3_projections(pts: np.ndarray, bounds: Tuple[np.ndarray, np.ndarray],
                           adapter: DatasetAdapter, scene_id: str, out_dir: str,
                           res: int = 256) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Execute Stage 3: Slice point cloud and project onto 256x256 top-down density & normal maps.
    """
    print("\n" + "=" * 80)
    print("STAGE 3: PROJECTING POINT CLOUD TO 256x256 TOP-DOWN DENSITY & NORMAL MAPS")
    print("=" * 80)

    min_c, max_c = bounds
    # Always anchor physical floor and ceiling elevation to point cloud percentiles
    z_floor = float(np.percentile(pts[:, 2], 1))
    z_ceil = float(np.percentile(pts[:, 2], 99))

    # Check adapter for precomputed exact horizontal bounds
    norm_bounds = adapter.get_normalization_bounds(scene_id)
    if norm_bounds is not None:
        min_xy = norm_bounds[0][:2] / 1000.0
        max_xy = norm_bounds[1][:2] / 1000.0
        span_xy = max_xy - min_xy
        print(f"  Ingested exact dataset horizontal extents from normalization dictionary")
    else:
        d_xy = (max_c[:2] - min_c[:2])
        min_xy = min_c[:2] - 0.1 * d_xy
        max_xy = max_c[:2] + 0.1 * d_xy
        span_xy = max_xy - min_xy

    # Filter vertical wall slice (between 0.30m and 2.50m elevation above floor)
    wall_pts = pts[(pts[:, 2] >= z_floor + 0.30) & (pts[:, 2] <= z_floor + 2.50)]
    if len(wall_pts) < 1000:
        wall_pts = pts


    px = np.clip(((wall_pts[:, 0] - min_xy[0]) / span_xy[0] * res).astype(int), 0, res - 1)
    py = np.clip(((wall_pts[:, 1] - min_xy[1]) / span_xy[1] * res).astype(int), 0, res - 1)

    coords_2d = np.stack([px, py], axis=-1)
    unique_coords, counts = np.unique(coords_2d, return_counts=True, axis=0)

    density = np.zeros((res, res), dtype=np.float32)
    density[unique_coords[:, 1], unique_coords[:, 0]] = np.log1p(counts)
    if density.max() > 0:
        density = density / density.max()

    density_uint8 = (density * 255.0).astype(np.uint8)

    # Ingest topdown normal map from adapter if available, otherwise compute Sobel normal map
    normal_uint8 = None
    if hasattr(adapter, "get_topdown_normal"):
        normal_uint8 = adapter.get_topdown_normal(scene_id)

    if normal_uint8 is not None:
        print(f"  Ingested precomputed top-down normal map from dataset")
    else:
        grad_x = cv2.Sobel(density_uint8, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(density_uint8, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(grad_x**2 + grad_y**2) + 1e-6
        nx = np.where(density_uint8 > 10, (grad_x / mag + 1.0) * 0.5 * 255.0, 0.0)
        ny = np.where(density_uint8 > 10, (grad_y / mag + 1.0) * 0.5 * 255.0, 0.0)
        nz = np.zeros_like(nx)
        normal_uint8 = np.clip(np.stack([nx, ny, nz], axis=-1), 0, 255).astype(np.uint8)

    # Composite HEAT Model input
    rgb_heat = np.maximum(cv2.cvtColor(density_uint8, cv2.COLOR_GRAY2BGR), normal_uint8)

    folder_prefix = scene_id if scene_id.startswith("scene_") else f"scene_{scene_id}"
    img_dir = os.path.join(out_dir, f"{folder_prefix}_images")
    os.makedirs(img_dir, exist_ok=True)
    dens_path = os.path.join(img_dir, f"{scene_id}_density.png")
    norm_path = os.path.join(img_dir, f"{scene_id}_normal.png")
    cv2.imwrite(dens_path, density_uint8)
    cv2.imwrite(norm_path, normal_uint8)

    print(f"  Density Map Saved: {dens_path}")
    print(f"  Normal Map Saved:  {norm_path}")

    norm_dict = {
        "min_coords": [float(min_xy[0] * 1000.0), float(min_xy[1] * 1000.0), float(z_floor * 1000.0)],
        "max_coords": [float(max_xy[0] * 1000.0), float(max_xy[1] * 1000.0), float(z_ceil * 1000.0)],
        "image_res": [res, res]
    }
    return rgb_heat, norm_dict

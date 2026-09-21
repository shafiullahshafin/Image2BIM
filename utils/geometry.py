"""
3D Differential Geometry, Backprojection, and Coordinate Transformation Utilities.
"""

import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2


def compute_spherical_normals(pano_depth_m: np.ndarray) -> np.ndarray:
    """
    Compute 360 spherical surface normals in world coordinates (Y=UP).
    Returns uint8 RGB array with range [0, 255].
    """
    H, W = pano_depth_m.shape
    theta_1d = (np.arange(W, dtype=np.float32) / float(W) - 0.5) * 2.0 * math.pi
    phi_1d = (0.5 - np.arange(H, dtype=np.float32) / float(H)) * math.pi
    tt, pp = np.meshgrid(theta_1d, phi_1d)

    X = pano_depth_m * np.cos(pp) * np.sin(tt)
    Y = pano_depth_m * np.sin(pp)
    Z = -pano_depth_m * np.cos(pp) * np.cos(tt)
    P = np.stack([X, Y, Z], axis=-1)

    dP_dy, dP_dx = np.gradient(P, axis=(0, 1))
    normals = np.cross(dP_dx, dP_dy)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-6
    normals = normals / norm

    flip = np.sum(normals * P, axis=-1) > 0
    normals[flip] *= -1.0
    return np.clip((normals + 1.0) * 128.0, 0, 255).astype(np.uint8)



def equi2pers_rgb(equi_img: np.ndarray, fov_deg: float = 90.0, yaw_deg: float = 0.0,
                  pitch_deg: float = 0.0, out_res: int = 512) -> np.ndarray:
    """Extract perspective pinhole view from 360 equirectangular image."""
    H_in, W_in = equi_img.shape[:2]
    f = (out_res / 2.0) / math.tan(math.radians(fov_deg) / 2.0)

    u, v = np.meshgrid(np.arange(out_res), np.arange(out_res))
    x = (u - out_res / 2.0) / f
    y = (v - out_res / 2.0) / f
    z = np.ones_like(x)
    norm = np.sqrt(x**2 + y**2 + z**2)
    xyz = np.stack([x / norm, y / norm, z / norm], axis=-1)

    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    R_pitch = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(pitch), -math.sin(pitch)],
        [0.0, math.sin(pitch), math.cos(pitch)]
    ])
    R_yaw = np.array([
        [math.cos(yaw), 0.0, math.sin(yaw)],
        [0.0, 1.0, 0.0],
        [-math.sin(yaw), 0.0, math.cos(yaw)]
    ])
    R = R_yaw @ R_pitch
    xyz_rot = xyz @ R.T

    theta = np.arctan2(xyz_rot[..., 0], xyz_rot[..., 2])
    phi = np.arcsin(np.clip(-xyz_rot[..., 1], -1.0, 1.0))

    map_x = ((theta / (2.0 * math.pi) + 0.5) * W_in).astype(np.float32) % W_in
    map_y = ((0.5 - phi / math.pi) * H_in).astype(np.float32)
    return cv2.remap(equi_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def backproject_panorama_continuous(depth_img: np.ndarray, rgb_img: np.ndarray,
                                    cam_xyz: np.ndarray, subsample: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Continuous spherical backprojection from equirectangular depth (mm) and RGB.
    Produces physical 3D points in meters and matching uint8 colors.
    """
    H, W = depth_img.shape[:2]
    x_idx = np.arange(0, H, subsample)
    y_idx = np.arange(0, W, subsample)
    xx, yy = np.meshgrid(x_idx, y_idx, indexing="ij")

    d = depth_img[xx, yy].astype(np.float64)
    valid = (d > 250.0) & (d < 15000.0)

    alpha = np.radians(90.0 - (xx[valid] * (180.0 / H)))
    beta = np.radians(yy[valid] * (360.0 / W) - 180.0)
    dv = d[valid]

    z_off = dv * np.sin(alpha)
    xy_off = dv * np.cos(alpha)
    x_off = xy_off * np.sin(beta)
    y_off = xy_off * np.cos(beta)

    pts = (np.stack([x_off, y_off, z_off], axis=-1) + np.array(cam_xyz, dtype=np.float64)) / 1000.0
    cols = rgb_img[xx[valid], yy[valid]]
    return pts, cols


def backproject_box_to_3d(box: List[float], depth_img: np.ndarray, cam_xyz: np.ndarray,
                          yaw_deg: float = 0.0, pitch_deg: float = 0.0,
                          fov_deg: float = 90.0, res: int = 512) -> Optional[np.ndarray]:
    """
    Backproject a 2D bounding box on a perspective image into 3D world coordinates.
    Filters depth penetration into background using median in-plane clustering.
    """
    u_min, v_min, u_max, v_max = box
    if u_max <= u_min or v_max <= v_min:
        return None

    n_samples = 20
    u_grid = np.linspace(u_min, u_max, n_samples)
    v_grid = np.linspace(v_min, v_max, n_samples)
    uu, vv = np.meshgrid(u_grid, v_grid)

    f = (res / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    x = (uu - res / 2.0) / f
    y = (vv - res / 2.0) / f
    z = np.ones_like(x)
    norm = np.sqrt(x**2 + y**2 + z**2)
    xyz = np.stack([x / norm, y / norm, z / norm], axis=-1)

    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    R_pitch = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(pitch), -math.sin(pitch)],
        [0.0, math.sin(pitch), math.cos(pitch)]
    ])
    R_yaw = np.array([
        [math.cos(yaw), 0.0, math.sin(yaw)],
        [0.0, 1.0, 0.0],
        [-math.sin(yaw), 0.0, math.cos(yaw)]
    ])
    R = R_yaw @ R_pitch
    xyz_rot = xyz @ R.T

    theta = np.arctan2(xyz_rot[..., 0], xyz_rot[..., 2])
    phi = np.arcsin(np.clip(-xyz_rot[..., 1], -1.0, 1.0))

    H_in, W_in = depth_img.shape[:2]
    map_x = np.clip(((theta / (2.0 * math.pi) + 0.5) * W_in).astype(int) % W_in, 0, W_in - 1)
    map_y = np.clip(((0.5 - phi / math.pi) * H_in).astype(int), 0, H_in - 1)

    d = depth_img[map_y, map_x].astype(np.float64)
    valid = (d > 250.0) & (d < 12000.0)
    if np.sum(valid) < 15:
        return None

    med_d = np.median(d[valid])
    in_plane = valid & (np.abs(d - med_d) < 350.0)
    if np.sum(in_plane) < 10:
        in_plane = valid

    alpha = np.radians(90.0 - (map_y[in_plane] * (180.0 / H_in)))
    beta = np.radians(map_x[in_plane] * (360.0 / W_in) - 180.0)
    dv = d[in_plane]

    z_off = dv * np.sin(alpha)
    xy_off = dv * np.cos(alpha)
    x_off = xy_off * np.sin(beta)
    y_off = xy_off * np.cos(beta)

    pts = (np.stack([x_off, y_off, z_off], axis=-1) + np.array(cam_xyz, dtype=np.float64)) / 1000.0
    return pts


def snap_to_wall(pts: np.ndarray, wall_segs: List[Dict[str, Any]],
                 default_height: float = 2.80, z_floor: float = 0.0) -> Tuple[Optional[Dict[str, Any]], float, float]:
    """
    Snap 3D point cluster of detected opening to the nearest 2D wall segment.
    """
    min_pt = np.min(pts, axis=0)
    max_pt = np.max(pts, axis=0)
    center = (min_pt + max_pt) / 2.0

    best_wall = None
    min_dist = float("inf")
    best_t = 0.0

    for wall in wall_segs:
        p1 = wall["p1"]
        p2 = wall["p2"]
        v = p2 - p1
        v_len_sq = np.sum(v**2)
        if v_len_sq < 1e-4:
            continue

        t = np.clip(np.dot(center[:2] - p1, v) / v_len_sq, 0.0, 1.0)
        proj = p1 + t * v
        dist = np.linalg.norm(center[:2] - proj)
        if dist < min_dist:
            min_dist = dist
            best_wall = wall
            best_t = t

    return best_wall, min_dist, best_t


def unproject_points_to_meters(points_px: np.ndarray, min_coords: np.ndarray,
                               max_coords: np.ndarray, image_size: int = 256) -> np.ndarray:
    """
    Unproject 2D pixel coordinates back to real-world metric meters using bounding box extents.
    """
    x_min, y_min = min_coords[0], min_coords[1]
    x_span = max_coords[0] - min_coords[0]
    y_span = max_coords[1] - min_coords[1]

    points_m = np.zeros_like(points_px, dtype=np.float64)
    points_m[:, 0] = (points_px[:, 0] / float(image_size) * x_span + x_min) / 1000.0
    points_m[:, 1] = (points_px[:, 1] / float(image_size) * y_span + y_min) / 1000.0
    return points_m

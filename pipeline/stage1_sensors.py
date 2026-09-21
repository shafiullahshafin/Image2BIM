"""
STAGE 1: Sensor Data Ingestion (dataset_sensors).
Ingests metric depth & surface normals directly from the dataset scan stations.
"""

import os
from typing import Dict, Any
import numpy as np
import cv2

from configs.dataset_adapter import DatasetAdapter
from utils.geometry import compute_spherical_normals, equi2pers_rgb


def run_stage1_sensors(adapter: DatasetAdapter, scene_id: str, out_dir: str,
                       mode: str = "dataset_sensors") -> Dict[str, Dict[str, Any]]:
    """
    Execute Stage 1: Load 360 metric depth and surface normal maps from dataset stations.
    """
    print("\n" + "=" * 80)
    print(f"STAGE 1: SENSOR PROCESSING (DATASET_SENSORS) FOR SCENE {scene_id}")
    print("=" * 80)

    stations = adapter.get_scene_stations(scene_id)
    if not stations:
        raise RuntimeError(f"No scan stations found for scene {scene_id} in {adapter.raw_dir}")

    sensor_data = {}

    for st in stations:
        cam_xyz = adapter.get_station_camera(scene_id, st)
        rgb_img = adapter.get_station_rgb(scene_id, st)

        depth_mm = adapter.get_station_depth(scene_id, st)
        if depth_mm is None:
            raise ValueError(f"Station {st} has no given depth in dataset.")

        normals_rgb = adapter.get_station_normal(scene_id, st)
        if normals_rgb is None:
            depth_m = depth_mm.astype(np.float32) / 1000.0
            normals_rgb = compute_spherical_normals(depth_m)
            print(f"  [Station {st}] Ingested given dataset depth.png and computed spherical normals")
        else:
            print(f"  [Station {st}] Ingested given dataset depth.png ({depth_mm.shape}) and normal.png")

        # Save station artifacts in scene-specific images subfolder (e.g. scene_00000_images)
        folder_prefix = scene_id if scene_id.startswith("scene_") else f"scene_{scene_id}"
        img_dir = os.path.join(out_dir, f"{folder_prefix}_images")
        os.makedirs(img_dir, exist_ok=True)

        st_prefix = st if st.startswith("station_") else f"station_{st}"
        depth_path = os.path.join(img_dir, f"{st_prefix}_depth.png")
        normal_path = os.path.join(img_dir, f"{st_prefix}_normal.png")
        cv2.imwrite(depth_path, depth_mm)
        cv2.imwrite(normal_path, cv2.cvtColor(normals_rgb, cv2.COLOR_RGB2BGR))

        # Save standard perspective views for each station
        persp_config = {
            "front": (0.0, 0.0),
            "right": (90.0, 0.0),
            "back": (180.0, 0.0),
            "left": (270.0, 0.0),
            "ceiling": (0.0, 90.0),
            "floor": (0.0, -90.0)
        }
        for view_name, (yaw, pitch) in persp_config.items():
            f_rgb = equi2pers_rgb(rgb_img, fov_deg=90.0, yaw_deg=yaw, pitch_deg=pitch, out_res=512)
            persp_path = os.path.join(img_dir, f"{st_prefix}_persp_{view_name}.png")
            cv2.imwrite(persp_path, f_rgb)

        sensor_data[st] = {
            "cam_xyz": cam_xyz,
            "depth_mm": depth_mm,
            "normals": normals_rgb,
            "rgb_img": rgb_img,
            "depth_path": depth_path,
            "normal_path": normal_path
        }

    return sensor_data

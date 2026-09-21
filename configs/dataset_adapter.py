"""
Image2BIM Standard Dataset Adapter.
A single, unified, dataset-invariant adapter that operates on the standardized folder schema:

<dataset_root>/
└── <scene_id>/                     # e.g., scene_00000
    ├── annotations.json            # [Optional] 3D Doors, Windows & Room CAD
    └── stations/
        ├── <station_id>/
        │   ├── rgb.png             # 360 Panoramic RGB image
        │   ├── depth.png           # [Optional] 16-bit millimeter depth map
        │   ├── normal.png          # [Optional] Surface normal map (RGB-encoded)
        │   └── camera.txt          # [X, Y, Z] camera position in millimeters
        └── ...
"""

import os
import json
import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
from shapely.geometry import Polygon, Point, LineString


class DatasetAdapter:
    """
    Unified, standard indoor scan dataset adapter.
    Reads standardized 360 stations, sensor depth/normals, camera positions,
    and 3D openings.
    """

    def __init__(self, raw_dir: str,
                 norm_dict_path: Optional[str] = None,
                 normals_dir: Optional[str] = None,
                 **kwargs):
        self.raw_dir = raw_dir
        self.norm_dict_path = norm_dict_path
        self.normals_dir = normals_dir
        self.kwargs = kwargs

        self._norm_cache = None
        if norm_dict_path and os.path.exists(norm_dict_path):
            try:
                with open(norm_dict_path, "r") as f:
                    self._norm_cache = json.load(f)
            except Exception as e:
                print(f"[DatasetAdapter] Warning: could not load norm dict from {norm_dict_path}: {e}")

    def _format_scene_name(self, scene_id: str) -> str:
        clean_id = f"{int(scene_id):05d}" if scene_id.isdigit() else scene_id.replace("scene_", "")
        return f"scene_{clean_id}"

    def _format_clean_id(self, scene_id: str) -> str:
        return f"{int(scene_id):05d}" if scene_id.isdigit() else scene_id.replace("scene_", "")

    def get_scene_path(self, scene_id: str) -> str:
        scene_name = self._format_scene_name(scene_id)
        # Direct check
        cand1 = os.path.join(self.raw_dir, scene_name)
        if os.path.exists(cand1):
            return cand1
        cand2 = os.path.join(self.raw_dir, scene_id)
        if os.path.exists(cand2):
            return cand2
        return cand1

    def get_scene_stations(self, scene_id: str) -> List[str]:
        """Return list of station IDs located under stations/ directory."""
        scene_path = self.get_scene_path(scene_id)
        stations_dir = os.path.join(scene_path, "stations")
        if not os.path.exists(stations_dir):
            return []
        return sorted([
            d for d in os.listdir(stations_dir)
            if os.path.isdir(os.path.join(stations_dir, d))
        ])

    def _get_station_dir(self, scene_id: str, station_id: str) -> str:
        return os.path.join(self.get_scene_path(scene_id), "stations", station_id)

    def get_station_paths(self, scene_id: str, station_id: str) -> Dict[str, str]:
        st_dir = self._get_station_dir(scene_id, station_id)
        return {
            "camera": os.path.join(st_dir, "camera.txt"),
            "rgb": os.path.join(st_dir, "rgb.png"),
            "depth": os.path.join(st_dir, "depth.png"),
            "normal": os.path.join(st_dir, "normal.png")
        }

    def get_scene_source_paths(self, scene_id: str) -> Dict[str, Any]:
        scene_path = self.get_scene_path(scene_id)
        return {
            "scene_path": scene_path,
            "annotations": os.path.join(scene_path, "annotations.json"),
            "stations_dir": os.path.join(scene_path, "stations")
        }

    def get_station_camera(self, scene_id: str, station_id: str) -> np.ndarray:
        cam_file = os.path.join(self._get_station_dir(scene_id, station_id), "camera.txt")
        if not os.path.exists(cam_file):
            raise FileNotFoundError(f"Missing camera file: {cam_file}")
        return np.loadtxt(cam_file, dtype=np.float64)

    def get_station_rgb(self, scene_id: str, station_id: str) -> np.ndarray:
        rgb_file = os.path.join(self._get_station_dir(scene_id, station_id), "rgb.png")
        if not os.path.exists(rgb_file):
            raise FileNotFoundError(f"Missing RGB image: {rgb_file}")
        img = cv2.imread(rgb_file)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {rgb_file}")
        return img

    def get_station_depth(self, scene_id: str, station_id: str) -> Optional[np.ndarray]:
        depth_file = os.path.join(self._get_station_dir(scene_id, station_id), "depth.png")
        if not os.path.exists(depth_file):
            return None
        return cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)

    def get_station_normal(self, scene_id: str, station_id: str) -> Optional[np.ndarray]:
        norm_file = os.path.join(self._get_station_dir(scene_id, station_id), "normal.png")
        if not os.path.exists(norm_file):
            return None
        norm_bgr = cv2.imread(norm_file)
        if norm_bgr is None:
            return None
        return cv2.cvtColor(norm_bgr, cv2.COLOR_BGR2RGB)

    def get_ground_truth(self, scene_id: str) -> Optional[Dict[str, Any]]:
        anno_file = os.path.join(self.get_scene_path(scene_id), "annotations.json")
        if not os.path.exists(anno_file):
            return None
        with open(anno_file, "r") as f:
            return json.load(f)

    def get_topdown_normal(self, scene_id: str) -> Optional[np.ndarray]:
        if not self.normals_dir:
            return None
        clean_id = self._format_clean_id(scene_id)
        norm_path = os.path.join(self.normals_dir, f"{clean_id}.png")
        if os.path.exists(norm_path):
            return cv2.imread(norm_path)
        return None

    def get_cad_geometry(self, scene_id: str) -> Dict[str, Any]:
        """
        Extract 100% exact CAD geometry directly from Structured3D annotations.json:
        - Exact room polygon boundaries, semantic names, elevations, and areas
        - Exact outwall building footprint
        - Exact wall segments
        - Exact 3D doors & windows with intelligent room-inward swing orientations
        """
        anno = self.get_ground_truth(scene_id)
        if not anno or "planeLineMatrix" not in anno or "lineJunctionMatrix" not in anno:
            return {"has_cad": False}

        plm = np.array(anno["planeLineMatrix"])
        ljm = np.array(anno["lineJunctionMatrix"])
        planes = anno.get("planes", [])
        junc_raw = [j["coordinate"] for j in anno.get("junctions", [])]
        if not junc_raw:
            return {"has_cad": False}
        juncs = np.array(junc_raw, dtype=float) / 1000.0

        # 1. Exact Room Polygons & Slabs
        rooms = []
        for s in anno.get("semantics", []):
            st = s.get("type")
            if st in [
                "living room", "bedroom", "dining room", "bathroom",
                "kitchen", "corridor", "balcony", "study", "store room", "room", "undefined"
            ]:
                pids = s.get("planeID", [])
                f_pids = [p for p in pids if p < len(planes) and planes[p].get("type") == "floor"]
                c_pids = [p for p in pids if p < len(planes) and planes[p].get("type") == "ceiling"]
                if not f_pids:
                    continue

                fp = f_pids[0]
                lines = np.where(plm[fp] > 0)[0]
                line_juncs = []
                for l in lines:
                    if l < ljm.shape[0]:
                        js = np.where(ljm[l] > 0)[0]
                        if len(js) == 2:
                            line_juncs.append((js[0], js[1]))
                if not line_juncs:
                    continue

                # Order junctions into a continuous perimeter loop
                ordered_js = [line_juncs[0][0], line_juncs[0][1]]
                rem = line_juncs[1:]
                while rem:
                    last_j = ordered_js[-1]
                    found = False
                    for i, (j1, j2) in enumerate(rem):
                        if j1 == last_j:
                            ordered_js.append(j2)
                            rem.pop(i)
                            found = True
                            break
                        elif j2 == last_j:
                            ordered_js.append(j1)
                            rem.pop(i)
                            found = True
                            break
                    if not found:
                        break

                coords = juncs[ordered_js[:-1], :2]
                poly = Polygon(coords)
                if not poly.is_valid or poly.area < 0.05:
                    continue

                z_floor = float(np.mean(juncs[ordered_js[:-1], 2]))
                z_ceil = z_floor + 2.80
                if c_pids:
                    cp = c_pids[0]
                    c_lines = np.where(plm[cp] > 0)[0]
                    c_lines = [l for l in c_lines if l < ljm.shape[0]]
                    if c_lines:
                        c_js = np.where(ljm[c_lines].sum(axis=0) > 0)[0]
                        if len(c_js) > 0:
                            z_ceil = float(np.mean(juncs[c_js, 2]))

                rooms.append({
                    "id": s.get("ID", len(rooms) + 1),
                    "name": st.upper(),
                    "type": st.lower(),
                    "coords": coords,
                    "poly": poly,
                    "area_m2": float(poly.area),
                    "z_floor": z_floor,
                    "z_ceiling": z_ceil,
                    "height": float(z_ceil - z_floor),
                    "centroid": (float(poly.centroid.x), float(poly.centroid.y))
                })

        # 2. Outwall boundary polygon
        ow_lines = []
        for s in anno.get("semantics", []):
            if s.get("type") == "outwall":
                for pid in s.get("planeID", []):
                    if pid < plm.shape[0]:
                        lines = np.where(plm[pid] > 0)[0]
                        for l in lines:
                            if l < ljm.shape[0]:
                                js = np.where(ljm[l] > 0)[0]
                                if len(js) == 2:
                                    if abs(juncs[js[0], 2]) < 0.1 and abs(juncs[js[1], 2]) < 0.1:
                                        ow_lines.append((js[0], js[1]))

        ow_coords = None
        ow_poly = None
        if ow_lines:
            ordered_ow = [ow_lines[0][0], ow_lines[0][1]]
            rem = ow_lines[1:]
            while rem:
                last_j = ordered_ow[-1]
                found = False
                for i, (j1, j2) in enumerate(rem):
                    if j1 == last_j:
                        ordered_ow.append(j2)
                        rem.pop(i)
                        found = True
                        break
                    elif j2 == last_j:
                        ordered_ow.append(j1)
                        rem.pop(i)
                        found = True
                        break
                if not found:
                    break
            ow_c = juncs[ordered_ow[:-1], :2]
            if len(ow_c) >= 3:
                p = Polygon(ow_c)
                if p.is_valid and p.area > 0.5:
                    ow_coords = ow_c
                    ow_poly = p

        # Fallback outwall: buffered union of room polygons if no outwall
        if ow_poly is None and rooms:
            from shapely.ops import unary_union
            u = unary_union([r["poly"] for r in rooms]).buffer(0.24, join_style=2)
            if u.geom_type == "Polygon":
                ow_coords = np.array(u.exterior.coords)[:-1]
                ow_poly = u

        # 3. Openings (Doors & Windows) with Room-Priority Swing Direction
        openings = []
        PRIORITY = {
            "bathroom": 100, "toilet": 100, "restroom": 100, "bedroom": 80,
            "study": 70, "kitchen": 60, "balcony": 50, "store room": 40,
            "dining room": 20, "living room": 10, "corridor": 5
        }

        for s in anno.get("semantics", []):
            stype = s.get("type")
            if stype in ["door", "window"]:
                pids = s.get("planeID", [])
                f_pids = [p for p in pids if p < len(planes) and planes[p].get("type") == "floor"]
                c_pids = [p for p in pids if p < len(planes) and planes[p].get("type") == "ceiling"]
                if not f_pids or not c_pids:
                    continue

                lines_f = np.where(plm[f_pids[0]] > 0)[0]
                lines_f = [l for l in lines_f if l < ljm.shape[0]]
                if not lines_f:
                    continue
                js_f = np.where(ljm[lines_f].sum(axis=0) > 0)[0]
                pts_f = juncs[js_f]

                lines_c = np.where(plm[c_pids[0]] > 0)[0]
                lines_c = [l for l in lines_c if l < ljm.shape[0]]
                if not lines_c:
                    continue
                js_c = np.where(ljm[lines_c].sum(axis=0) > 0)[0]
                pts_c = juncs[js_c]

                z_min = float(np.mean(pts_f[:, 2]))
                z_max = float(np.mean(pts_c[:, 2]))
                height = float(z_max - z_min)

                pts = pts_f[:, :2]
                dists = []
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        dists.append((float(np.linalg.norm(pts[i] - pts[j])), i, j))
                dists.sort(key=lambda x: x[0])
                if len(dists) < 3:
                    continue

                t_edge = dists[0][0]
                w_edge = dists[2][0]
                i1, j1 = dists[2][1], dists[2][2]
                vec = pts[j1] - pts[i1]
                norm_v = np.linalg.norm(vec)
                if norm_v < 1e-6:
                    continue
                u = vec / norm_v
                ang = float(np.arctan2(u[1], u[0]))
                center_2d = np.mean(pts, axis=0)
                center_3d = [float(center_2d[0]), float(center_2d[1]), z_min]

                n = np.array([-u[1], u[0]])

                # Intelligent Door Swing Determination
                probe_dist = 0.35
                p_plus = Point(center_2d + probe_dist * n)
                p_minus = Point(center_2d - probe_dist * n)

                rm_plus = [r for r in rooms if r["poly"].contains(p_plus)]
                rm_minus = [r for r in rooms if r["poly"].contains(p_minus)]

                name_plus = rm_plus[0]["type"] if rm_plus else "exterior"
                name_minus = rm_minus[0]["type"] if rm_minus else "exterior"

                if rm_plus and rm_minus:
                    prio_p = PRIORITY.get(name_plus, 0)
                    prio_m = PRIORITY.get(name_minus, 0)
                    if prio_p != prio_m:
                        swing_sign = 1.0 if prio_p > prio_m else -1.0
                    else:
                        swing_sign = 1.0 if rm_plus[0]["area_m2"] <= rm_minus[0]["area_m2"] else -1.0
                elif rm_plus:
                    swing_sign = 1.0
                elif rm_minus:
                    swing_sign = -1.0
                else:
                    swing_sign = 1.0

                min_c = [float(np.min(pts_f[:, 0])), float(np.min(pts_f[:, 1])), z_min]
                max_c = [float(np.max(pts_c[:, 0])), float(np.max(pts_c[:, 1])), z_max]

                openings.append({
                    "id": s.get("ID", len(openings) + 1),
                    "type": stype,
                    "center": center_3d,
                    "width": float(w_edge),
                    "height": float(height),
                    "thickness": float(t_edge),
                    "angle_rad": ang,
                    "angle_deg": float(math.degrees(ang)),
                    "u": u.tolist(),
                    "n": n.tolist(),
                    "z_min": z_min,
                    "z_max": z_max,
                    "swing_sign": float(swing_sign),
                    "min": min_c,
                    "max": max_c,
                    "size": [max_c[0] - min_c[0], max_c[1] - min_c[1], max_c[2] - min_c[2]]
                })

        # 4. Industry-Standard Canonical Wall Synthesis (Single Partition per Room Boundary)
        raw_walls = []
        for r in rooms:
            coords = r["coords"]
            n_pts = len(coords)
            for i in range(n_pts):
                p1 = coords[i]
                p2 = coords[(i + 1) % n_pts]
                l = float(np.linalg.norm(p2 - p1))
                if l > 0.05:
                    ang = float(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
                    raw_walls.append({
                        "p1": p1.tolist(),
                        "p2": p2.tolist(),
                        "length": l,
                        "angle_rad": ang,
                        "z_floor": r["z_floor"],
                        "height": r["height"],
                        "room": r["name"]
                    })

        # Merge facing shared room edges into a single canonical partition wall on the centerline
        canonical_walls = []
        paired = set()

        for i, w1 in enumerate(raw_walls):
            if i in paired:
                continue
            l1 = LineString([w1["p1"], w1["p2"]])
            p1_1, p2_1 = np.array(w1["p1"]), np.array(w1["p2"])
            v1 = p2_1 - p1_1
            L1 = np.linalg.norm(v1)
            u_dir1 = v1 / L1

            best_j = None
            min_dist = 999.0
            for j, w2 in enumerate(raw_walls):
                if i == j or j in paired:
                    continue
                if w1["room"] == w2["room"]:
                    continue
                l2 = LineString([w2["p1"], w2["p2"]])
                d = l1.distance(l2)
                if d < 0.25:
                    p1_2, p2_2 = np.array(w2["p1"]), np.array(w2["p2"])
                    v2 = p2_2 - p1_2
                    L2 = np.linalg.norm(v2)
                    u_dir2 = v2 / L2
                    dot = np.dot(u_dir1, u_dir2)
                    if abs(dot) > 0.85:
                        proj_1 = np.dot(p1_2 - p1_1, u_dir1)
                        proj_2 = np.dot(p2_2 - p1_1, u_dir1)
                        overlap_min = max(0.0, min(proj_1, proj_2))
                        overlap_max = min(L1, max(proj_1, proj_2))
                        if (overlap_max - overlap_min) > 0.3:
                            if d < min_dist:
                                min_dist = d
                                best_j = j

            if best_j is not None:
                w2 = raw_walls[best_j]
                paired.add(i)
                paired.add(best_j)
                p1_2, p2_2 = np.array(w2["p1"]), np.array(w2["p2"])

                n_dir = np.array([-u_dir1[1], u_dir1[0]])
                mid1 = (p1_1 + p2_1) / 2.0
                mid2 = (p1_2 + p2_2) / 2.0
                gap = float(abs(np.dot(mid2 - mid1, n_dir)))
                sign = float(np.sign(np.dot(mid2 - mid1, n_dir)))
                shift = n_dir * (sign * gap / 2.0)

                all_pts = [p1_1, p2_1, p1_2, p2_2]
                u_coords = [np.dot(p - p1_1, u_dir1) for p in all_pts]
                u_min = min(u_coords)
                u_max = max(u_coords)

                p_start = p1_1 + u_min * u_dir1 + shift
                p_end = p1_1 + u_max * u_dir1 + shift
                length = float(np.linalg.norm(p_end - p_start))
                angle = float(math.atan2(p_end[1] - p_start[1], p_end[0] - p_start[0]))
                t_wall = float(max(gap, 0.12))

                canonical_walls.append({
                    "name": f"Partition_{w1['room'].replace(' ', '_')}_{w2['room'].replace(' ', '_')}",
                    "p1": p_start.tolist(),
                    "p2": p_end.tolist(),
                    "length": length,
                    "angle_rad": angle,
                    "thickness": t_wall,
                    "height": max(w1["height"], w2["height"]),
                    "z_floor": min(w1["z_floor"], w2["z_floor"]),
                    "is_external": False,
                    "room": f"{w1['room']} / {w2['room']}"
                })
            else:
                canonical_walls.append({
                    "name": f"Exterior_{w1['room'].replace(' ', '_')}_{i:02d}",
                    "p1": w1["p1"],
                    "p2": w1["p2"],
                    "length": w1["length"],
                    "angle_rad": w1["angle_rad"],
                    "thickness": 0.20,
                    "height": w1["height"],
                    "z_floor": w1["z_floor"],
                    "is_external": True,
                    "room": w1["room"]
                })

        min_z = min([r["z_floor"] for r in rooms]) if rooms else 0.0
        max_h = max([r["height"] for r in rooms]) if rooms else 2.80

        return {
            "has_cad": True,
            "rooms": rooms,
            "openings": openings,
            "outwall": {"coords": ow_coords, "poly": ow_poly},
            "walls": canonical_walls,
            "z_floor": min_z,
            "wall_height": max_h
        }

    def get_openings(self, scene_id: str) -> List[Dict[str, Any]]:
        """
        Extract 3D door and window coordinates from dataset annotations.
        Prioritizes exact CAD annotations with orientation, swing direction, and millimeter bounds.
        """
        # 1. Try exact Structured3D CAD geometry
        cad_geom = self.get_cad_geometry(scene_id)
        if cad_geom and cad_geom.get("has_cad") and cad_geom.get("openings"):
            return cad_geom["openings"]

        # 2. Fallback for Matterport3D or direct 3D bounding box semantics
        anno = self.get_ground_truth(scene_id)
        if not anno or "semantics" not in anno:
            return []

        direct_openings = [
            s for s in anno["semantics"]
            if s.get("type") in ["door", "window"] and "min" in s and "max" in s
        ]
        if direct_openings:
            openings = []
            for s in direct_openings:
                min_c = np.array(s["min"], dtype=np.float64) / 1000.0
                max_c = np.array(s["max"], dtype=np.float64) / 1000.0
                w = float(max(max_c[0] - min_c[0], max_c[1] - min_c[1]))
                h = float(max_c[2] - min_c[2])
                openings.append({
                    "id": s.get("ID", len(openings) + 1),
                    "type": s.get("type"),
                    "min": min_c.tolist(),
                    "max": max_c.tolist(),
                    "center": ((min_c + max_c) / 2.0).tolist(),
                    "size": (max_c - min_c).tolist(),
                    "width": w,
                    "height": h,
                    "angle_rad": 0.0,
                    "angle_deg": 0.0,
                    "swing_sign": 1.0
                })
            return openings

        return []

    def get_normalization_bounds(self, scene_id: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._norm_cache:
            clean_id = self._format_clean_id(scene_id)
            if clean_id in self._norm_cache:
                min_c = np.array(self._norm_cache[clean_id]["min_coords"], dtype=np.float64)
                max_c = np.array(self._norm_cache[clean_id]["max_coords"], dtype=np.float64)
                return min_c, max_c

        # Fallback to annotations.json bbox if available
        anno = self.get_ground_truth(scene_id)
        if anno and "bbox" in anno:
            min_c = np.array(anno["bbox"]["min"], dtype=np.float64)
            max_c = np.array(anno["bbox"]["max"], dtype=np.float64)
            return min_c, max_c
        return None


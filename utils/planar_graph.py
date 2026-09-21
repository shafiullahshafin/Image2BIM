"""
Planar Graph Room Boundary Loop Extraction and Topology Repair.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
from scipy import ndimage


def _compute_degree(c1: np.ndarray, c2: np.ndarray) -> float:
    vec = (c2[0] - c1[0], -(c2[1] - c1[1]))
    cos = (vec[0] * 1 + vec[1] * 0) / np.sqrt(vec[0] ** 2 + vec[1] ** 2)
    theta = np.arccos(np.clip(cos, -1.0, 1.0))
    if vec[1] < 0:
        theta = np.pi * 2 - theta
    return theta


def _sort_neighbours(adj_mat: np.ndarray, corners: np.ndarray) -> Dict[int, List[int]]:
    nb_orders = dict()
    for idx, c in enumerate(corners):
        nb_ids = np.nonzero(adj_mat[idx])[0]
        nb_degrees = [_compute_degree(c, corners[other_idx]) for other_idx in nb_ids]
        degree_ranks = np.argsort(nb_degrees)
        sort_nb_ids = [nb_ids[i] for i in degree_ranks]
        nb_orders[idx] = sort_nb_ids
    return nb_orders


def _find_wedge_nbs(v_s: int, nb_orders: Dict[int, List[int]], adj_mat: np.ndarray):
    sorted_nbs = nb_orders[v_s]
    start_idx = 0
    while True:
        if start_idx == -len(sorted_nbs):
            return None, None
        v_p, v_q = sorted_nbs[start_idx], sorted_nbs[start_idx - 1]
        if adj_mat[v_p, v_s] == 1 and adj_mat[v_s, v_q] == 1:
            return v_p, v_q
        else:
            start_idx -= 1


def _find_wedge_third_v(v1: int, v2: int, nb_orders: Dict[int, List[int]], adj_mat: np.ndarray, dir: int):
    sorted_nbs = nb_orders[v2]
    v1_idx = sorted_nbs.index(v1)
    if dir == 1:
        v3_idx = v1_idx - 1
        while adj_mat[v2, sorted_nbs[v3_idx]] == 0:
            if sorted_nbs[v3_idx] == v1:
                return None
            v3_idx -= 1
    elif dir == -1:
        v3_idx = v1_idx + 1 if v1_idx <= len(sorted_nbs) - 2 else 0
        while adj_mat[sorted_nbs[v3_idx], v2] == 0:
            if sorted_nbs[v3_idx] == v1:
                return None
            v3_idx = v3_idx + 1 if v3_idx <= len(sorted_nbs) - 2 else 0
    else:
        raise ValueError(f"Unknown direction {dir}")
    return sorted_nbs[v3_idx]


def _get_new_start(adj_mat: np.ndarray, cur_idx: int, corners: np.ndarray):
    for i in range(cur_idx, len(corners)):
        if adj_mat[i].sum() > 0:
            return i
    return None


def _get_regions_for_corner(cur_idx: int, adj_mat: np.ndarray, nb_orders: Dict[int, List[int]]) -> List[List[int]]:
    regions = list()
    if adj_mat[cur_idx].sum() == 0:
        return regions

    v_s = cur_idx
    know_v_q = False
    while v_s is not None:
        if not know_v_q:
            v_p, v_q = _find_wedge_nbs(v_s, nb_orders, adj_mat)
            if v_p is None:
                adj_mat[v_s, :] = 0
                adj_mat[:, v_s] = 0
                break
        else:
            v_p = _find_wedge_third_v(v_q, v_s, nb_orders, adj_mat, dir=-1)
            if v_p is None:
                adj_mat[v_s, :] = 0
                adj_mat[:, v_s] = 0
                break

        cur_region = [v_p, v_s]
        adj_mat[v_p, v_s] = 0
        region_i = 0
        closed_polygon = False

        while v_q is not None:
            cur_region.append(v_q)
            adj_mat[v_s, v_q] = 0
            if v_q == cur_region[0]:
                closed_polygon = True
                break
            else:
                v_p = cur_region[region_i + 1]
                v_s = cur_region[region_i + 2]
                v_q = _find_wedge_third_v(v_p, v_s, nb_orders, adj_mat, dir=1)
                if v_q is None:
                    closed_polygon = False
                    break
                region_i += 1

        if closed_polygon:
            regions.append(cur_region)
            found_next = False
            for temp_i in range(1, len(cur_region)):
                if adj_mat[cur_region[temp_i], cur_region[temp_i - 1]] == 1:
                    found_next = True
                    v_s_idx = temp_i
                    break
            if not found_next:
                v_s = None
            else:
                v_s = cur_region[v_s_idx]
                v_q = cur_region[v_s_idx - 1]
                know_v_q = True
        else:
            break
    return regions


def _compute_region_area(region: np.ndarray) -> int:
    edge_map = np.zeros([256, 256])
    for idx, c in enumerate(region[:-1]):
        cv2.line(edge_map, tuple(c), tuple(region[idx + 1]), 1, 3)
    reverse_edge_map = 1 - edge_map
    label, num_features = ndimage.label(reverse_edge_map)
    if num_features < 2:
        return 0
    bg_label = label[0, 0]
    num_labels = [(label == l).sum() for l in range(1, num_features + 1)]
    num_labels[bg_label - 1] = 0
    room_label = np.argmax(num_labels) + 1
    return int((label == room_label).sum())


def get_outwall(all_regions: List[List[int]], corners: np.ndarray, corner_sorted: bool) -> int:
    if corner_sorted:
        regions_for_top_bot = np.nonzero([(0 in region and len(corners) - 1 in region) for region in all_regions])[0]
        if len(regions_for_top_bot) == 1:
            return regions_for_top_bot[0]
    areas = [_compute_region_area(corners[all_regions[idx]]) for idx in range(len(all_regions))]
    return int(np.argmax(areas)) if areas else 0


def extract_regions(adj_mat: np.ndarray, corners: np.ndarray, corner_sorted: bool = True) -> List[np.ndarray]:
    all_regions = list()
    cur_idx = 0
    corners = corners.astype(int)
    nb_orders = _sort_neighbours(adj_mat, corners)
    while cur_idx is not None:
        regions = _get_regions_for_corner(cur_idx, adj_mat, nb_orders)
        all_regions.extend(regions)
        cur_idx = _get_new_start(adj_mat, cur_idx, corners)

    if not all_regions:
        return []

    outwall_idx = get_outwall(all_regions, corners, corner_sorted)
    if 0 <= outwall_idx < len(all_regions):
        all_regions.pop(outwall_idx)

    all_regions_coords = [corners[regions] for regions in all_regions]
    return all_regions_coords


def _remove_corner(idx: int, adj_list: List[List[int]]):
    if len(adj_list[idx]) == 0:
        return
    nbs = list(adj_list[idx])
    adj_list[idx].pop(0)
    for nb in nbs:
        adj_list[nb].remove(idx)
        if len(adj_list[nb]) < 2:
            _remove_corner(nb, adj_list)


def cleanup_pg(pg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    corners = pg['corners']
    edge_pairs = pg['edges']
    adj_list = [[] for _ in range(len(corners))]

    for edge_pair in edge_pairs:
        adj_list[edge_pair[0]].append(edge_pair[1])
        adj_list[edge_pair[1]].append(edge_pair[0])

    for idx in range(len(corners)):
        if len(adj_list[idx]) < 2:
            _remove_corner(idx, adj_list)

    new_corners = list()
    old_to_new = dict()
    counter = 0
    for c_i in range(len(adj_list)):
        if len(adj_list[c_i]) > 0:
            new_corners.append(corners[c_i])
            old_to_new[c_i] = counter
            counter += 1

    new_edges = list()
    for c_i_1 in range(len(adj_list)):
        for c_i_2 in adj_list[c_i_1]:
            if c_i_1 < c_i_2:
                new_edges.append((old_to_new[c_i_1], old_to_new[c_i_2]))

    return {
        'corners': np.array(new_corners),
        'edges': np.array(new_edges)
    }


def preprocess_pg(pg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    corners = pg['corners']
    edge_pairs = pg['edges']
    adj_mat = np.zeros([len(corners), len(corners)])
    for edge_pair in edge_pairs:
        c1, c2 = edge_pair
        adj_mat[c1][c2] = 1
        adj_mat[c2][c1] = 1
    return corners, adj_mat


def get_regions_from_pg(pg: Dict[str, Any], corner_sorted: bool = True) -> List[np.ndarray]:
    cleaned = cleanup_pg(pg)
    corners, adj_mat = preprocess_pg(cleaned)
    if len(corners) == 0:
        return []
    return extract_regions(adj_mat, corners, corner_sorted)

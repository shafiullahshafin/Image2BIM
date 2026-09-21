"""
STAGE 4: HEAT Neural Transformer 2D Floorplan Reconstruction.
"""

import os
from typing import Dict, Any, Tuple
import numpy as np
import cv2
import torch
import torch.nn as nn
import scipy.ndimage.filters as filters

from models.resnet import ResNetBackbone
from models.corner_models import HeatCorner
from models.edge_models import HeatEdge
from models.corner_to_edge import get_infer_edge_pairs
from configs.data_utils import get_pixel_features


def corner_nms(preds: np.ndarray, confs: np.ndarray, image_size: int = 256):
    data = np.zeros([image_size, image_size])
    neighborhood_size = 5
    threshold = 0

    for i in range(len(preds)):
        data[preds[i, 1], preds[i, 0]] = confs[i]

    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    results = np.where(maxima > 0)
    filtered_preds = np.stack([results[1], results[0]], axis=-1)

    new_confs = list()
    for i, pred in enumerate(filtered_preds):
        new_confs.append(data[pred[1], pred[0]])
    new_confs = np.array(new_confs)
    return filtered_preds, new_confs


def postprocess_preds(corners: np.ndarray, confs: np.ndarray, edges: np.ndarray):
    corner_degrees = dict()
    for edge_i, edge_pair in enumerate(edges):
        corner_degrees[edge_pair[0]] = corner_degrees.setdefault(edge_pair[0], 0) + 1
        corner_degrees[edge_pair[1]] = corner_degrees.setdefault(edge_pair[1], 0) + 1
    good_ids = [i for i in range(len(corners)) if i in corner_degrees]
    if len(good_ids) == len(corners):
        return corners, confs, edges
    else:
        good_corners = corners[good_ids]
        good_confs = confs[good_ids]
        id_mapping = {value: idx for idx, value in enumerate(good_ids)}
        new_edges = list()
        for edge_pair in edges:
            new_pair = (id_mapping[edge_pair[0]], id_mapping[edge_pair[1]])
            new_edges.append(new_pair)
        new_edges = np.array(new_edges)
        return good_corners, good_confs, new_edges


def run_stage4_heat_floorplan(rgb_heat: np.ndarray, ckpt_path: str, out_dir: str,
                              scene_id: str, image_size: int = 256,
                              infer_times: int = 3, corner_thresh: float = 0.01) -> Dict[str, np.ndarray]:
    """
    Execute Stage 4: Run HEAT neural transformer model to predict corners and wall edges.
    """
    print("\n" + "=" * 80)
    print("STAGE 4: EXECUTING HEAT NEURAL TRANSFORMER (2D FLOORPLAN RECONSTRUCTION)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Loading HEAT Checkpoint: {ckpt_path} on {device}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    backbone = ResNetBackbone()
    corner_model = HeatCorner(input_dim=128, hidden_dim=256, num_feature_levels=4,
                              backbone_strides=backbone.strides, backbone_num_channels=backbone.num_channels)
    edge_model = HeatEdge(input_dim=128, hidden_dim=256, num_feature_levels=4,
                          backbone_strides=backbone.strides, backbone_num_channels=backbone.num_channels)

    if torch.cuda.is_available():
        backbone = nn.DataParallel(backbone).cuda()
        corner_model = nn.DataParallel(corner_model).cuda()
        edge_model = nn.DataParallel(edge_model).cuda()
        backbone.load_state_dict(ckpt["backbone"])
        corner_model.load_state_dict(ckpt["corner_model"])
        edge_model.load_state_dict(ckpt["edge_model"])
    else:
        def clean_sd(sd):
            return {k.replace("module.", ""): v for k, v in sd.items()}
        backbone.load_state_dict(clean_sd(ckpt["backbone"]))
        corner_model.load_state_dict(clean_sd(ckpt["corner_model"]))
        edge_model.load_state_dict(clean_sd(ckpt["edge_model"]))

    backbone.eval()
    corner_model.eval()
    edge_model.eval()

    # Preprocess image
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = rgb_heat.astype(np.float32) / 255.0
    img = (img - np.array(mean)) / np.array(std)
    img_tensor = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)

    pixels, pixel_features = get_pixel_features(image_size=image_size)

    with torch.no_grad():
        image_feats, feat_mask, all_image_feats = backbone(img_tensor)
        p_feat = pixel_features.unsqueeze(0).repeat(img_tensor.shape[0], 1, 1, 1).to(device)
        preds_s1 = corner_model(image_feats, feat_mask, p_feat, pixels, all_image_feats)

        c_outputs_np = preds_s1[0].detach().cpu().numpy()
        pos_indices = np.where(c_outputs_np >= corner_thresh)
        pred_corners = pixels[pos_indices]
        pred_confs = c_outputs_np[pos_indices]
        pred_corners, pred_confs = corner_nms(pred_corners, pred_confs, image_size=preds_s1.shape[1])

        pred_corners, pred_confs, edge_coords, edge_mask, edge_ids = get_infer_edge_pairs(pred_corners, pred_confs)

        if len(pred_corners) < 2 or edge_coords.shape[1] == 0:
            pos_edges = np.empty((0, 2), dtype=int)
            edge_confs = np.empty((0,), dtype=float)
        else:
            edge_coords = edge_coords.to(device)
            edge_mask = edge_mask.to(device)
            edge_ids = edge_ids.to(device)

            corner_nums = torch.tensor([len(pred_corners)]).to(device)
            corner_multiplier = getattr(ckpt["args"], "corner_to_edge_multiplier", 3)
            max_candidates = torch.stack([corner_nums.max() * corner_multiplier] * len(corner_nums), dim=0)

            all_pos_ids = set()
            all_edge_confs = dict()

            for tt in range(infer_times):
                if tt == 0:
                    gt_values = torch.zeros_like(edge_mask).long()
                    gt_values[:, :] = 2

                s1_logits, s2_logits_hb, s2_logits_rel, selected_ids, s2_mask, s2_gt_values = edge_model(
                    image_feats, feat_mask, p_feat, edge_coords, edge_mask,
                    gt_values, corner_nums, max_candidates, True
                )

                num_total = s1_logits.shape[2]
                num_selected = selected_ids.shape[1]
                num_filtered = num_total - num_selected

                s2_preds_hb = s2_logits_hb.squeeze().softmax(0)
                s2_preds_hb_np = s2_preds_hb[1, :].detach().cpu().numpy()
                selected_ids_np = selected_ids.squeeze().detach().cpu().numpy()

                if tt != infer_times - 1:
                    s2_preds_np = s2_preds_hb_np
                    pos_edge_ids = np.where(s2_preds_np >= 0.9)
                    neg_edge_ids = np.where(s2_preds_np <= 0.01)
                    for pos_id in pos_edge_ids[0]:
                        actual_id = selected_ids_np[pos_id]
                        if gt_values[0, actual_id] != 2:
                            continue
                        all_pos_ids.add(actual_id)
                        all_edge_confs[actual_id] = s2_preds_np[pos_id]
                        gt_values[0, actual_id] = 1
                    for neg_id in neg_edge_ids[0]:
                        actual_id = selected_ids_np[neg_id]
                        if gt_values[0, actual_id] != 2:
                            continue
                        gt_values[0, actual_id] = 0
                    num_to_pred = (gt_values == 2).sum()
                    if num_to_pred <= num_filtered:
                        break
                else:
                    s2_preds_np = s2_preds_hb_np
                    pos_edge_ids = np.where(s2_preds_np >= 0.5)
                    for pos_id in pos_edge_ids[0]:
                        actual_id = selected_ids_np[pos_id]
                        if s2_mask[0][pos_id] is True or gt_values[0, actual_id] != 2:
                            continue
                        all_pos_ids.add(actual_id)
                        all_edge_confs[actual_id] = s2_preds_np[pos_id]

            pos_edge_ids = list(all_pos_ids)
            edge_confs = [all_edge_confs[idx] for idx in pos_edge_ids]
            pos_edges = edge_ids[pos_edge_ids].cpu().numpy()
            edge_confs = np.array(edge_confs)

    if len(pred_corners) > 0 and len(pos_edges) > 0:
        pred_corners, pred_confs, pos_edges = postprocess_preds(pred_corners, pred_confs, pos_edges)
    else:
        pos_edges = np.empty((0, 2), dtype=int)

    pg_data = {
        "corners": pred_corners,
        "edges": pos_edges
    }

    out_npy = os.path.join(out_dir, f"{scene_id}_floorplan.npy")
    np.save(out_npy, pg_data)

    print(f"  HEAT Floorplan Reconstructed: {len(pred_corners)} corners, {len(pos_edges)} wall edges")
    print(f"  Saved Vector CAD Floorplan:   {out_npy}")

    # Save visualizations into scene-specific images subfolder (e.g. scene_00000_images)
    folder_prefix = scene_id if scene_id.startswith("scene_") else f"scene_{scene_id}"
    img_dir = os.path.join(out_dir, f"{folder_prefix}_images")
    os.makedirs(img_dir, exist_ok=True)

    # 1. Prediction Corners Visualization (red dots)
    viz_corner = rgb_heat.copy()
    for c in pred_corners:
        cv2.circle(viz_corner, tuple(c.astype(int)), 4, (0, 0, 255), -1)
    out_corner_viz = os.path.join(img_dir, f"{scene_id}_pred_corner.png")
    cv2.imwrite(out_corner_viz, viz_corner)

    # 2. Prediction Edges Visualization (yellow wall lines)
    viz_edge = rgb_heat.copy()
    for e in pos_edges:
        pt1 = tuple(pred_corners[e[0]].astype(int))
        pt2 = tuple(pred_corners[e[1]].astype(int))
        cv2.line(viz_edge, pt1, pt2, (0, 255, 255), 2)
    out_edge_viz = os.path.join(img_dir, f"{scene_id}_pred_edge.png")
    cv2.imwrite(out_edge_viz, viz_edge)

    # 3. Combined Floorplan Overlay (lines + corners)
    viz_img = viz_edge.copy()
    for c in pred_corners:
        cv2.circle(viz_img, tuple(c.astype(int)), 4, (0, 0, 255), -1)
    out_viz = os.path.join(img_dir, f"{scene_id}_predicted_floorplan.png")
    cv2.imwrite(out_viz, viz_img)

    print(f"  Saved Visual Overlays:        {out_corner_viz}, {out_edge_viz}, {out_viz}")

    return pg_data

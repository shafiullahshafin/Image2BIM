import itertools
import numpy as np
import torch

# pre-compute all combinations to generate edge candidates faster
all_combibations = dict()
for length in range(2, 351):
    ids = np.arange(length)
    combs = np.array(list(itertools.combinations(ids, 2)))
    all_combibations[length] = combs


def get_infer_edge_pairs(corners: np.ndarray, confs: np.ndarray):
    """
    Generate all pairwise combinations of predicted corners as candidate edges.
    """
    if len(corners) < 2:
        return corners, confs, torch.empty((1, 0, 2, 2)).long(), torch.empty((1, 0)).bool(), torch.empty((0, 2)).long()

    ind = np.lexsort(corners.T)
    corners = corners[ind]
    confs = confs[ind]

    N = len(corners)
    if N not in all_combibations:
        ids = np.arange(N)
        edge_ids = np.array(list(itertools.combinations(ids, 2)))
    else:
        edge_ids = all_combibations[N]

    edge_coords = corners[edge_ids]
    edge_coords = torch.tensor(np.array(edge_coords)).unsqueeze(0).long()
    mask = torch.zeros([edge_coords.shape[0], edge_coords.shape[1]]).bool()
    edge_ids_tensor = torch.tensor(np.array(edge_ids))
    return corners, confs, edge_coords, mask, edge_ids_tensor

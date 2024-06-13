import kornia
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def hinge_embedding_loss(embedding_chw, plane_ids_1hw, t_pull=0.5, t_push=1.5):
    unique_plane_ids = torch.unique(plane_ids_1hw)
    num_planes = (unique_plane_ids > 0).sum()
    if num_planes == 0:
        return 0.0

    plane_embeddings_kcn = []
    # select embedding with segmentation
    for plane_id in unique_plane_ids:
        if plane_id != 0:
            plane_embedding = embedding_chw[:, plane_ids_1hw[0] == plane_id]
            plane_embeddings_kcn.append(plane_embedding)

    centers_kc1 = []
    for feature in plane_embeddings_kcn:
        center = torch.mean(feature, dim=-1, keepdim=True)
        centers_kc1.append(center)

    # intra-embedding loss within a plane
    pull_loss = 0.0
    for feature, center in zip(plane_embeddings_kcn, centers_kc1):
        dis = torch.norm(feature - center, 2, dim=0) - t_pull
        dis = F.relu(dis)
        pull_loss += torch.nanmean(dis)
    pull_loss = pull_loss / num_planes

    if num_planes <= 1 or len(centers_kc1) == 0:
        return pull_loss

    # inter-plane loss
    centers_kc = torch.stack(centers_kc1).squeeze(-1)
    A_1kc = centers_kc.unsqueeze(0)
    B_k1c = centers_kc.unsqueeze(1)
    distance_kk = torch.norm(A_1kc - B_k1c, 2, dim=2)

    # select pair wise distance from distance matrix
    eye = torch.eye(int(num_planes)).to(distance_kk.device)
    pair_distance = torch.masked_select(distance_kk, eye == 0)

    pair_distance = t_push - pair_distance
    pair_distance = F.relu(pair_distance)
    push_loss = torch.nanmean(pair_distance)

    loss = pull_loss + push_loss
    return loss

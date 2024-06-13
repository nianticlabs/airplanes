"""
This function has been adapted from PlanarRecon

From PlanarRecon:
https://github.com/neu-vi/PlanarRecon/blob/c4c9b891ebed3789db37c249b11c5d559a0a6dd3/tools/evaluate_utils.py#L30

Released under Apache-2.0 License
Full license available at https://github.com/neu-vi/PlanarRecon/blob/main/LICENSE
"""

from typing import Optional

import numpy as np
import open3d as o3d
from trimesh import Trimesh


def project_to_mesh(
    from_mesh: Trimesh,
    to_mesh: Trimesh,
    attribute: np.ndarray,
    attr_name: str,
    color_mesh: Optional[Trimesh] = None,
    dist_thresh: Optional[float] = None,
) -> Trimesh:
    """Transfers attributs from from_mesh to to_mesh using nearest neighbors

    Each vertex in to_mesh gets assigned the attribute of the nearest
    vertex in from mesh. Used for semantic evaluation.

    Params:
        from_mesh: Trimesh with known attributes
        to_mesh: Trimesh to be labeled
        attribute: Which attribute to transfer
        dist_thresh: Do not transfer attributes beyond this distance
            (None transfers regardless of distance between from and to vertices)

    Returns:
        Trimesh containing transfered attribute
    """

    if len(from_mesh.vertices) == 0:
        to_mesh.vertex_attributes[attr_name] = np.zeros((0), dtype="uint32")
        to_mesh.visual.vertex_colors = np.zeros((0), dtype="uint32")
        return to_mesh

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(from_mesh.vertices)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    pred_ids = attribute.copy()
    pred_colors = (
        from_mesh.visual.vertex_colors if color_mesh is None else color_mesh.visual.vertex_colors
    )

    matched_ids = np.zeros((to_mesh.vertices.shape[0]), dtype="uint32")
    matched_colors = np.zeros((to_mesh.vertices.shape[0], 4), dtype="uint32")

    for i, vert in enumerate(to_mesh.vertices):
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        if dist_thresh is None or dist[0] < dist_thresh:
            matched_ids[i] = pred_ids[inds[0]]
            matched_colors[i] = pred_colors[inds[0]]

    mesh = to_mesh.copy()
    mesh.vertex_attributes[attr_name] = matched_ids
    mesh.visual.vertex_colors = matched_colors
    return mesh, matched_ids

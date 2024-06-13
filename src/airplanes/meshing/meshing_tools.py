import logging
import math
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import open3d as o3d
import trimesh
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.ERROR)


@cuda.jit
def find_nearest_neighbors_gpu(coords_A, coords_B, indices, distances, max_dist):
    tid = cuda.grid(1)

    if tid < coords_A.shape[0]:
        min_dist = max_dist
        min_idx = -1

        for i in range(coords_B.shape[0]):
            dist = 0.0
            for j in range(3):
                dist += (coords_A[tid, j] - coords_B[i, j]) ** 2
            dist = math.sqrt(dist)

            if dist < min_dist:
                min_dist = dist
                min_idx = i

        indices[tid] = min_idx
        distances[tid] = min_dist


def find_nearest_neighbors(points_from: np.ndarray, points_to: np.ndarray, max_dist: float = 1.0):
    """Brute force nearest neighbour search between two point clouds on the GPU."""
    # allocate memory on the device
    indices = cuda.device_array((points_from.shape[0],), dtype="int32")
    distances = cuda.device_array((points_from.shape[0],), dtype="float32")

    # move data to the device
    points_from = cuda.to_device(points_from)
    points_to = cuda.to_device(points_to)

    # compute nearest neighbors
    block_size = 1024
    grid_size = (points_from.shape[0] + block_size - 1) // block_size
    find_nearest_neighbors_gpu[grid_size, block_size](points_from, points_to, indices, distances, max_dist)  # type: ignore

    # move back to host and return distances and indices
    return distances.copy_to_host(), indices.copy_to_host()


def compute_point_cloud_metrics(
    gt_pcd: o3d.geometry.PointCloud,
    pred_pcd: o3d.geometry.PointCloud,
    max_dist: float = 1.0,
    dist_threshold: float = 0.05,
    visible_pred_indices: Optional[list[int]] = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Compute metrics for a predicted and gt point cloud.

    If the predicted point cloud is empty, all the lower-is-better metrics will be set to max_dist
    and all the higher-is-better metrics to 0.

    Args:
        gt_pcd (o3d.geometry.PointCloud): gt point cloud.
        pred_pcd (o3d.geometry.PointCloud): predicted point cloud, will be compared to gt_pcd.
        max_dist (float, optional): Maximum distance to clip distances to in meters.
            Defaults to 1.0.
        dist_threshold (float, optional): Distance threshold to use for precision
            and recall in meters. Defaults to 0.05.
        visible_pred_indices (list[int], optional): Indices of the predicted points that are
            visible in the scene. Defaults to None. When not None will be used to filter out
            predicted points when computing pred to gt.

    Returns:
        dict[str, float]: Metrics for this point cloud comparison.
    """
    metrics: Dict[str, float] = {}

    if len(pred_pcd.points) == 0:
        metrics["acc↓"] = max_dist
        metrics["compl↓"] = max_dist
        metrics["chamfer↓"] = max_dist
        metrics["precision↑"] = 0.0
        metrics["recall↑"] = 0.0
        metrics["f1_score↑"] = 0.0
        distances_pred2gt = np.zeros([])
        distances_gt2pred = np.zeros(len(gt_pcd.points))
        return metrics, distances_pred2gt, distances_gt2pred

    # find nearest neighbors
    distances_gt2pred, indices_gt2pred = find_nearest_neighbors(
        np.array(gt_pcd.points), np.array(pred_pcd.points), max_dist
    )

    # only use the visibility masks when computing pred to gt distances
    if visible_pred_indices is not None:
        pred_points = np.array(pred_pcd.points)[visible_pred_indices]
    else:
        pred_points = np.array(pred_pcd.points)
    distances_pred2gt, indices_pred2gt = find_nearest_neighbors(
        pred_points, np.array(gt_pcd.points), max_dist
    )

    # find points sampled from faces which span 2 or more planes.
    # we store this in the red channel and remove from
    # the gt_pcd, and remove points from pred if their nearest gt neighbour is invalid
    valid_gt = np.array(gt_pcd.colors)[:, 0] > 0.01
    distances_gt2pred = distances_gt2pred[valid_gt]
    valid_pred = np.array([valid_gt[i] for i in indices_pred2gt])
    distances_pred2gt = distances_pred2gt[valid_pred]

    # accuracy
    metrics["acc↓"] = float(np.mean(distances_pred2gt))

    # completion
    metrics["compl↓"] = float(np.mean(distances_gt2pred))

    # chamfer distance
    metrics["chamfer↓"] = 0.5 * (metrics["acc↓"] + metrics["compl↓"])

    # precision
    metrics["precision↑"] = (distances_pred2gt <= dist_threshold).astype("float32").mean()

    # recall
    metrics["recall↑"] = (distances_gt2pred <= dist_threshold).astype("float32").mean()

    # F1 score
    # catch the edge case where both precision and recall are 0
    if metrics["precision↑"] + metrics["recall↑"] > 0.0:
        metrics["f1_score↑"] = (2 * metrics["precision↑"] * metrics["recall↑"]) / (
            metrics["precision↑"] + metrics["recall↑"]
        )
    else:
        metrics["f1_score↑"] = 0.0

    return metrics, distances_pred2gt, distances_gt2pred


def subsample_mesh(mesh: o3d.geometry.TriangleMesh, num_samples: int) -> o3d.geometry.PointCloud:
    """Subsample a mesh to get a point cloud. If the mesh is empty, return an empty point cloud.

    Params:
        mesh: open3d triangle mesh
        num_samples: number of samples to extract from the mesh
    Returns:
        an open3d point cloud object
    """
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return o3d.geometry.PointCloud()
    return mesh.sample_points_uniformly(number_of_points=num_samples)


def squash_vertices_onto_planes(
    mesh: trimesh.Trimesh, core_mask: Optional[np.ndarray] = None
) -> trimesh.Trimesh:
    """Given a mesh with plane IDs encoded in RGB channel, squash each vertex
    into its corresponding planar surface.

    Params:
        mesh: trimesh Mesh
        core_mask: Optional[np.ndarray]: An optional array indicating which points we should
            use when computing the plane parameters. This allows us to exclude 'bad' points from
            the plane fitting in this step. If None, all points will be used.

    Returns:
        trimesh mesh, with a planarised scene.
    """
    cols = np.unique(mesh.visual.vertex_colors, axis=0)
    vert_norms = mesh.vertex_normals

    verts = mesh.vertices.copy()
    core_mask = core_mask if core_mask is not None else np.ones((verts.shape[0])).astype(bool)
    unlabelled_mask = np.zeros((verts.shape[0])).astype(bool)
    for col in cols:
        mask = (mesh.visual.vertex_colors == col).all(axis=1)
        points = mesh.vertices[mask]
        if col[:3].max() > 0:  # and mask.sum() > 100:
            # compute plane parameters and move all points onto the plane
            normal = np.median(vert_norms[mask & core_mask], 0)
            normal = normal / np.linalg.norm(normal)
            offset = -np.median((points * normal).sum(-1))

            dist = (points * normal).sum(-1) + offset
            points = points - normal[None, :] * dist[:, None]

            verts[mask] = points
        else:
            unlabelled_mask[mask] = True

    plane_mesh = trimesh.Trimesh(
        vertices=verts, faces=mesh.faces, vertex_colors=mesh.visual.vertex_colors, process=False
    )

    # remove faces and vertices with non-planar vertices or spanning multiple planes
    v0, v1, v2 = plane_mesh.faces[:, 0], plane_mesh.faces[:, 1], plane_mesh.faces[:, 2]
    c0, c1, c2 = (
        plane_mesh.visual.vertex_colors[v0],
        plane_mesh.visual.vertex_colors[v1],
        plane_mesh.visual.vertex_colors[v2],
    )
    unlabelled_faces = (
        unlabelled_mask[v0]
        | unlabelled_mask[v1]
        | unlabelled_mask[v2]
        | (c0 != c1).any(axis=1)
        | (c0 != c2).any(axis=1)
        | (c1 != c2).any(axis=1)
    )

    plane_mesh.update_faces(~unlabelled_faces)
    plane_mesh.update_vertices(~unlabelled_mask)
    return plane_mesh


def squash_gt_vertices_onto_planes(mesh: trimesh.Trimesh, planes: np.ndarray) -> trimesh.Trimesh:
    """Given a mesh, squash each vertex to its corresponding plane to obtain a planar representation
    of the scene.

    Params:
        mesh: Trimesh object. Each color in the mesh encodes the plane ID
        planes: array that contains the mapping (plane IDs, plane parameters)
    Returns:
        a mesh with a planar representation of the scene
    """
    plane_ids = mesh.visual.vertex_colors.copy().astype("int32")
    plane_ids = (plane_ids[:, 0] * 256 * 256 + plane_ids[:, 1] * 256 + plane_ids[:, 2]) // 100 - 1

    vertices = mesh.vertices.copy()
    non_planar_vertices = np.zeros((vertices.shape[0])).astype(bool)

    # project vertices onto planes
    for plane_id in np.unique(plane_ids):
        instance_mask = plane_ids == plane_id
        instance_vertices = vertices[instance_mask]

        if plane_id >= 0:
            params = planes[plane_id]
            offset = np.linalg.norm(params)
            if offset > 0:
                normal = params / offset
                dist = (instance_vertices * normal).sum(-1) - 1 / offset
                instance_vertices = instance_vertices - normal[None, :] * dist[:, None]
                vertices[instance_mask] = instance_vertices
            else:
                non_planar_vertices[instance_mask] = True

    mesh.vertices = vertices

    # remove faces and vertices with non-planar vertices
    faces = mesh.faces.copy()
    v0 = faces[:, 0]
    v1 = faces[:, 1]
    v2 = faces[:, 2]
    remove_face_mask = non_planar_vertices[v0] | non_planar_vertices[v1] | non_planar_vertices[v2]

    mesh.update_faces(~remove_face_mask)
    mesh.update_vertices(~non_planar_vertices)

    return mesh

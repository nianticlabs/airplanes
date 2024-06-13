# type: ignore
import time
from functools import cache
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from numba import cuda
from skimage.measure import marching_cubes

from airplanes.meshing.connected_components import (
    connected_components_gpu,
    extract_labels_gpu,
    fill_noncore_cpu,
    fill_noncore_gpu,
    final_label,
    get_vertex_connectivity_gpu,
    initialize_parent_gpu,
    merge_planes_gpu,
)


@cache
def get_colormap(no_colors: int = 100000):
    # this is wrapped in a function so we can safely seed the random number generator
    # the output is cached so we don't need to recompute it every time
    torch.manual_seed(1)
    colormap = (torch.rand((3, no_colors)).numpy() * 255).astype("uint8")
    colormap[:, 0] *= 0
    return colormap


def extract_mesh_and_attributes(
    tsdf_values: np.ndarray,
    planar_logits: np.ndarray,
) -> Tuple[trimesh.Trimesh, np.ndarray, np.ndarray]:
    """Extract a mesh from the TSDF and compute edge probabilities and normals at each vertex"""

    # extract a mesh from the tsdf, masking out non planar regions
    planar_probability = F.sigmoid(torch.tensor(planar_logits)).numpy()
    tsdf_values[planar_probability < 0.25] = -1
    verts, faces, _, _ = marching_cubes(tsdf_values, single_mesh=True)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    # just use the mesh vertex normals as normals
    vertex_normals = mesh.vertex_normals.T

    return mesh, vertex_normals


def get_connected_components_split(
    mesh: trimesh.Trimesh,
    core_mask: np.ndarray,
    embeddings: np.ndarray,
    num_iterations: int = 5,
    deterministic: bool = True,
) -> np.ndarray:
    """Get connected components of a mesh. Again run union find on core points to establish initial planes.
    Then run label propagation - assigning non-core points to plane with the most similar normal. Also
    merge planes with similar normals."""
    mesh_edges = mesh.edges
    vert_norms = mesh.vertex_normals.copy()
    d_parent = cuda.device_array(mesh.vertices.shape[0], dtype=np.int32)

    block_size = 1024
    grid_size = (mesh_edges.shape[0] + block_size - 1) // block_size

    d_mesh_edges = cuda.to_device(mesh_edges)
    d_core_mask = cuda.to_device(core_mask)
    # d_edge_prob = cuda.to_device(edge_prob)

    # get vertex connectivity
    d_vert_connectivity = cuda.to_device(-np.ones((mesh.vertices.shape[0], 24), dtype=np.int32))
    get_vertex_connectivity_gpu[grid_size, block_size](d_mesh_edges, d_vert_connectivity)

    # union find for CC of core points
    initialize_parent_gpu[grid_size, block_size](mesh.vertices.shape[0], d_parent, d_core_mask)
    for _ in range(num_iterations):
        connected_components_gpu[grid_size, block_size](d_mesh_edges, d_parent, d_core_mask)

    # Set the normals of grouped planes to the mean normal
    # First extract the labelling
    d_labels = cuda.device_array(mesh.vertices.shape[0], dtype="int32")
    extract_labels_gpu[grid_size, block_size](d_parent, d_labels)
    labels = d_labels.copy_to_host()
    final_label(labels)

    # TODO: keep data on the GPU
    # now set normals of grouped points to mean normal
    for label in np.unique(labels):
        if label >= 0:
            mask = labels == label
            points = mesh.vertices[mask]
            if len(points) > 0:
                normal = np.mean(vert_norms[mask], 0)
                normal = normal / np.linalg.norm(normal)
                vert_norms[mask] = normal

    d_vert_norm = cuda.to_device(vert_norms)
    if deterministic:
        # use cpu - gpu is currently non-deterministic (but much faster)
        # Run label propagation of non-core points
        parent = d_parent.copy_to_host()
        vert_connectivity = d_vert_connectivity.copy_to_host()
        core_mask = d_core_mask.copy_to_host()
        for _ in range(20):
            fill_noncore_cpu(vert_connectivity, parent, core_mask, vert_norms)
        # merge planes
        # merge_planes_cpu(mesh_edges, parent, edge_prob, vert_norms)
        d_parent = cuda.to_device(parent)
    else:
        # Run label propagation of non-core points
        for _ in range(20):
            fill_noncore_gpu[grid_size, block_size](
                d_vert_connectivity, d_parent, d_core_mask, d_vert_norm
            )
        # merge planes
        merge_planes_gpu[grid_size, block_size](d_mesh_edges, d_parent, d_edge_prob, d_vert_norm)

    # final labelling
    extract_labels_gpu[grid_size, block_size](d_parent, d_labels)
    labels = d_labels.copy_to_host()
    final_label(labels)
    return labels


def color_mesh_from_labels(mesh: trimesh.Trimesh, labels: np.ndarray) -> trimesh.Trimesh:
    """Apply planeIds to the mesh and get a coloured mesh."""
    col = mesh.visual.vertex_colors
    colormap = get_colormap()
    col[:, :3] = colormap[:, labels + 1].T
    mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=col)
    return mesh

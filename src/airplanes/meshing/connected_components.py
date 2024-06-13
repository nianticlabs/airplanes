import warnings

import numpy as np
from numba import cuda, jit
from numba.core.errors import NumbaPerformanceWarning

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


###################
# Connected Components via Union Find
@cuda.jit(cache=True)
def initialize_parent_gpu(size, parent, core_mask):
    idx = cuda.grid(1)
    if idx < size:
        if core_mask[idx]:
            parent[idx] = idx
        else:
            parent[idx] = -1


@cuda.jit(cache=True)
def find_root_gpu(parent, node):
    if parent[node] == -1:
        return -1
    while parent[node] != node:
        node = parent[node]
    return node


@jit(nopython=True, cache=True)
def find_root_cpu(parent, node):
    if parent[node] == -1:
        return -1
    while parent[node] != node:
        node = parent[node]
    return node


@cuda.jit(cache=True)
def union_gpu(parent, node1, node2):
    root1 = find_root_gpu(parent, node1)
    root2 = find_root_gpu(parent, node2)

    if root1 < root2:
        cuda.atomic.min(parent, root2, root1)
    elif root2 < root1:
        cuda.atomic.min(parent, root1, root2)
    else:
        pass


@jit(nopython=True, cache=True)
def union_cpu(parent, node1, node2):
    root1 = find_root_cpu(parent, node1)
    root2 = find_root_cpu(parent, node2)

    if root1 < root2:
        parent[root2] = root1
    elif root2 < root1:
        parent[root1] = root2
    else:
        pass


@cuda.jit
def connected_components_gpu(mesh_edges, parent, core_mask):
    idx = cuda.grid(1)
    if idx < mesh_edges.shape[0]:
        node1, node2 = mesh_edges[idx]
        if core_mask[node1] and core_mask[node2]:
            union_gpu(parent, node1, node2)


###################


###################
# Label propagation of non-core vertices
@cuda.jit(cache=True)
def fill_noncore_gpu(vertex_connections, labels, core_mask, vert_norm):
    """
    Fill non-core vertices with the label of its neighbour with the most similar normal.
    WARNING - slightly non-deterministic!
    """
    idx = cuda.grid(1)

    if idx < vertex_connections.shape[0]:
        if ~core_mask[idx]:
            root = find_root_gpu(labels, idx)
            root_norm = vert_norm[root]
            norm = vert_norm[idx]

            best_dot = (
                -1.0
                if root == -1
                else abs(root_norm[0] * norm[0] + root_norm[1] * norm[1] + root_norm[2] * norm[2])
            )
            best_other = -1
            for offset in range(vertex_connections.shape[1]):
                other_idx = vertex_connections[idx, offset]
                if other_idx > -1:
                    other_root = find_root_gpu(labels, other_idx)
                    if other_root > -1:
                        other_norm = vert_norm[other_root]
                        dot = abs(
                            norm[0] * other_norm[0]
                            + norm[1] * other_norm[1]
                            + norm[2] * other_norm[2]
                        )
                        if dot > best_dot and dot > 0.25:
                            best_dot = dot
                            best_other = other_root
            if best_other > -1:
                labels[idx] = best_other


@jit(nopython=True)
def fill_noncore_cpu(vertex_connections, labels, core_mask, vert_norm):
    """
    Deterministic version of fill_noncore_gpu
    """
    for idx in range(vertex_connections.shape[0]):
        if ~core_mask[idx]:
            root = find_root_cpu(labels, idx)
            root_norm = vert_norm[root]
            norm = vert_norm[idx]

            best_dot = (
                -1.0
                if root == -1
                else np.abs(
                    root_norm[0] * norm[0] + root_norm[1] * norm[1] + root_norm[2] * norm[2]
                )
            )
            best_other = -1
            for offset in range(vertex_connections.shape[1]):
                other_idx = vertex_connections[idx, offset]
                if other_idx > -1:
                    other_root = find_root_cpu(labels, other_idx)
                    if other_root > -1:
                        other_norm = vert_norm[other_root]
                        dot = np.abs(
                            norm[0] * other_norm[0]
                            + norm[1] * other_norm[1]
                            + norm[2] * other_norm[2]
                        )
                        if dot > best_dot and dot > 0.75:
                            best_dot = dot
                            best_other = other_root
            if best_other > -1:
                labels[idx] = best_other


@cuda.jit(cache=True)
def merge_planes_gpu(mesh_edges, labels, edge_prob, vert_norm):
    """
    Merge connected planes with similar normals
    """
    idx = cuda.grid(1)

    if idx < mesh_edges.shape[0]:
        node1, node2 = mesh_edges[idx]
        root1 = find_root_gpu(labels, node1)
        root2 = find_root_gpu(labels, node2)

        if root1 > -1 and root2 > -1 and root1 != root2:
            prob1 = edge_prob[node1]
            prob2 = edge_prob[node2]
            norm1 = vert_norm[root1]
            norm2 = vert_norm[root2]

            if prob1 > 0.5 and prob2 > 0.5:
                dot = abs(norm1[0] * norm2[0] + norm1[1] * norm2[1] + norm1[2] * norm2[2])
                if dot > 0.8:
                    union_gpu(labels, node1, node2)


###################
# Compress Union Find tree to get labels
@cuda.jit(cache=True)
def extract_labels_gpu(parent, labels):
    idx = cuda.grid(1)
    if idx < len(parent):
        labels[idx] = find_root_gpu(parent, idx)


###################


###################
# Filter small planes
@cuda.jit(cache=True)
def get_label_counts_gpu(labels, counts):
    idx = cuda.grid(1)
    if idx < len(labels):
        label = labels[idx]
        cuda.atomic.add(counts, label, 1)


@cuda.jit(cache=True)
def filter_small_planes_gpu(labels, counts, threshold):
    idx = cuda.grid(1)
    if idx < len(labels):
        label = labels[idx]
        count = counts[label]
        if count < threshold:
            labels[idx] = -1


###################


###################
# Get contiguous labelling of planes (i.e. labels go 0 to n_planes - 1)
@jit(nopython=True)
def final_label(labels):
    label_to_idx = np.full_like(labels, -1)
    idx = 0
    pos = 0
    for label in labels:
        if label > -1:
            parent = label_to_idx[label]
            if parent == -1:
                parent = idx
                label_to_idx[label] = idx
                idx += 1
            labels[pos] = parent
        pos += 1


###################


# get vertex connectivity for mesh edges
@cuda.jit(cache=True)
def get_vertex_connectivity_gpu(mesh_edges, vertex_connections):
    idx = cuda.grid(1)
    if idx < mesh_edges.shape[0]:
        node1, node2 = mesh_edges[idx]
        for offset in range(vertex_connections.shape[1]):
            row = vertex_connections[node1]
            # prevent race conditions using cas
            old_val = cuda.atomic.cas(row, offset, -1, node2)
            if old_val == -1 or old_val == node2:
                break
        for offset in range(vertex_connections.shape[1]):
            row = vertex_connections[node2]
            # prevent race conditions using cas
            old_val = cuda.atomic.cas(row, offset, -1, node1)
            if old_val == -1 or old_val == node1:
                break

import numba
from numba import cuda

from airplanes.meshing.connected_components import union_gpu


@cuda.jit(fastmath=True)
def compute_d(points, normals, ds):
    idx = cuda.grid(1)

    if idx < points.shape[0]:
        point = points[idx]
        normal = normals[idx]
        d = -(normal[0] * point[0] + normal[1] * point[1] + normal[2] * point[2])

        ds[idx] = d


@cuda.jit(fastmath=True, cache=True)
def compute_d_optimized(points, normals, ds):
    """
    Kernel to compute the 'd' parameter of the plane equation for each point
    """
    idx = cuda.grid(1)

    if idx < points.shape[1]:  # Assuming points is now (3, N) and normals is (3, N)
        # Load data into local variables
        px = points[0, idx]
        py = points[1, idx]
        pz = points[2, idx]
        nx = normals[0, idx]
        ny = normals[1, idx]
        nz = normals[2, idx]

        # Perform the computation
        d = -(nx * px + ny * py + nz * pz)

        # Store the result
        ds[idx] = d


@cuda.jit(fastmath=True, cache=True)
def fill_inlier_matrix_dot(
    sampled_idxs,
    points,
    normals,
    ds,
    inlier_matrix,
    dist_threshold,
    normal_threshold,
    inlier_sums,
    n_rows,
    n_cols,
):
    """
    Kernel to fill the inlier matrix for ransac without embeddings, i.e. the baseline, using
    normals and distance
    """
    i, j = cuda.grid(2)
    if i < n_rows and j < n_cols:
        sampled_idx = sampled_idxs[i]
        sampled_normal = normals[sampled_idx]
        sampled_d = ds[sampled_idx]

        point = points[j]
        normal = normals[j]

        dist = abs(
            sampled_normal[0] * point[0]
            + sampled_normal[1] * point[1]
            + sampled_normal[2] * point[2]
            + sampled_d
        )
        dot = abs(
            sampled_normal[0] * normal[0]
            + sampled_normal[1] * normal[1]
            + sampled_normal[2] * normal[2]
        )

        inlier = dist < dist_threshold and dot > normal_threshold
        inlier = numba.int8(inlier)
        inlier_matrix[i, j] = inlier

        # # use atomics to compute a sum as we go and keep track of the highest scoring hypothesis
        cuda.atomic.add(inlier_sums, i, inlier)


@cuda.jit(cache=True)
def assign_nearest_label(edges, points, labels, min_distances, core_mask):
    idx = cuda.grid(1)

    if idx < edges.shape[0]:
        edge = edges[idx]
        v0 = edge[0]
        v1 = edge[1]

        label0 = labels[v0]
        label1 = labels[v1]

        p0 = points[v0]
        p1 = points[v1]

        c0 = core_mask[v0]
        c1 = core_mask[v1]

        if c0 == 0 and label1 > 0:
            dist = (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 + (p0[2] - p1[2]) ** 2
            cuda.atomic.min(min_distances, v0, dist)
            if dist == min_distances[v0]:
                labels[v0] = label1
                min_distances[v0] = dist
        elif c1 == 0 and label0 > 0:
            dist = (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 + (p0[2] - p1[2]) ** 2
            cuda.atomic.min(min_distances, v1, dist)
            if dist == min_distances[v1]:
                labels[v1] = label0
                min_distances[v1] = dist


@cuda.jit
def connected_components_labels(edges, labels, new_labels):
    idx = cuda.grid(1)
    if idx < edges.shape[0]:
        edge = edges[idx]
        v0 = edge[0]
        v1 = edge[1]

        label0 = labels[v0]
        label1 = labels[v1]

        if label0 == label1 and label0 > 0 and label1 > 0:
            union_gpu(new_labels, v0, v1)


@cuda.jit(fastmath=True, cache=True)
def fill_inlier_matrix_embeddings_optimised_shared_mem(
    sampled_idxs,
    points,
    normals,
    ds,
    inlier_matrix,
    dist_threshold,
    normal_threshold,
    embedding_threshold,
    embeddings,
    inlier_sums,
    n_rows,
    n_cols,
):
    """
    Kernel to fill the inlier matrix for ransac with embeddings -> uses shared memory
    for speed
    """
    i, j = cuda.grid(2)

    # Use shared memory for frequently accessed data
    shared_normals = cuda.shared.array(shape=(3, 1024), dtype=numba.float32)
    shared_embeddings = cuda.shared.array(shape=(3, 1024), dtype=numba.float32)
    shared_ds = cuda.shared.array(shape=(1, 1024), dtype=numba.float32)
    sm_inlier_sums = cuda.shared.array(shape=(1024), dtype=numba.int32)

    if i < n_rows and j < n_cols:
        # Load data into shared memory by the first thread of the block
        if cuda.threadIdx.y == 0:
            sampled_idx = sampled_idxs[i]
            shared_normals[0, i] = normals[0, sampled_idx]
            shared_normals[1, i] = normals[1, sampled_idx]
            shared_normals[2, i] = normals[2, sampled_idx]
            shared_embeddings[0, i] = embeddings[0, sampled_idx]
            shared_embeddings[1, i] = embeddings[1, sampled_idx]
            shared_embeddings[2, i] = embeddings[2, sampled_idx]
            shared_ds[0, i] = ds[sampled_idx]

        # Synchronize to make sure the data is loaded
        cuda.syncthreads()

        # Load data from shared memory
        sampled_nx, sampled_ny, sampled_nz = shared_normals[:, i]
        sampled_ex, sampled_ey, sampled_ez = shared_embeddings[:, i]
        sampled_d = shared_ds[0, i]

        px = points[0, j]
        py = points[1, j]
        pz = points[2, j]
        ex = embeddings[0, j]
        ey = embeddings[1, j]
        ez = embeddings[2, j]

        dist = abs((sampled_nx * px) + (sampled_ny * py) + (sampled_nz * pz) + sampled_d)
        edist = (ex - sampled_ex) ** 2 + (ey - sampled_ey) ** 2 + (ez - sampled_ez) ** 2

        inlier = dist < dist_threshold and edist <= embedding_threshold
        inlier = numba.int8(inlier)
        inlier_matrix[i, j] = inlier

        # Reduce inlier_sums using shared memory to minimize atomic operations on non shared memory
        # make sure the first thread in the block zeros the shared memory
        if cuda.threadIdx.y == 0:
            sm_inlier_sums[i] = 0
        # synchronize to make sure the zeroing is complete before we start adding
        cuda.syncthreads()

        # add the inlier to the shared memory
        cuda.atomic.add(sm_inlier_sums, i, inlier)
        # synchronize to make sure the addition is complete before we start adding to the global
        # inlier sums
        cuda.syncthreads()

        # make sure the first thread in the block adds the shared memory to the global inlier sums
        if cuda.threadIdx.y == 0:
            cuda.atomic.add(inlier_sums, i, sm_inlier_sums[i])


@cuda.jit(cache=True)
def count_occurrences(arr, counts):
    idx = cuda.grid(1)
    if idx < arr.shape[0]:
        value = arr[idx]
        cuda.atomic.add(counts, value, 1)


@cuda.jit(cache=True)
def filter_small_planes(labels, counts, min_planar_size):
    idx = cuda.grid(1)
    if idx < labels.shape[0]:
        label = labels[idx]
        count = counts[label]
        if count < min_planar_size:
            labels[idx] = 0

import collections
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import click
import numba
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import trimesh
from loguru import logger
from numba import cuda
from skimage.measure import marching_cubes
from tqdm import tqdm, trange

from airplanes.clustering.ransac_kernels import (
    assign_nearest_label,
    compute_d,
    compute_d_optimized,
    connected_components_labels,
    count_occurrences,
    fill_inlier_matrix_dot,
    fill_inlier_matrix_embeddings_optimised_shared_mem,
    filter_small_planes,
)
from airplanes.meshing.connected_components import extract_labels_gpu, final_label
from airplanes.meshing.extract_planes import color_mesh_from_labels
from airplanes.meshing.meshing_tools import squash_vertices_onto_planes
from airplanes.plane_extraction import extract_tsdf_data, extract_vertex_embeddings
from airplanes.scene_embeddings_optimisation import MLP
from airplanes.utils.generic_utils import read_scannetv2_filename
from airplanes.utils.io_utils import AssetFileNames


@dataclass
class StoppingCriteria:
    # if we start finding planes smaller than this then stop
    min_planar_size: int = 100
    # let's not find any more planes than this
    max_planes: int = 100


@dataclass
class RansacOptions:
    normal_inlier_threshold: float = 0.8
    distance_inlier_threshold: float = 0.1  ## r_d in the paper
    embeddings_inlier_threshold: float = 0.8
    num_iterations: int = 1000
    # if this is true, we guarantee that all points will end up assigned to a plane
    force_assign_points_to_planes: bool = False


class CustomSequentialRansac:
    def __init__(
        self,
        stopping_criteria: StoppingCriteria,
        ransac_options: RansacOptions,
        embeddings_usage: str,
        merge_planes_with_similar_embeddings: bool = False,
    ):
        self.stopping_criteria = stopping_criteria
        self.ransac_options = ransac_options
        self.embeddings_usage = embeddings_usage
        self.merge_planes_with_similar_embeddings = merge_planes_with_similar_embeddings

        self.inlier_mat = None
        self.d_inlier_mat = None

        self.timings = collections.defaultdict(list)

    def allocate_inlier_matrix(self, num_iterations: int, num_points: int):
        """
        Allocate memory for the inlier matrix.
        params:
            num_iterations: number of iterations to allocate for
            num_points: number of points to allocate for
        """

        self.inlier_mat = torch.zeros(
            (num_iterations, num_points), dtype=torch.int8, requires_grad=False, device="cuda"
        )
        self.d_inlier_mat = cuda.as_cuda_array(self.inlier_mat)

    def __call__(
        self,
        pcd: o3d.geometry.PointCloud,
        mesh_edges: np.ndarray,
        embeddings: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Runs sequential ransac on the given point cloud and returns a per-point label array
            and a boolean array indicating which points were assigned labels by ransac.

        Args:
            pcd (o3d.geometry.PointCloud): The point cloud to run ransac on.
            embeddings: if set, consider predicted embeddings when fitting planes

        Returns:
            np.ndarray: A per-point label array. Each point is labelled with the plane it has been
                assigned to
            np.ndarray: A boolean array indicating which points were assigned labels by ransac,
                i.e. which points are 'core points'.
        """
        # We seed numpy before each run, rather than at the start of the script.
        # This ensures results are deterministic regardless of the order we process the scenes.
        np.random.seed(10)
        torch.manual_seed(10)
        torch.cuda.manual_seed(10)

        per_point_plane_assignment = np.zeros(len(pcd.points), np.int32)
        original_points_array = np.asarray(pcd.points)
        original_normals_array = np.asarray(pcd.normals)
        original_embeddings = embeddings.copy() if embeddings is not None else None

        # We use original_idx_for_each_point to keep track of the idx each point corresponds to in
        # the original (full) point cloud. As we remove points from pcd.points, we also remove
        # points from this array. This ensures we maintain a 1:1 mapping between the values in this
        # array and the points in pcd.
        original_idx_for_each_point = np.arange(len(pcd.points))
        original_idx_for_each_point = torch.tensor(
            original_idx_for_each_point, device="cuda", dtype=torch.int32
        )

        plane_idx = 0

        points_tensor = torch.tensor(
            np.array(pcd.points), requires_grad=False, device="cuda", dtype=torch.float32
        )
        normals_tensor = torch.tensor(
            np.array(pcd.normals), requires_grad=False, device="cuda", dtype=torch.float32
        )
        if embeddings is not None:
            assert embeddings.shape[1] == 3, "embeddings must be Nx3"
            embeddings_tensor = torch.tensor(
                embeddings, requires_grad=False, device="cuda", dtype=torch.float32
            )
        else:
            embeddings_tensor = None

        mesh_edges = torch.tensor(mesh_edges, device="cuda")

        per_point_plane_assignment = torch.tensor(per_point_plane_assignment).cuda().long()
        if embeddings is not None:
            original_embeddings = torch.tensor(original_embeddings).cuda().float()
        original_normals_array = torch.tensor(original_normals_array).cuda().float()
        original_points_array = torch.tensor(original_points_array).cuda().float()

        num_points = len(points_tensor)
        num_iterations = self.ransac_options.num_iterations

        # we have a limited amount of shared memory we can use for keeping track of the
        # inlier sums. We have set it to 1024 32-bit ints in shared memory, so we need to make sure
        # we don't exceed this.
        assert num_iterations < 1024, f"num_iterations must be less than 1024, got {num_iterations}"

        self.allocate_inlier_matrix(num_iterations=num_iterations, num_points=num_points)
        for _ in trange(self.stopping_criteria.max_planes, unit="plane"):
            # Find a single plane with ransac
            if self.embeddings_usage == "ransac":
                inliers, inlier_sum = self.find_single_largest_plane_torch_cuda(
                    points=points_tensor, normals=normals_tensor, embeddings=embeddings_tensor
                )
            else:
                inliers, inlier_sum = self.find_single_largest_plane_dot_torch_cuda(
                    points=points_tensor, normals=normals_tensor
                )

            # for debugging
            assert inliers.sum() == inlier_sum, f"{inliers.sum()} != {inlier_sum}"

            # See if we want to early exit.
            if inlier_sum < self.stopping_criteria.min_planar_size:
                break

            # Update the global assignment array (we use plane_idx + 1 so that the first plane
            # starts from index 1. This means unassigned points will end up as 0.)
            this_plane_original_idxs = original_idx_for_each_point[inliers]
            per_point_plane_assignment[this_plane_original_idxs] = plane_idx + 1
            plane_idx += 1

            # Remove the points from the pcd and the original point array
            points_tensor = points_tensor[~inliers]
            normals_tensor = normals_tensor[~inliers]

            original_idx_for_each_point = original_idx_for_each_point[~inliers]

            if embeddings is not None:
                embeddings_tensor = embeddings_tensor[~inliers]

            assert len(original_idx_for_each_point) == len(
                points_tensor
            ), f"{len(original_idx_for_each_point)} != {len(points_tensor)}"

        # core plane points are the points which were given a label by ransac. We compute this
        # from the per_point_plane_assignment array now, before we potentially give all unlabelled
        # points a label (which we do if force_assign_points_to_planes is True)
        core_plane_points = (per_point_plane_assignment != 0).cpu().numpy()

        if self.merge_planes_with_similar_embeddings:
            per_point_plane_assignment = self.merge_planes(
                per_point_plane_assignment,
                original_points_array,
                original_normals_array,
                original_embeddings,
            )
        # run CC to split up planes which are not connected
        per_point_plane_assignment = self.refine_labels_with_cc(
            mesh_edges=mesh_edges,
            labels=per_point_plane_assignment,
        )
        if self.ransac_options.force_assign_points_to_planes:
            # use embeddings to help assign unlabelled points to planes if we have them
            if self.embeddings_usage == "ransac":
                unlabelled_assignment_embeddings = original_embeddings
            else:
                unlabelled_assignment_embeddings = original_points_array

            per_point_plane_assignment = self.assign_unlabelled_points_to_plane_cuda(
                points=original_points_array,
                labels=per_point_plane_assignment,
                mesh_edges=mesh_edges,
                embeddings=unlabelled_assignment_embeddings,
            )

        per_point_plane_assignment = self.remove_small_planes_cuda(per_point_plane_assignment)

        per_point_plane_assignment = per_point_plane_assignment.cpu().numpy()

        return per_point_plane_assignment, core_plane_points

    def find_single_largest_plane_torch_cuda(
        self, points: torch.tensor, normals: torch.tensor, embeddings: Optional[torch.tensor]
    ) -> np.ndarray:
        """
        run ransac on the given point cloud and return the inliers for the largest plane found
        params:
            points: Nx3 tensor of points
            normals: Nx3 tensor of normals
            embeddings: Nx3 tensor of embeddings
        """
        num_points = len(points)

        num_iterations = self.ransac_options.num_iterations
        distance_inlier_threshold = self.ransac_options.distance_inlier_threshold
        normal_inlier_threshold = self.ransac_options.normal_inlier_threshold
        embedding_inlier_threshold = self.ransac_options.embeddings_inlier_threshold

        # Allocate memory for the result
        # we use inlier_sums to keep track of the best hypothesis
        ds = torch.zeros((num_points), dtype=torch.float32, requires_grad=False, device="cuda")
        sample_idxs = torch.randint(
            size=(num_iterations,),
            low=0,
            high=num_points - 1,
            requires_grad=False,
            device="cuda",
            dtype=torch.int32,
        )
        inlier_sums = torch.zeros(
            (num_iterations), dtype=torch.int32, requires_grad=False, device="cuda"
        )

        # Define block and grid dimensions for compute_d
        threads_per_block_1d = 1024
        blocks_per_grid_1d = (num_points + threads_per_block_1d - 1) // threads_per_block_1d

        # Define block and grid dimensions for fill_inlier_matrix
        threads_per_block = (8, 64)
        blocks_per_grid_x = (num_iterations + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (num_points + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        # Copy data to device
        # we need to transpose the points and normals as this encourages coalesced memory access
        d_points = cuda.as_cuda_array(points.permute(1, 0))
        d_normals = cuda.as_cuda_array(normals.permute(1, 0))

        # we need to transpose the embeddings as this encourages coalesced memory access
        d_embeddings = cuda.as_cuda_array(embeddings.permute(1, 0))

        d_ds = cuda.as_cuda_array(ds)
        d_sample_idxs = cuda.as_cuda_array(sample_idxs)
        d_inlier_sums = cuda.as_cuda_array(inlier_sums)

        cuda.synchronize()

        # Launch the kernel
        compute_d_optimized[blocks_per_grid_1d, threads_per_block_1d](d_points, d_normals, d_ds)

        fill_inlier_matrix_embeddings_optimised_shared_mem[blocks_per_grid, threads_per_block](
            d_sample_idxs,
            d_points,
            d_normals,
            d_ds,
            self.d_inlier_mat,
            numba.float32(distance_inlier_threshold),
            numba.float32(normal_inlier_threshold),
            numba.float32(
                embedding_inlier_threshold**2
            ),  # Pass squared threshold to avoid sqrt in kernel
            d_embeddings,
            d_inlier_sums,
            numba.int32(num_iterations),  # n_rows
            numba.int32(num_points),  # n_cols
        )

        # Synchronize to ensure all GPU work is finished
        cuda.synchronize()

        # get the best iteration, we use the last column as it is the sum of inliers
        best_iter = inlier_sums.argmax()
        torch.cuda.synchronize()

        # get the inliers for the best iteration, we ignore the last column as it is the sum of inliers
        best_plane_inliers = self.inlier_mat[best_iter, :num_points]
        torch.cuda.synchronize()

        return best_plane_inliers.bool(), inlier_sums[best_iter]

    def find_single_largest_plane_dot_torch_cuda(
        self,
        points: torch.tensor,
        normals: torch.tensor,
    ) -> torch.tensor:
        """
        run ransac on the given point cloud and return the inliers for the largest plane found
        params:
            points: Nx3 tensor of points
            normals: Nx3 tensor of normals
            embeddings: Nx3 tensor of embeddings
        """
        num_points = len(points)

        num_iterations = self.ransac_options.num_iterations
        distance_inlier_threshold = self.ransac_options.distance_inlier_threshold
        normal_inlier_threshold = self.ransac_options.normal_inlier_threshold

        # Allocate memory for the result
        # we use inlier_sums to keep track of the best hypothesis
        ds = torch.zeros((num_points), dtype=torch.float32, requires_grad=False, device="cuda")
        sample_idxs = torch.randint(
            size=(num_iterations,),
            low=0,
            high=num_points - 1,
            requires_grad=False,
            device="cuda",
            dtype=torch.int32,
        )
        inlier_sums = torch.zeros(
            (num_iterations), dtype=torch.int32, requires_grad=False, device="cuda"
        )

        # Define block and grid dimensions for compute_d
        threads_per_block_1d = 1024
        blocks_per_grid_1d = (num_points + threads_per_block_1d - 1) // threads_per_block_1d

        # Define block and grid dimensions for fill_inlier_matrix
        threads_per_block = (8, 32)
        blocks_per_grid_x = (num_iterations + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (num_points + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        # Copy data to device
        d_points = cuda.as_cuda_array(points)
        d_normals = cuda.as_cuda_array(normals)

        d_ds = cuda.as_cuda_array(ds)
        d_sample_idxs = cuda.as_cuda_array(sample_idxs)
        d_inlier_sums = cuda.as_cuda_array(inlier_sums)

        cuda.synchronize()

        # Launch the kernel
        compute_d[blocks_per_grid_1d, threads_per_block_1d](d_points, d_normals, d_ds)

        fill_inlier_matrix_dot[blocks_per_grid, threads_per_block](
            d_sample_idxs,
            d_points,
            d_normals,
            d_ds,
            self.d_inlier_mat,
            numba.float32(distance_inlier_threshold),
            numba.float32(normal_inlier_threshold),
            d_inlier_sums,
            numba.int32(num_iterations),  # n_rows
            numba.int32(num_points),  # n_cols
        )

        # Synchronize to ensure all GPU work is finished
        cuda.synchronize()

        # get the best iteration, we use the last column as it is the sum of inliers
        best_iter = inlier_sums.argmax()
        torch.cuda.synchronize()

        # get the inliers for the best iteration, we ignore the last column as it is the sum of inliers
        best_plane_inliers = self.inlier_mat[best_iter, :num_points]

        return best_plane_inliers.bool(), inlier_sums[best_iter]

    @staticmethod
    def assign_unlabelled_points_to_plane_cuda(
        points: torch.Tensor,
        labels: torch.Tensor,
        mesh_edges: torch.Tensor,
        embeddings: torch.Tensor,
        iterations=10,
    ):
        min_dist = torch.ones((points.shape[0],), dtype=torch.float32, device="cuda") * 10.0
        core_mask = (labels > 0).int()

        threads_per_block = 1024
        blocks_per_grid = (mesh_edges.shape[0] + threads_per_block - 1) // threads_per_block

        for _ in range(iterations):
            assign_nearest_label[blocks_per_grid, threads_per_block](
                cuda.as_cuda_array(mesh_edges),
                cuda.as_cuda_array(embeddings),
                cuda.as_cuda_array(labels),
                cuda.as_cuda_array(min_dist),
                cuda.as_cuda_array(core_mask),
            )

        return labels

    def merge_planes(
        self,
        per_point_plane_assignment: torch.Tensor,
        original_points_array: torch.Tensor,
        original_normals_array: torch.Tensor,
        original_embeddings: Optional[torch.Tensor],
    ):
        if self.embeddings_usage == "none":
            original_embeddings = torch.zeros((len(original_points_array), 3), device="cuda")

        # get mean embeddings per plane
        max_label = per_point_plane_assignment.max() + 1
        mean_normal = torch.zeros((max_label, 3), device="cuda")
        mean_offset = torch.zeros((max_label, 1), device="cuda")
        mean_embedding = torch.zeros((max_label, original_embeddings.shape[-1]), device="cuda")

        for label in range(max_label):
            if label > 0:
                mask = per_point_plane_assignment == label
                # compute plane parameters and move all points onto the plane
                plane_points = original_points_array[mask]
                plane_normal = torch.median(original_normals_array[mask], dim=0)[0]
                normal = plane_normal / torch.norm(plane_normal)

                offset = -torch.median((plane_points * normal[None, :]).sum(-1), dim=0)[0]

                mean_normal[label] = normal
                mean_offset[label] = offset
                mean_embedding[label] = torch.median(original_embeddings[mask], dim=0)[0]

        # create final inlier matrix
        embeddings_diff = torch.sqrt(
            ((mean_embedding[None, :] - mean_embedding[:, None]) ** 2).sum(2)
        )
        normal_diff = (mean_normal[None, :] * mean_normal[:, None]).sum(2)
        offset_diff = torch.abs(mean_offset[None, :] - mean_offset[:, None])[..., 0]

        if self.embeddings_usage == "ransac":
            inlier_matrix = ((embeddings_diff < 0.2) * (normal_diff > 0.6)).long()
        else:
            inlier_matrix = ((normal_diff > 0.8) * (offset_diff < 0.3)).long()

        for label in range(max_label):
            if label > 0:
                mask = per_point_plane_assignment == label
                per_point_plane_assignment[mask] = inlier_matrix[label].argmax()

        return per_point_plane_assignment

    def refine_labels_with_cc(
        self,
        mesh_edges,
        labels,
    ):
        new_labels = torch.arange(labels.shape[0], device="cuda").int()
        new_labels[labels == 0] = -1

        threads_per_block = 1024
        blocks_per_grid = (mesh_edges.shape[0] + threads_per_block - 1) // threads_per_block

        for _ in range(5):
            connected_components_labels[blocks_per_grid, threads_per_block](
                cuda.as_cuda_array(mesh_edges),
                cuda.as_cuda_array(labels),
                cuda.as_cuda_array(new_labels),
            )

        blocks_per_grid = (labels.shape[0] + threads_per_block - 1) // threads_per_block
        extract_labels_gpu[blocks_per_grid, threads_per_block](
            cuda.as_cuda_array(new_labels), cuda.as_cuda_array(labels)
        )

        labels = labels.cpu().numpy()
        final_label(labels)

        return torch.tensor(labels + 1).cuda()

    def remove_small_planes_cuda(self, labels):
        counts = torch.zeros((labels.shape[0]), device="cuda", dtype=torch.int32)
        threads_per_block = 1024
        blocks_per_grid = (labels.shape[0] + threads_per_block - 1) // threads_per_block
        count_occurrences[blocks_per_grid, threads_per_block](labels, counts)
        filter_small_planes[blocks_per_grid, threads_per_block](
            labels, counts, self.stopping_criteria.min_planar_size
        )
        return labels


def run_ransac_on_mesh(
    mesh: trimesh.Trimesh,
    out_ply_file: Path,
    embeddings: Optional[np.ndarray],
    embeddings_usage: str,
    force_assign_points_to_planes: bool,
    normals_inlier_threshold: float,
    embeddings_inlier_threshold: float,
    merge_planes_with_similar_embeddings: bool,
) -> None:
    # Convert mesh to a point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(mesh.vertices)
    point_cloud.normals = o3d.utility.Vector3dVector(mesh.vertex_normals)

    # Run ransac
    ransac_plane_finder = CustomSequentialRansac(
        stopping_criteria=StoppingCriteria(),
        ransac_options=RansacOptions(
            force_assign_points_to_planes=force_assign_points_to_planes,
            normal_inlier_threshold=normals_inlier_threshold,
            embeddings_inlier_threshold=embeddings_inlier_threshold,
        ),
        embeddings_usage=embeddings_usage,
        merge_planes_with_similar_embeddings=merge_planes_with_similar_embeddings,
    )

    per_point_labels, core_plane_points = ransac_plane_finder(
        pcd=point_cloud, embeddings=embeddings, mesh_edges=mesh.edges
    )

    # make colourised planar mesh
    # We subtract 1 from the labels so zero-indexed points are coloured black.
    mesh = color_mesh_from_labels(mesh=mesh, labels=per_point_labels - 1)

    # Save a squashed version
    planar_mesh = squash_vertices_onto_planes(mesh, core_mask=core_plane_points)

    result = trimesh.exchange.ply.export_ply(planar_mesh, encoding="ascii")
    out_ply_file.parent.mkdir(exist_ok=True)

    with open(out_ply_file, "wb+") as fh:
        fh.write(result)
        fh.close()


def extract_mesh_from_tsdf(tsdf_path: Path, voxel_planar_threshold: float) -> trimesh.Trimesh:
    """Extracts a mesh from a tsdf, optionally removing points that are not planar.

    Args:
        tsdf_path (Path): Path to the tsdf file
        voxel_planar_threshold (float): The threshold above which a voxel is considered planar

    Returns:
        trimesh.Trimesh: A 3D mesh
    """
    tsdf_dict = np.load(tsdf_path)
    tsdf_values, planar_logits, origin = extract_tsdf_data(tsdf_dict)

    # extract a mesh from the tsdf, masking out non planar regions
    planar_probability = F.sigmoid(torch.tensor(planar_logits)).numpy()
    tsdf_values[planar_probability < voxel_planar_threshold] = -1
    verts, faces, _, _ = marching_cubes(tsdf_values, single_mesh=True)
    # scale vertices to world coordinates

    voxel_size = float(tsdf_dict["voxel_size"])
    verts = origin.reshape(1, 3) + verts * voxel_size
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    return mesh


def get_mesh(
    pred_root: Path,
    scene: str,
    file_type: str,
    voxel_planar_threshold: float,
) -> trimesh.Trimesh:
    """
    Extract a mesh from a file
    params:
        pred_root: path to the predicted meshes
        scene: scene name
        file_type: one of "tsdf" or "mesh"
        voxel_planar_threshold: threshold above which to be considered a planar voxel
    """
    assert file_type in ("tsdf", "mesh")

    if file_type == "tsdf":
        filename = pred_root / scene / AssetFileNames.get_tsdf_filename(scene)
        mesh = extract_mesh_from_tsdf(filename, voxel_planar_threshold=voxel_planar_threshold)
        return mesh

    filename = pred_root / scene / AssetFileNames.get_pred_mesh_filename(scene)
    mesh = trimesh.exchange.load.load(filename)
    return mesh


def get_embeddings(
    pred_root: Path,
    scene: str,
    mesh: trimesh.Trimesh,
    embeddings_usage: str,
    num_harmonics: int,
    embedding_dim: int,
    embeddings_scale_factor: float,
) -> Optional[np.ndarray]:
    """
    Extract embeddings from a mesh
    params:
        mesh: trimesh mesh
        embeddings_usage: one of "none" or "ransac"
        num_harmonics: number of harmonics to use
        embedding_dim: embedding dimension
    """
    embeddings = None
    if embeddings_usage == "ransac":
        mlp = MLP(num_harmonics=num_harmonics, num_outputs=embedding_dim).cuda()
        mlp.load_state_dict(
            torch.load(
                pred_root
                / scene
                / AssetFileNames.get_embeddings_mlp_filename(scene, num_harmonics, embedding_dim)
            )
        )
        embeddings = extract_vertex_embeddings(mlp, mesh.vertices) * embeddings_scale_factor
    return embeddings


def run_ransac_on_scenes(
    scenes: list[str],
    pred_root: Path,
    output_dir: Path,
    file_type: str,
    voxel_planar_threshold: float,
    embeddings_usage: str,
    num_harmonics: int,
    embedding_dim: int,
    force_assign_points: bool,
    embeddings_scale_factor: float,
    embeddings_inlier_threshold: float,
    normal_inlier_threshold: float,
    merge_planes_with_similar_embeddings: bool,
) -> None:
    for scene in tqdm(scenes, unit="scan", desc="All scans"):
        mesh = get_mesh(
            pred_root=pred_root,
            scene=scene,
            file_type=file_type,
            voxel_planar_threshold=voxel_planar_threshold,
        )

        embeddings = get_embeddings(
            pred_root=pred_root,
            scene=scene,
            mesh=mesh,
            embeddings_usage=embeddings_usage,
            num_harmonics=num_harmonics,
            embedding_dim=embedding_dim,
            embeddings_scale_factor=embeddings_scale_factor,
        )

        out_ply_file = output_dir / scene / AssetFileNames.get_planar_mesh_filename(scene)
        run_ransac_on_mesh(
            mesh=mesh,
            out_ply_file=out_ply_file,
            embeddings_usage=embeddings_usage,
            embeddings=embeddings,
            force_assign_points_to_planes=force_assign_points,
            normals_inlier_threshold=normal_inlier_threshold,
            embeddings_inlier_threshold=embeddings_inlier_threshold,
            merge_planes_with_similar_embeddings=merge_planes_with_similar_embeddings,
        )


@click.command()
@click.option("--pred_root", type=Path, help="Path to predicted tsdfs", required=True)
@click.option(
    "--output-dir",
    type=Path,
    help="folder that contains final results",
    default=None,
)
@click.option(
    "--validation-file",
    type=Path,
    default=Path("src/airplanes/data_splits/ScanNetv2/standard_split/scannetv2_test_planes.txt"),
    help="Path to the file that contains the test scenes",
)
@click.option(
    "--file-type",
    type=click.Choice(["tsdf", "mesh"]),
    default="tsdf",
    help="File type to run ransac on, either tsdf (npz) or mesh (ply)",
)
@click.option(
    "--voxel-planar-threshold",
    type=float,
    default=0.0,
    help="Threshold above which to be considered a planar voxel",
)
@click.option(
    "--embeddings-usage",
    type=click.Choice(["none", "ransac"]),
    default="none",
)
@click.option(
    "--num-harmonics",
    type=int,
    default=24,
)
@click.option(
    "--embedding-dim",
    type=int,
    default=3,
)
@click.option(
    "--force-assign-points",
    is_flag=True,
    help=(
        "If set, this will force all points to be assigned to a plane, even if they are not "
        "Assigned to a plane in the RANSAC step."
    ),
)
@click.option(
    "--embeddings-scale-factor",
    type=float,
    default=1.0,
)
@click.option(
    "--embeddings-inlier-threshold",
    type=float,
    default=0.8,
)
@click.option(
    "--normal-inlier-threshold",
    type=float,
    default=0.8,
)
@click.option(
    "--merge-planes-with-similar-embeddings",
    is_flag=True,
)
def cli(
    pred_root: Path,
    output_dir: Path,
    validation_file: Path,
    file_type: str,
    voxel_planar_threshold: float,
    embeddings_usage: str,
    num_harmonics: int,
    embedding_dim: int,
    force_assign_points: bool,
    embeddings_scale_factor: float,
    embeddings_inlier_threshold: float,
    normal_inlier_threshold: float,
    merge_planes_with_similar_embeddings: bool,
):
    """
    Extract planar meshes from ply files using sequential ransac
    """
    if file_type == "mesh" and voxel_planar_threshold > 0:
        raise ValueError("Voxel planar threshold not supported for mesh files")

    if embeddings_usage == "gt" and file_type != "mesh":
        raise ValueError("Embeddings usage gt only supported for mesh files")

    if embeddings_usage == "ransac":
        logger.info(f"Using embeddings in ransac")

    scenes = read_scannetv2_filename(validation_file)

    savepath = pred_root if output_dir is None else output_dir
    savepath.mkdir(parents=True, exist_ok=True)

    run_ransac_on_scenes(
        scenes=scenes,
        pred_root=pred_root,
        output_dir=savepath,
        file_type=file_type,
        voxel_planar_threshold=voxel_planar_threshold,
        embeddings_usage=embeddings_usage,
        num_harmonics=num_harmonics,
        embedding_dim=embedding_dim,
        force_assign_points=force_assign_points,
        embeddings_scale_factor=embeddings_scale_factor,
        embeddings_inlier_threshold=embeddings_inlier_threshold,
        normal_inlier_threshold=normal_inlier_threshold,
        merge_planes_with_similar_embeddings=merge_planes_with_similar_embeddings,
    )


if __name__ == "__main__":
    cli()

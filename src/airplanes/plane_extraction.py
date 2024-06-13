from pathlib import Path
from typing import Dict, Tuple

import click
import numpy as np
import torch
import trimesh
from loguru import logger
from tqdm import tqdm

from airplanes.meshing.extract_planes import (
    color_mesh_from_labels,
    extract_mesh_and_attributes,
    get_connected_components_split,
)
from airplanes.meshing.meshing_tools import squash_vertices_onto_planes
from airplanes.scene_embeddings_optimisation import MLP
from airplanes.utils.generic_utils import read_scannetv2_filename
from airplanes.utils.io_utils import AssetFileNames


def extract_vertex_embeddings(mlp: MLP, vertices_N3: np.ndarray):
    vertices_t = torch.tensor(vertices_N3).cuda().unsqueeze(0).float()
    embeddings_N3 = mlp(vertices_t).squeeze(0).detach().cpu().numpy()
    return embeddings_N3


def sample_volume(
    volume_values_chwd: torch.Tensor,
    origin: torch.Tensor,
    voxel_size: float,
    world_points_N3: torch.Tensor,
):
    """
    Expects tensors.
    """
    assert len(volume_values_chwd.shape) == 4, "volume_values_chwd should be of shape (C, H, W, D)"

    # convert world coordinates to voxel coordinates
    voxel_coords_N3 = world_points_N3 - origin.view(1, 3)
    voxel_coords_N3 = voxel_coords_N3 / voxel_size

    # divide by the volume_size - 1 for align corners True!
    dim_size_3 = torch.tensor(
        volume_values_chwd.shape[1:],
        dtype=world_points_N3.dtype,
        device=world_points_N3.device,
    )
    voxel_coords_N3 = voxel_coords_N3 / (dim_size_3.view(1, 3) - 1)
    # convert from 0-1 to [-1, 1] range
    voxel_coords_N3 = voxel_coords_N3 * 2 - 1
    voxel_coords_111N3 = voxel_coords_N3[None, None, None]

    # sample the volume
    # grid_sample expects y, x, z and we have x, y, z
    # swap the axes of the coords to match the pytorch grid_sample convention
    voxel_coords_111N3 = voxel_coords_111N3[:, :, :, :, [2, 1, 0]]

    # in case we're asked to support fp16 and cpu, we need to cast to fp32 for the
    # grid_sample call
    if volume_values_chwd.device == torch.device("cpu"):
        tensor_dtype = torch.float32
    else:
        tensor_dtype = volume_values_chwd.dtype

    values_cN = (
        torch.nn.functional.grid_sample(
            volume_values_chwd.unsqueeze(0).type(tensor_dtype),
            voxel_coords_111N3.type(tensor_dtype),
            align_corners=True,
        )
        .squeeze(0)
        .flatten(1, -1)
    )

    return values_cN


def generate_planar_mesh(
    tsdf_values: np.ndarray,
    planar_probabilities: np.ndarray,
    mlp: MLP,
    normal_threshold: float = 0.9,
) -> trimesh.Trimesh:
    """Generate the mesh that encodes the planar representation of the scene starting from
    a TSDF volume and the probabilities of being a 3D edge or a plane.

    Params:
        tsdf_values:
    """
    mesh, vertex_normals = extract_mesh_and_attributes(tsdf_values, planar_probabilities)

    embeddings = extract_vertex_embeddings(mlp, mesh.vertices)

    v0 = mesh.edges[:, 0]
    v1 = mesh.edges[:, 1]
    res1, res2 = embeddings[v0], embeddings[v1]
    edge_mask = np.linalg.norm(res1 - res2, axis=1) < 0.5
    core_mask = np.zeros((mesh.vertices.shape[0])).astype(bool)
    core_mask[v0[edge_mask]] = True
    core_mask[v1[edge_mask]] = True

    labels = get_connected_components_split(mesh, core_mask, embeddings)

    mesh = color_mesh_from_labels(mesh, labels)
    planar_mesh = squash_vertices_onto_planes(mesh, core_mask=None)

    return planar_mesh


def extract_tsdf_data(
    tsdf_dict: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract tsdf data from the tsdf dict - stores numpy arrays
    Params:
        tsdf_dict: dict with TSDF features
    Returns:
        tsdf values, logits of planar mask, origin of the volume
    """
    tsdf_values = tsdf_dict["tsdf_values"]
    origin = tsdf_dict["origin"]
    planar_logits = tsdf_dict["feature_grid"][0]
    return tsdf_values, planar_logits, origin


def extract_planes_from_tsdf(
    tsdf_path: Path,
    output_filepath: Path,
    mlp: MLP,
    normal_threshold: float = 0.9,
) -> None:
    """Extract planes from a TSDF and save them as a mesh
    Params:
        tsdf_path: path to TSDF
        output_filepath: output path for the mesh.
        edge_threshold: threshold for the edge check
        normal_threshold: threshold for the normal check
    """
    tsdf_dict = np.load(tsdf_path)
    tsdf_values, planar_logits, origin = extract_tsdf_data(tsdf_dict)
    planar_mesh = generate_planar_mesh(
        tsdf_values=tsdf_values,
        planar_probabilities=planar_logits,
        mlp=mlp,
        normal_threshold=normal_threshold,
    )

    # scale the planar_mesh to world coordinates
    voxel_size = float(tsdf_dict["voxel_size"])

    scaled_verts = origin.reshape(1, 3) + planar_mesh.vertices * voxel_size
    planar_mesh = trimesh.Trimesh(
        vertices=scaled_verts,
        faces=planar_mesh.faces,
        vertex_colors=planar_mesh.visual.vertex_colors,
    )

    result = trimesh.exchange.ply.export_ply(planar_mesh, encoding="ascii")
    with open(output_filepath, "wb+") as fh:
        fh.write(result)
        fh.close()


@click.command()
@click.option(
    "-pred_root",
    type=Path,
    help="Path to predicted data",
)
@click.option(
    "-normal_threshold",
    type=float,
    default=0.9,
)
@click.option(
    "-validation-file",
    type=Path,
    default=Path("src/airplanes/data_splits/ScanNetv2/standard_split/scannetv2_test_planes.txt"),
    help="Path to the file that contains the test scenes",
)
@click.option(
    "-num-harmonics",
    type=int,
    default=24,
)
@click.option(
    "-embedding-dim",
    type=int,
    default=3,
)
def cli(
    pred_root: Path,
    normal_threshold: float,
    validation_file: Path,
    num_harmonics: int,
    embedding_dim: int,
):
    """
    Extract planar meshes from predicted tsdfs
    """

    # select only eval scenes
    scenes = sorted(read_scannetv2_filename(filepath=validation_file))

    logger.info(f"Running plane extraction on {len(scenes)} tsdfs!")
    for scene_name in tqdm(scenes):
        scene_path = pred_root / scene_name
        planar_mesh_output_path = scene_path / AssetFileNames.get_planar_mesh_filename(scene_name)
        tsdf_filepath = scene_path / AssetFileNames.get_tsdf_filename(scene_name)

        mlp = MLP(num_harmonics=num_harmonics, num_outputs=embedding_dim).cuda()
        mlp.load_state_dict(
            torch.load(
                pred_root
                / scene_name
                / AssetFileNames.get_embeddings_mlp_filename(
                    scene_name, num_harmonics, embedding_dim
                )
            )
        )

        print(planar_mesh_output_path)
        extract_planes_from_tsdf(
            tsdf_path=tsdf_filepath,
            output_filepath=planar_mesh_output_path,
            mlp=mlp,
            normal_threshold=normal_threshold,
        )


if __name__ == "__main__":
    cli()  # type: ignore

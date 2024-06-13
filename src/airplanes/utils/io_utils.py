from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from airplanes.utils.volume_utils import SimpleVolume


class AssetFileNames:
    """Class for asset names suffixes"""

    GT_MESH = "mesh_with_planes.ply"
    PRED_MESH = "{}_pred_mesh.ply"
    PLANAR_MESH = "{}_planar_mesh.ply"
    PLANES = "{}_planes.npy"
    VISIBILITY_VOLUME = "{}_volume.npz"
    INSTANCES = "{}_plane_instances.txt"
    TSDF = "{}_tsdf.npz"
    EMBEDDINGS_MLP = "{}_embeddings_mlp_{}h_{}d.pth"
    GT_RENDERED_PLANE_IMAGE = "{}_planes.png"

    @staticmethod
    def get_gt_mesh_filename(scene_name: str) -> str:
        """Get the gt mesh filename for a scene"""
        return AssetFileNames.GT_MESH

    @staticmethod
    def get_pred_mesh_filename(scene_name: str) -> str:
        """Get the pred mesh filename for a scene"""
        return AssetFileNames.PRED_MESH.format(scene_name)

    @staticmethod
    def get_planar_mesh_filename(scene_name: str) -> str:
        """Get the planar mesh filename for a scene"""
        return AssetFileNames.PLANAR_MESH.format(scene_name)

    @staticmethod
    def get_planes_filename(scene_name: str) -> str:
        """Get the planes filename for a scene"""
        return AssetFileNames.PLANES.format(scene_name)

    @staticmethod
    def get_visibility_volume_filename(scene_name: str) -> str:
        """Get the visibility volume filename for a scene"""
        return AssetFileNames.VISIBILITY_VOLUME.format(scene_name)

    @staticmethod
    def get_plane_instances_filename(scene_name: str) -> str:
        """Get the visibility volume filename for a scene"""
        return AssetFileNames.INSTANCES.format(scene_name)

    @staticmethod
    def get_tsdf_filename(scene_name: str) -> str:
        """Get the visibility volume filename for a scene"""
        return AssetFileNames.TSDF.format(scene_name)

    @staticmethod
    def get_embeddings_mlp_filename(scene_name: str, num_harmonics: int, embedding_dim: int) -> str:
        """Get the embedding filename for a given scene"""
        return AssetFileNames.EMBEDDINGS_MLP.format(scene_name, num_harmonics, embedding_dim)

    @staticmethod
    def get_gt_plane_image_filename(frame_id: str) -> str:
        """Get the filename to images with ground truth rendered plane ids"""
        return AssetFileNames.GT_RENDERED_PLANE_IMAGE.format(frame_id)


@dataclass
class MeshingBundle:
    scene_root: Path
    scene: str
    planes: Optional[np.ndarray] = None
    planar_mesh: Optional[trimesh.Trimesh] = None
    mesh: Optional[trimesh.Trimesh] = None
    visibility_volume: Optional[SimpleVolume] = None
    instances: Optional[np.ndarray] = None


def load_gt_bundle(scene_path: Path, scene_name: str, load_visibility_volumes: bool = True):
    """
    Load a mesh from a file
    Params:
        scene_path: path to the scene directory
        scene_name: name of the scene
        load_visibility_volumes: whether or not to load the visibility volume
    """
    # explicitly load all of these files with little forgiveness to missing files.
    # forces us to use the correct version of files we share among us.
    mesh_filepath = scene_path / AssetFileNames.get_gt_mesh_filename(scene_name)
    mesh = trimesh.exchange.load.load(str(mesh_filepath), process=False)
    planes_path = scene_path / AssetFileNames.get_planes_filename(scene_name)
    volume_path = scene_path / AssetFileNames.get_visibility_volume_filename(scene_name)

    planes = np.array(np.load(planes_path))
    # planeRCNN saves meshes with n * d -> we want n / d
    d = np.linalg.norm(planes, axis=1)
    planes[d > 0] /= d[d > 0, None] ** 2

    visibility_volume = None
    if load_visibility_volumes and volume_path.exists():
        visibility_volume = SimpleVolume.load(volume_path)

    instances_filepath = scene_path / AssetFileNames.get_plane_instances_filename(scene_name)
    instances = None
    if instances_filepath.exists():
        instances = np.array(np.loadtxt(instances_filepath), dtype="int32")

    return MeshingBundle(
        scene=scene_name,
        scene_root=scene_path,
        mesh=mesh,
        planes=planes,
        visibility_volume=visibility_volume,
        instances=instances,
    )


def load_pred_bundle(scene_path: Path, scene_name: str):
    """
    Load a mesh from a file
    Params:
        scene_path: path to the scene directory
        scene_name: name of the scene
    """
    # check if the pred mesh exists. Some methods might only produce a planar mesh
    # like PlanarRecon
    mesh_filepath = scene_path / AssetFileNames.get_pred_mesh_filename(scene_name)
    mesh = None
    if mesh_filepath.exists():
        mesh = trimesh.exchange.load.load(str(mesh_filepath), process=False)

    # check if the planar mesh exists.
    planar_mesh_filepath = scene_path / AssetFileNames.get_planar_mesh_filename(scene_name)
    planar_mesh = None
    if planar_mesh_filepath.exists():
        planar_mesh = trimesh.exchange.load.load(str(planar_mesh_filepath), process=False)

    planes_path = scene_path / AssetFileNames.get_planes_filename(scene_name)
    # check if the planes exist. Some methods might only produce a planar mesh instead of
    # producing intermediate plane params
    planes = None
    if planes_path.exists():
        planes = np.array(np.load(planes_path))
        # planeRCNN saves meshes with n * d -> we want n / d
        d = np.linalg.norm(planes, axis=1)
        planes[d > 0] /= d[d > 0, None] ** 2

    instances_filepath = scene_path / AssetFileNames.get_plane_instances_filename(scene_name)
    instances = None
    if instances_filepath.exists():
        instances = np.array(np.loadtxt(instances_filepath), dtype="int32")

    return MeshingBundle(
        scene=scene_name,
        scene_root=scene_path,
        mesh=mesh,
        planar_mesh=planar_mesh,
        planes=planes,
        instances=instances,
    )

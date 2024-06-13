from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
import torch
import trimesh
from loguru import logger

from airplanes.datasets.scannet_dataset import ScannetDataset
from airplanes.meshing.tsdf import TSDF, TSDFFuser


class DepthFuser:
    def __init__(
        self,
        gt_path: Optional[str] = None,
        fusion_resolution: float = 0.04,
        max_fusion_depth: float = 3.0,
        fuse_color: bool = False,
    ):
        self.gt_path = gt_path
        self.fusion_resolution = fusion_resolution
        self.max_fusion_depth = max_fusion_depth


class Open3DFuser(DepthFuser):
    """
    Wrapper class for the open3d fuser.

    This wrapper does not support fusion of tensors with higher than batch 1.
    """

    def __init__(
        self,
        gt_path: Optional[str] = None,
        fusion_resolution: float = 0.04,
        max_fusion_depth: float = 3,
        fuse_color: bool = False,
        use_upsample_depth: bool = False,
    ):
        super().__init__(
            gt_path=gt_path,
            fusion_resolution=fusion_resolution,
            max_fusion_depth=max_fusion_depth,
            fuse_color=fuse_color,
        )

        self.fuse_color = fuse_color
        self.use_upsample_depth = use_upsample_depth
        self.fusion_max_depth = max_fusion_depth

        voxel_size = fusion_resolution * 100
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=float(voxel_size) / 100,
            sdf_trunc=3 * float(voxel_size) / 100,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

    def fuse_frames(
        self,
        depths_b1hw: torch.Tensor,
        K_b44: torch.Tensor,
        cam_T_world_b44: torch.Tensor,
        color_b3hw: torch.Tensor,
    ):
        width = depths_b1hw.shape[-1]
        height = depths_b1hw.shape[-2]

        if self.fuse_color:
            color_b3hw = torch.nn.functional.interpolate(
                color_b3hw, size=(height, width), mode="nearest"
            )

        for batch_index in range(depths_b1hw.shape[0]):
            if self.fuse_color:
                image_i = color_b3hw[batch_index].permute(1, 2, 0)

                color_im = (image_i * 255).cpu().numpy().astype("uint8").copy(order="C")
            else:
                # mesh will now be grey
                color_im = (
                    0.7 * torch.ones_like(depths_b1hw[batch_index]).squeeze().cpu().clone().numpy()
                )
                color_im = np.repeat(color_im[:, :, np.newaxis] * 255, 3, axis=2).astype("uint8")

            depth_pred = depths_b1hw[batch_index].squeeze().cpu().clone().numpy()
            depth_pred = o3d.geometry.Image(depth_pred)
            color_im = o3d.geometry.Image(color_im)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_im,
                depth_pred,
                depth_scale=1.0,
                depth_trunc=self.fusion_max_depth,
                convert_rgb_to_intensity=False,
            )
            cam_intr = K_b44[batch_index].cpu().clone().numpy()
            cam_T_world_44 = cam_T_world_b44[batch_index].cpu().clone().numpy()

            self.volume.integrate(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(
                    width=width,
                    height=height,
                    fx=cam_intr[0, 0],
                    fy=cam_intr[1, 1],
                    cx=cam_intr[0, 2],
                    cy=cam_intr[1, 2],
                ),
                cam_T_world_44,
            )

    def export_mesh(self, path: str) -> None:
        """Save the mesh."""
        o3d.io.write_triangle_mesh(path, self.volume.extract_triangle_mesh())

    def get_mesh(self) -> o3d.geometry.TriangleMesh:
        """Get a mesh from the TSDF volume."""
        mesh = self.volume.extract_triangle_mesh()
        return mesh

    def get_point_cloud(self) -> o3d.geometry.PointCloud:
        """Get a point cloud from the TSDF volume."""
        return self.volume.extract_point_cloud()


class OurFuser(DepthFuser):
    def __init__(
        self,
        gt_path: Optional[str] = None,
        fusion_resolution: float = 0.04,
        max_fusion_depth: float = 3,
        fuse_color: bool = False,
        num_features: int = 2,
    ):
        super().__init__(
            gt_path=gt_path,
            fusion_resolution=fusion_resolution,
            max_fusion_depth=max_fusion_depth,
            fuse_color=fuse_color,
        )
        if fuse_color:
            logger.warning(
                "fusing color using custom fuser is not supported, " "Color will not be fused."
            )

        if gt_path is not None:
            if not Path(gt_path).exists():
                raise FileNotFoundError(f"GT mesh {gt_path} not found.")

            gt_mesh = trimesh.load(gt_path, force="mesh")
            tsdf_pred = TSDF.from_mesh(
                gt_mesh, voxel_size=fusion_resolution, num_features=num_features
            )
        else:
            bounds = {}
            bounds["xmin"] = -10.0
            bounds["xmax"] = 10.0
            bounds["ymin"] = -10.0
            bounds["ymax"] = 10.0
            bounds["zmin"] = -10.0
            bounds["zmax"] = 10.0

            tsdf_pred = TSDF.from_bounds(
                bounds, voxel_size=fusion_resolution, num_features=num_features
            )

        self.tsdf_fuser_pred = TSDFFuser(tsdf_pred, max_depth=max_fusion_depth)

    def fuse_frames(
        self,
        depths_b1hw: torch.Tensor,
        K_b44: torch.Tensor,
        cam_T_world_b44: torch.Tensor,
        color_b3hw: torch.Tensor,
    ):
        self.tsdf_fuser_pred.integrate_depth(
            depth_b1hw=depths_b1hw.half(),
            cam_T_world_T_b44=cam_T_world_b44.half(),
            K_b44=K_b44.half(),
        )

    def fuse_frames_features(self, depths_b1hw, K_b44, cam_T_world_b44, features_bchw):
        self.tsdf_fuser_pred.integrate_depth(
            depth_b1hw=depths_b1hw.half(),
            cam_T_world_T_b44=cam_T_world_b44.half(),
            K_b44=K_b44.half(),
            features_bchw=features_bchw.half(),
        )

    def export_mesh(self, path: str, export_single_mesh: bool = True):
        trimesh.exchange.export.export_mesh(
            self.tsdf_fuser_pred.tsdf.to_mesh(export_single_mesh=export_single_mesh),
            path,
        )

    def get_mesh(self, export_single_mesh: bool = True) -> trimesh.Trimesh:
        return self.tsdf_fuser_pred.tsdf.to_mesh(export_single_mesh=export_single_mesh)

    def get_tsdf(self):
        return self.tsdf_fuser_pred.tsdf


__AVAILABLE_FUSERS__ = {"ours": OurFuser, "open3d": Open3DFuser}


def get_fuser(opts, scan: Optional[str] = None) -> DepthFuser:
    """Returns the depth fuser required"""

    if opts.inference.depth_fuser not in __AVAILABLE_FUSERS__.keys():
        raise ValueError(
            f"Selected TSDF fuser {opts.depth_fuser} not found. Available fusers are {__AVAILABLE_FUSERS__.keys()}"
        )
    gt_path = None
    if opts.data.dataset == "scannet":
        gt_path = ScannetDataset.get_gt_mesh_path(opts.data.dataset_path, opts.split, scan)

    return __AVAILABLE_FUSERS__[opts.inference.depth_fuser](
        gt_path=gt_path,
        fusion_resolution=opts.inference.fusion_resolution,
        max_fusion_depth=opts.inference.fusion_max_depth,
        fuse_color=opts.inference.fuse_color,
    )

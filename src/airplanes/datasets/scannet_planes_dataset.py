import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms

from airplanes.datasets.scannet_dataset import ScannetDataset
from airplanes.utils.io_utils import AssetFileNames
from airplanes.utils.plane_geometry import transform_plane_parameters


class ScannetPlaneDataset(ScannetDataset):
    """This class extends Scannet MVS dataset to keep into account also plane data"""

    def __init__(
        self,
        dataset_path,
        split,
        mv_tuple_file_suffix,
        data_opts: DictConfig,
        include_full_res_depth=False,
        limit_to_scan_id=None,
        num_images_in_tuple=None,
        tuple_info_file_location=None,
        image_height=384,
        image_width=512,
        high_res_image_width=640,
        high_res_image_height=480,
        image_depth_ratio=2,
        shuffle_tuple=False,
        include_full_depth_K=False,
        include_high_res_color=False,
        pass_frame_id=False,
        skip_frames=None,
        skip_to_frame=None,
        verbose_init=True,
        min_valid_depth=0.001,
        max_valid_depth=10,
        max_number_of_planes=15,
        load_src_depth=False,
        rotate_images=False,
    ):
        """
        Params:
            all those from ScannetDataset
            max_number_of_planes: defines how many planes we are using from the gt mask. For instance,
                if set to 15 it means we are going to use only the first 15 largest planes for the image.
                If the number of valid planes is lower than `max_number_of_planes`, all the valid planes are used.
        """
        super().__init__(
            dataset_path=dataset_path,
            split=split,
            mv_tuple_file_suffix=mv_tuple_file_suffix,
            data_opts=data_opts,
            include_full_res_depth=include_full_res_depth,
            limit_to_scan_id=limit_to_scan_id,
            num_images_in_tuple=num_images_in_tuple,
            tuple_info_file_location=tuple_info_file_location,
            image_height=image_height,
            image_width=image_width,
            high_res_image_width=high_res_image_width,
            high_res_image_height=high_res_image_height,
            image_depth_ratio=image_depth_ratio,
            shuffle_tuple=shuffle_tuple,
            include_full_depth_K=include_full_depth_K,
            include_high_res_color=include_high_res_color,
            pass_frame_id=pass_frame_id,
            skip_frames=skip_frames,
            skip_to_frame=skip_to_frame,
            verbose_init=verbose_init,
            min_valid_depth=min_valid_depth,
            max_valid_depth=max_valid_depth,
            load_src_depth=load_src_depth,
            rotate_images=rotate_images,
        )
        self.planes_path = data_opts.planes_path
        self.renders_path = data_opts.renders_path

        self.max_number_of_planes = max_number_of_planes
        self.plane_cache: dict[str, np.ndarray] = {}

    @staticmethod
    def get_gt_planes_path(dataset_path: str, split: str, scan_id: str) -> str:
        """
        Returns a path to a gt plane  file.
        A plane file is a npy file generated by PlaneRCNN that contains all the planes of the scene
        in world coords.

        Params:
            dataset_path: path to the plane data root folder
            split: not used here
            scan_id: scene to load
        Returns:
            a str representing the path to the plane file
        """
        gt_path = os.path.join(
            dataset_path, scan_id, AssetFileNames.get_planes_filename(scene_name=scan_id)
        )
        return gt_path

    @staticmethod
    def get_gt_plane_segmentation_path(dataset_path: str, scan_id: str, frame_id: str) -> str:
        """
        Returns a path to the plane segmentation mask. The segmentation mask contains the planes for
        the current frame, in world coords. It has been obtained by rendering the planes of the scene
        in the current frame.

        Params:
            dataset_path: path to the dataset folder with ground truth plane images.
            scan_id: the scan this file belongs to.
            frame_id: id of the frame to load.
        """
        gt_path = os.path.join(
            dataset_path,
            scan_id,
            "frames",
            AssetFileNames.get_gt_plane_image_filename(frame_id=frame_id),
        )
        return gt_path

    def load_planes(self, scan_id: str) -> np.ndarray:
        """This function loads a planes.npy file created by the ground-truth generation script.
        Since the planes are computed per scene, we can cache them.

        Params:
            scan_id: the scan this file belongs to.
        Returns:
            a copy of the set of planes for the requested scene. If the scene does not exist in the
            cache, first popolate the cache.
        """
        key = f"{self.split}/{scan_id}"
        if self.plane_cache.get(key) is None:
            plane_bundle_path = self.get_gt_planes_path(
                self.planes_path, split=self.split, scan_id=scan_id
            )
            plane_bundle = np.array(np.load(plane_bundle_path))

            # PlaneRCNN stores as n*d and we want n/d
            d = np.linalg.norm(plane_bundle, axis=1)
            plane_bundle[d > 0] /= d[d > 0, None] ** 2

            self.plane_cache[key] = plane_bundle

        return np.copy(self.plane_cache[key])

    def load_plane_image(self, scan_id: str, frame_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load plane image, obtained by rendering the mesh with plane ids at the current
        camera position.
        The alpha channel represents the validity of each pixel.

        Params:
            scan_id: the scan this file belongs to.
            frame_id: id of the frame to load.
        Returns:
            plane ids (H,W,), validity mask (H,W)
        """

        plane_segmentation_path = self.get_gt_plane_segmentation_path(
            dataset_path=self.renders_path, scan_id=scan_id, frame_id=frame_id
        )
        if not Path(plane_segmentation_path).exists():
            raise ValueError(
                f"Cannot find ground truth planes ids image: {plane_segmentation_path}"
            )

        plane_segmentation = Image.open(plane_segmentation_path).resize(
            (self.depth_width, self.depth_height), resample=Image.NEAREST
        )
        plane_segmentation = np.array(plane_segmentation).astype(float)
        validity_mask = plane_segmentation[:, :, 3] / 255.0
        return plane_segmentation[:, :, :3], validity_mask

    @staticmethod
    def decode_plane_image(plane_image: np.ndarray) -> np.ndarray:
        """Extract planes IDs from a plane image, obtained by rendering the plane mesh.
        For more details about the conversions, please look at:
            https://github.com/NVlabs/planercnn/blob/2698414a44eaa164f5174f7fe3c87dfc4d5dea3b/datasets/scannet_scene.py#L156

        Params:
            plane_image: (H,W,3) image with encoded planes ids
        Returns:
            (H,W) array with plane ids
        """
        return (
            plane_image[:, :, 0] * 256 * 256 + plane_image[:, :, 1] * 256 + plane_image[:, :, 2]
        ) // 100 - 1

    @staticmethod
    def map_plane_image_to_plane_parameters(
        plane_image: np.ndarray, planes_parameters: np.ndarray
    ) -> np.ndarray:
        """Given an image that encodes planes for the current frame and an array that contains all
        the planes in the scene, this function returns per-pixel plane parameters by solving the
        mapping between instances and plane values.

        Params:
            frame_plane_ids: (H,W) array that contains the index of the each plane
            planes_parameters: (N,3) array with all planes in a given scene, in world reference system
        Returns:
            (H,W,3) array with per-pixel plane parameters
        """
        planes_id_hw = ScannetPlaneDataset.decode_plane_image(plane_image=plane_image)
        h, w = planes_id_hw.shape
        planes_params = np.zeros((h, w, 3)).astype(float)
        mask = (planes_id_hw >= 0) * (planes_id_hw < len(planes_parameters))

        planes_params[mask] = planes_parameters[planes_id_hw.astype("int32")[mask]]
        return planes_params

    def filter_planes(
        self,
        plane_image: np.ndarray,
        planes_parameters_world: np.ndarray,
    ) -> np.ndarray:
        """This function creates the final set of planes ids given the original plane image and the
        plane parameters.
        1. We sort valid planes by number of pixels, and we remove planes with a coverage smaller
        than 1% of the image.
        2. From this set we keep the first `max_number_of_planes` planes.
        If the number of valid planes is smaller, we keep all the valid ones.
        3. Finally, we re-assign plane ids by descending order of plane size (from larger to smaller).

        Params:
            plane_image: (H,W,3) image with encoded planes ids
            planes_parameters: (H,W,3) array with plane parameters in world coords.
        Returns:
            (H,W) array with remapped plane ids, where ids=0 means invalid,
            ids=1 means the largest plane, 1 the second larger and so on.
        """
        planes_id_hw = ScannetPlaneDataset.decode_plane_image(plane_image=plane_image)
        h, w = planes_id_hw.shape

        # set -1 to pixels that belong to invalid planes
        validity_mask_hw = np.linalg.norm(planes_parameters_world, axis=-1) > 1e-3
        planes_id_hw[~validity_mask_hw] = -1

        # find unique ids of planes
        unique_ids, counts = np.unique(planes_id_hw, return_counts=True)
        final_planes_id_hw = np.zeros((h, w))

        # define a threshold as 1% of the pixels in the image
        threshold = (h * w) // 100

        # sort planes by descending order
        combined = [(x, y) for x, y in zip(unique_ids.tolist(), counts.tolist())]
        planes_descending_ord = sorted(combined, key=lambda x: x[1], reverse=True)

        num_valid_planes_found = 0
        for idx, num_pix in planes_descending_ord:
            if num_pix < threshold or idx == -1:
                continue

            # we have found a valid plane, update the final mask with the new id
            curr_plane_mask_hw = planes_id_hw == idx
            final_planes_id_hw[curr_plane_mask_hw] = num_valid_planes_found + 1

            num_valid_planes_found += 1

            if num_valid_planes_found >= self.max_number_of_planes:
                # we have found enough planes, so we can stop iterating
                break

        return final_planes_id_hw

    def load_plane_data(self, scan_id, frame_id, cam_T_world_b44, flip=False):
        """Function that loads plane data for the current frame.
        Returns:
        A dict with the following keys:
            planes_id: tensor 1hw with plane ids from 0 to `self.max_number_of_planes`.
                Here 0 means non-planar/not in the set of largest planes, 1 is the largest plane,
                2 the second largest and so on.
            planes_id_masked: as planes_id, but we apply also the validity mask to remove planes
                connecting two faces.
            non_planar_mask: boolean mask 1hw that is True for non-planar pixels.
        """
        plane_data = {}

        # load plane image: this image encodes planes ids for the current frame.
        plane_image_hw, validity_mask_hw = self.load_plane_image(scan_id=scan_id, frame_id=frame_id)

        # get planes parameters for the current scene in world ref
        scene_planes_N3 = self.load_planes(scan_id=scan_id)

        h, w = self.depth_height, self.depth_width

        # map scene planes parameters to the current frame.
        # Each pixel contains 3 scalar representing the plane parameters in the world ref. system.
        try:
            planes_parameters_world_hw3 = self.map_plane_image_to_plane_parameters(
                plane_image=plane_image_hw, planes_parameters=scene_planes_N3
            )
        except IndexError as e:
            logger.warning(
                "Your plane ids are bigger than the number of planes saved - this should not happen"
                f" and suggests a bug in dataset generation! Error for {scan_id} - {frame_id}. "
                f" Original error: {e}"
            )
            planes_parameters_world_hw3 = np.zeros((h, w, 3))

        # now map planes parameters into camera coordinates
        planes_parameters_world_n31 = (
            torch.tensor(planes_parameters_world_hw3).reshape(-1, 3, 1).float()
        )
        cam_T_world_b44 = torch.tensor(cam_T_world_b44).unsqueeze(0)
        planes_parameters_camera_n31 = transform_plane_parameters(
            plane_params_bN1=planes_parameters_world_n31, tranformation_b44=cam_T_world_b44
        )
        planes_parameters_camera_3hw = (
            planes_parameters_camera_n31.squeeze().reshape(h, w, 3).permute(2, 0, 1).float()
        )

        # orientate all normals to face the camera
        flip_mask_hw = planes_parameters_camera_3hw[2] < 0
        planes_parameters_camera_3hw[:, flip_mask_hw] *= -1

        # filter planes: we remove small planes and we remap planes ids
        # between 0 and max_number_of_planes.
        planes_id_hw = self.filter_planes(
            plane_image=plane_image_hw,
            planes_parameters_world=planes_parameters_world_hw3,
        )

        # move everything to torch tensors
        planes_id_1hw = torch.tensor(planes_id_hw).long().unsqueeze(0)
        validity_mask_1hw = torch.tensor(validity_mask_hw).unsqueeze(0)

        # in case of flip, flip both the mask and the planes
        if flip:
            planes_id_1hw = torch.flip(planes_id_1hw, (-1,))
            validity_mask_1hw = torch.flip(validity_mask_1hw, (-1,))

        invalid_mask = validity_mask_1hw < 0.01

        plane_data["planes_id"] = planes_id_1hw
        plane_data["planes_id_masked"] = planes_id_1hw * (1 - invalid_mask.long())
        plane_data["non_planar_mask"] = (planes_id_1hw == 0).bool()

        return plane_data

    def __getitem__(self, idx: int):
        """Extend Scannet loaders to include also plane info for the target frame
        Params:
            idx: idx of the sample
        Returns:
            cur_data, src_data: two dictionaries with info for the current frame and all the src ones.
            Specifically, we update `cur_data` to save:
                - planes_id: tensor 1hw with plane ids from 0 to `self.max_number_of_planes`.
                - planes_id_masked: as planes_id, but we apply also the validity mask to remove planes
                    connecting two faces.
                - non_planar_mask: boolean mask 1hw that is True for non-planar pixels.
        """
        cur_data, src_data = super().__getitem__(idx)

        scan_id, *frame_ids = self.frame_tuples[idx].split(" ")
        cur_frame_id = frame_ids[0]

        cur_plane_data = self.load_plane_data(
            scan_id,
            cur_frame_id,
            cam_T_world_b44=cur_data["cam_T_world_b44"],
            flip=cur_data["flip"],
        )
        cur_data.update(cur_plane_data)

        return cur_data, src_data
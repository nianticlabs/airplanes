import os
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

import airplanes.modules.cost_volume as cost_volume
from airplanes.datasets.scannet_dataset import ScannetDataset
from airplanes.experiment_modules.depth_planes_embeddings_model import (
    DepthPlanesEmbeddingsModel,
)
from airplanes.meshing.tsdf_fusion import Open3DFuser, OurFuser
from airplanes.utils.dataset_utils import get_dataset
from airplanes.utils.generic_utils import cache_model_outputs, to_gpu
from airplanes.utils.io_utils import AssetFileNames
from airplanes.utils.visualization_utils import (
    quick_viz_export,
    reverse_imagenet_normalize,
)


@torch.no_grad()
@hydra.main(
    version_base=None, config_path=os.getcwd() + "/configs", config_name="test_2D_network.yaml"
)
def hydra_main_wrapper(opts: DictConfig):
    main(opts)


def main(opts: DictConfig):
    # if no GPUs are available for us then, use the 32 bit on CPU
    if opts.gpus == 0:
        print("Setting precision to 32 bits since --gpus is set to 0.")
        opts.precision = 32

    # shortcuts to configs
    inference_opts = opts.inference
    data_opts = opts.data
    model_opts = opts.models

    # get dataset
    dataset_class, scans = get_dataset(
        dataset_mode="depth",
        dataset_name=data_opts.dataset,
        split_filepath=data_opts.dataset_scan_split_file,
        single_debug_scan_id=opts.single_debug_scan_id,
    )

    # path where results for this model, dataset, and tuple type are.
    results_path = os.path.join(opts.output_base_path, model_opts.name, data_opts.dataset)

    viz_output_dir = None
    mesh_output_dir = None
    depth_output_dir = None
    fuser = None
    # set up directories for fusion

    if opts.run_fusion:
        if inference_opts.fuse_color:
            mesh_output_folder_name = mesh_output_folder_name + "_color"

        mesh_output_dir = os.path.join(results_path, "meshes")

        Path(mesh_output_dir).mkdir(parents=True, exist_ok=True)

        if opts.save_intermediate_meshes:
            assert opts.batch_size == 1, "Batch size must be 1 to save intermediate meshes."
            print(f"".center(80, "#"))
            print(f" Saving intermediate meshes.".center(80, "#"))
            print(f"Output directory:\n{mesh_output_dir + '/intermediate_meshes'} ".center(80, "#"))
            print(f"".center(80, "#"))
            print("")

    # set up directories for caching depths
    if opts.cache_depths:
        # path where we cache depth maps
        depth_output_dir = os.path.join(results_path, "depths")
        print(depth_output_dir)

        Path(depth_output_dir).mkdir(parents=True, exist_ok=True)
        print(f"".center(80, "#"))
        print(f" Caching depths.".center(80, "#"))
        print(f"Output directory:\n{depth_output_dir} ".center(80, "#"))
        print(f"".center(80, "#"))
        print("")

    # set up directories for quick depth visualizations
    if opts.dump_depth_visualization:
        viz_output_folder_name = "quick_viz"
        viz_output_dir = os.path.join(results_path, "viz", viz_output_folder_name)

        Path(viz_output_dir).mkdir(parents=True, exist_ok=True)
        print(f"".center(80, "#"))
        print(f" Saving quick viz.".center(80, "#"))
        print(f"Output directory:\n{viz_output_dir} ".center(80, "#"))
        print(f"".center(80, "#"))
        print("")

    model = DepthPlanesEmbeddingsModel.load_from_checkpoint(
        opts.load_weights_from_checkpoint, opts=opts, args=None, strict=False
    )

    if opts.fast_cost_volume and isinstance(model.cost_volume, cost_volume.FeatureVolumeManager):
        model.cost_volume = model.cost_volume.to_fast()

    model = model.cuda().eval()

    with torch.inference_mode():
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        print("###### Data path")
        print(data_opts.dataset_path)

        # loop over scans
        for scan in tqdm(scans, unit="scan", desc="All scans"):
            # set up dataset with current scan
            dataset = dataset_class(
                dataset_path=data_opts.dataset_path,
                split=data_opts.split,
                mv_tuple_file_suffix=data_opts.mv_tuple_file_suffix,
                data_opts=data_opts,
                limit_to_scan_id=scan,
                include_full_res_depth=True,
                tuple_info_file_location=data_opts.tuple_info_file_location,
                num_images_in_tuple=None,
                shuffle_tuple=opts.shuffle_tuple,
                include_high_res_color=opts.dump_depth_visualization,
                include_full_depth_K=True,
                skip_frames=opts.skip_frames,
                skip_to_frame=opts.skip_to_frame,
                image_width=opts.image_width,
                image_height=opts.image_height,
                pass_frame_id=True,
                rotate_images=opts.rotate_images,
            )

            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=opts.batch_size,
                shuffle=False,
                num_workers=opts.num_workers,
                drop_last=False,
            )

            # initialize fuser if we need to fuse
            if opts.run_fusion:
                gt_path = None
                if data_opts.dataset == "scannet":
                    gt_path = ScannetDataset.get_gt_mesh_path(
                        dataset_path=data_opts.dataset_path, split=opts.split, scan_id=scan
                    )

                if inference_opts.fuse_color:
                    fuser = Open3DFuser(
                        gt_path=gt_path,
                        fusion_resolution=inference_opts.fusion_resolution,
                        max_fusion_depth=inference_opts.fusion_max_depth,
                        fuse_color=True,
                    )
                else:
                    fuser = OurFuser(
                        gt_path=gt_path,
                        fusion_resolution=inference_opts.fusion_resolution,
                        max_fusion_depth=inference_opts.fusion_max_depth,
                        num_features=inference_opts.num_features,
                        fuse_color=False,
                    )

            initial_t = None
            # normal_generator = None
            for batch_ind, batch in enumerate(tqdm(dataloader, unit="batch", desc="Current scan")):
                # get data, move to GPU
                cur_data, src_data = batch
                cur_data = to_gpu(cur_data, key_ignores=["frame_id_string"])
                src_data = to_gpu(src_data, key_ignores=["frame_id_string"])

                if opts.shift_world_origin:
                    if initial_t is None:
                        initial_t = cur_data["world_T_cam_b44"][0, :3, 3].clone()

                    cur_data["world_T_cam_b44"][:, :3, 3] = (
                        cur_data["world_T_cam_b44"][:, :3, 3] - initial_t
                    )
                    cur_data["cam_T_world_b44"] = torch.linalg.inv(cur_data["world_T_cam_b44"])

                    for src_ind in range(src_data["world_T_cam_b44"].shape[1]):
                        src_data["world_T_cam_b44"][:, src_ind, :3, 3] = (
                            src_data["world_T_cam_b44"][:, src_ind, :3, 3] - initial_t
                        )
                        src_data["cam_T_world_b44"][:, src_ind] = torch.linalg.inv(
                            src_data["world_T_cam_b44"][:, src_ind]
                        )

                depth_gt = cur_data["full_res_depth_b1hw"]

                # run to get output, also measure time
                start_time.record()
                # use unbatched (looping) matching encoder image forward passes
                # for numerically stable testing. If opts.fast_cost_volume, then
                # batch.
                outputs = model(
                    "test",
                    cur_data,
                    src_data,
                    unbatched_matching_encoder_forward=(not opts.fast_cost_volume),
                    return_mask=True,
                )
                end_time.record()
                torch.cuda.synchronize()

                elapsed_model_time = start_time.elapsed_time(end_time)

                upsampled_depth_pred_b1hw = F.interpolate(
                    outputs["depth_pred_s0_b1hw"],
                    size=(depth_gt.shape[-2], depth_gt.shape[-1]),
                    mode="nearest",
                )

                # inf max depth matches DVMVS metrics, using minimum of 0.5m
                valid_mask_b = cur_data["full_res_depth_b1hw"] > 0.5

                ######################### DEPTH FUSION #########################
                if opts.run_fusion:
                    # mask predicted depths when no vaiid MVS information
                    # exists, off by default
                    assert fuser is not None

                    feature_list = [outputs["plane_mask_pred_s0_b1hw"]]
                    features_bchw = torch.cat(feature_list, 1)

                    upsampled_features_bchw = F.interpolate(
                        features_bchw,
                        size=(depth_gt.shape[-2], depth_gt.shape[-1]),
                        mode="nearest",
                    )

                    if inference_opts.fuse_color:
                        color_b3hw = cur_data["image_b3hw"].cpu().clone()
                        color_b3hw = reverse_imagenet_normalize(color_b3hw)
                        fuser.fuse_frames(
                            depths_b1hw=upsampled_depth_pred_b1hw,
                            K_b44=cur_data["K_full_depth_b44"],
                            cam_T_world_b44=cur_data["cam_T_world_b44"],
                            color_b3hw=color_b3hw,
                        )
                    else:
                        fuser.fuse_frames_features(
                            depths_b1hw=upsampled_depth_pred_b1hw,
                            K_b44=cur_data["K_full_depth_b44"],
                            cam_T_world_b44=cur_data["cam_T_world_b44"],
                            features_bchw=upsampled_features_bchw,
                        )

                    if opts.save_intermediate_meshes:
                        # export intermediate mesh
                        intermediate_mesh_path = (
                            Path(mesh_output_dir)
                            / scan
                            / AssetFileNames.get_intermediate_pred_mesh_filename(
                                scan, cur_data["frame_id_string"][0]
                            )
                        )
                        intermediate_mesh_path.parent.mkdir(parents=True, exist_ok=True)
                        fuser.export_mesh(str(intermediate_mesh_path))

                ########################### Quick Viz ##########################
                if opts.dump_depth_visualization:
                    # make a dir for this scan
                    assert viz_output_dir is not None
                    output_path = os.path.join(viz_output_dir, scan)
                    Path(output_path).mkdir(parents=True, exist_ok=True)

                    quick_viz_export(
                        output_path,
                        outputs,
                        cur_data,
                        batch_ind,
                        valid_mask_b,
                        opts.batch_size,
                    )
                ########################## Cache Depths ########################
                if opts.cache_depths:
                    assert depth_output_dir is not None
                    output_path = os.path.join(depth_output_dir, scan)
                    Path(output_path).mkdir(parents=True, exist_ok=True)

                    keys_to_save = ["depth_pred_s0_b1hw", "plane_mask_pred_s0_b1hw"]

                    if "embedding_pred_s0_b3hw" in outputs:
                        keys_to_save.append("embedding_pred_s0_b3hw")
                    else:
                        logger.warning("Embedding prediction not found, not saving embeddings.")

                    cache_model_outputs(
                        output_path,
                        outputs,
                        cur_data,
                        src_data,
                        batch_ind,
                        opts.batch_size,
                        keys_to_save=keys_to_save,
                    )

            if opts.run_fusion:
                assert fuser is not None
                assert mesh_output_dir is not None
                scene = scan.replace("/", "_")

                if inference_opts.fuse_color:
                    path = os.path.join(
                        os.path.join(mesh_output_dir, scene),
                        AssetFileNames.get_pred_mesh_filename(scene),
                    )
                    fuser.export_mesh(path)
                else:
                    tsdf = fuser.get_tsdf()
                    tsdf.save(
                        os.path.join(mesh_output_dir, scene),
                        AssetFileNames.get_tsdf_filename(scene),
                    )
                    # NOTE: enable this to save non-planar mesh
                    fuser.export_mesh(
                        os.path.join(
                            os.path.join(mesh_output_dir, scene),
                            AssetFileNames.get_pred_mesh_filename(scene),
                        ),
                    )

    print("Processing with 2D network finished")


if __name__ == "__main__":
    hydra_main_wrapper()  # type: ignore

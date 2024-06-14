import os

from airplanes.datasets.scannet_dataset import ScannetDataset

# do this before importing numpy! (doing it right up here in case numpy is dependency of e.g. json)
os.environ["MKL_NUM_THREADS"] = "4"  # noqa: E402
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # noqa: E402
os.environ["OMP_NUM_THREADS"] = "4"  # noqa: E402
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # noqa: E402
from pathlib import Path

import click
import numpy as np
import torch
import tqdm
import trimesh
from loguru import logger
from omegaconf import DictConfig
from PIL import Image
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings, TexturesVertex
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection
from torch.utils.data import DataLoader

from airplanes.utils.generic_utils import read_scannetv2_filename
from airplanes.utils.rendering_utils import interpolate_face_attributes_nearest


def run_frame(
    mesh: Meshes,
    cam_T_world_b44: torch.Tensor,
    K_b44: torch.Tensor,
    image_size: torch.Tensor,
    raster_settings: RasterizationSettings,
    frame_name: str,
    save_path: Path,
    render_depth: bool,
) -> None:
    """Render a frame.
    Params:
        mesh: mesh to render, as pytorch3d.structures.Meshes
        cam_T_world_b44: camera pose from world to camera, as (b,4,4) tensor
        image_size: size of the image to render, as a tensor with two elements (H,W)
        raster_setting: setting for the rasterizer
        frame_name: final name of the frame
        save_path: path to the folder in which we are going to save the frame
    """

    R = cam_T_world_b44[:, :3, :3]
    T = cam_T_world_b44[:, :3, 3]
    K = K_b44[:, :3, :3]
    cams = cameras_from_opencv_projection(R=R, tvec=T, camera_matrix=K, image_size=image_size)

    mesh = mesh.cuda()
    cams = cams.cuda()

    rasterizer = MeshRasterizer(
        cameras=cams,
        raster_settings=raster_settings,
    )

    _mesh = mesh.extend(len(cams))
    fragments = rasterizer(_mesh)

    # nearest sampling
    faces_packed = _mesh.faces_packed()
    verts_features_packed = _mesh.textures.verts_features_packed()
    faces_verts_features = verts_features_packed[faces_packed]
    texture_bhw14 = interpolate_face_attributes_nearest(
        fragments.pix_to_face, fragments.bary_coords, faces_verts_features
    )

    # bilinear sampling
    bilinear_texture_bhw14 = _mesh.textures.sample_textures(fragments, _mesh.faces_packed())
    rendered_depth_bhw = fragments.zbuf[..., 0]

    # we want nearest for RGB and bilinear for alpha - so combine
    texture_bhw14[..., 3] = bilinear_texture_bhw14[..., 3]
    plane_ids = texture_bhw14.cpu().numpy()[0, ..., 0, :] * 255
    rendered_depth = rendered_depth_bhw.cpu().numpy().squeeze()

    # save image with plane ids
    plane_ids = plane_ids.astype("uint8")
    plane_ids = Image.fromarray(plane_ids)
    plane_ids.save(save_path / f"{frame_name}_planes.png")

    # save rendered depth map
    if render_depth:
        np.save(save_path / f"{frame_name}_depth.npy", rendered_depth)


def run(
    data_dir: Path,
    planes_dir: Path,
    split: str,
    mv_tuple_file_suffix: str,
    tuple_info_file_locations: Path,
    output_dir: Path,
    filename_file: Path,
    height: int,
    width: int,
    render_depth: bool,
) -> None:
    """Run the rendering over all the scenes defined in filename file.
    For each camera position provided by the dataloader for the given the scene we generate an image
    where each pixel contains the plane ID and the depth map.
    """
    all_scans = read_scannetv2_filename(filepath=filename_file)
    if len(all_scans) == 0:
        raise ValueError("Cannot find any scans")

    for scan in tqdm.tqdm(all_scans):
        logger.info(f"starting scan {scan}!")

        save_path = output_dir / scan / "frames"
        save_path.mkdir(exist_ok=True, parents=True)

        mesh = None

        opts = DictConfig(
            {
                "data": {
                    "dataset_path": str(data_dir),
                    "tuple_info_file_location": str(tuple_info_file_locations),
                    "dataset": "scannet",
                    "split": split,
                    "mv_tuple_file_suffix": mv_tuple_file_suffix,
                }
            }
        )

        dataset = ScannetDataset(  # type: ignore
            dataset_path=data_dir,
            split=split,
            mv_tuple_file_suffix=opts.data.mv_tuple_file_suffix,
            include_full_res_depth=False,
            num_images_in_tuple=2,
            tuple_info_file_location=opts.data.tuple_info_file_location,
            limit_to_scan_id=scan,
            data_opts=opts.data,
        )

        # NOTE: we want to disable the flipping. We set the split as val
        dataset.split = "val"
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

        try:
            mesh_trimesh = trimesh.exchange.load.load(planes_dir / scan / "mesh_with_planes.ply")
        except ValueError:
            # For some reason some meshes fail to load with certain numpy/trimesh versions - after digging it looks like some vertices have missing values when loaded
            # so everything breaks.
            # I've manually saved new meshes by reading the ply into trimesh, making sure thigns are fine and saving it again - saved as planes_mesh2.ply
            logger.warning(
                f"Could not load mesh! Trying to load a manually saved version at path {planes_dir / scan / 'annotation' / 'mesh_with_planes2.ply'}"
            )
            mesh_trimesh = trimesh.exchange.load.load(planes_dir / scan / "mesh_with_planes2.ply")

        mesh = Meshes(
            verts=[torch.tensor(mesh_trimesh.vertices).float()],
            faces=[torch.tensor(mesh_trimesh.faces).float()],
            textures=TexturesVertex(
                torch.tensor(mesh_trimesh.visual.vertex_colors).unsqueeze(0).float() / 255.0
            ),
        )
        raster_settings = RasterizationSettings(
            image_size=(height, width),
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        image_size = torch.tensor((height, width)).unsqueeze(0)
        assert len(dataset) > 0, ValueError("Dataset is empty")

        # run the rendered!
        with torch.no_grad():
            for idx, data in tqdm.tqdm(enumerate(dataloader)):
                curr_data, _ = data
                frame_ids = dataset.frame_tuples[idx].split(" ")

                frame_name = frame_ids[1]
                cam_T_world_b44 = curr_data["cam_T_world_b44"]
                K_b44 = curr_data["K_s0_b44"]

                run_frame(
                    mesh=mesh,
                    cam_T_world_b44=cam_T_world_b44,
                    K_b44=K_b44,
                    image_size=image_size,
                    raster_settings=raster_settings,
                    save_path=save_path,
                    frame_name=frame_name,
                    render_depth=render_depth,
                )

        # cleaning up
        del dataloader
        del dataset


@click.command()
@click.option(
    "--data-dir",
    type=Path,
    help="Path to ScanNetv2",
)
@click.option(
    "--planes-dir",
    type=Path,
    help="Path to root folder generated by PlaneRCNN",
)
@click.option("--split", type=click.Choice(["train", "val", "test"]), default="train")
@click.option(
    "--tuple-info-file-locations",
    type=Path,
    help="Path to the folder with tuple files",
    default=Path("src/airplanes/data_splits/ScanNetv2/standard_split/"),
)
@click.option(
    "--mv-tuple-file-suffix",
    type=str,
    help="Suffix of the tuple file with keyframes",
    default="_eight_view_deepvmvs.txt",
)
@click.option("--output-dir", type=Path, default="data/rendered_planes")
@click.option(
    "--filename-file",
    type=Path,
    default=Path("data_splits/ScanNetv2/standard_split/scannetv2_train.txt"),
    help="Path to filename file to use. It contains the scenes to process",
)
@click.option("--height", type=int, help="height of the image to render", default=192)
@click.option("--width", type=int, help="width of the image to render", default=256)
@click.option("--render-depth", is_flag=True, help="Render depth maps", default=False)
def cli(
    data_dir: Path,
    planes_dir: Path,
    split: str,
    tuple_info_file_locations: Path,
    mv_tuple_file_suffix: str,
    output_dir: Path,
    filename_file: Path,
    height: int,
    width: int,
    render_depth: bool,
):
    torch.manual_seed(10)
    np.random.seed(10)

    run(
        data_dir=data_dir,
        planes_dir=planes_dir,
        split=split,
        tuple_info_file_locations=tuple_info_file_locations,
        mv_tuple_file_suffix=mv_tuple_file_suffix,
        output_dir=output_dir,
        filename_file=filename_file,
        height=height,
        width=width,
        render_depth=render_depth,
    )


if __name__ == "__main__":
    cli()  # type: ignore

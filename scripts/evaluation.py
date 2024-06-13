"""
Run evaluation of AirPlanes

Usage:

1. Generate Depth maps, 2D embeddings and TSDFs
>>> python -m scripts.evaluation model-inference [--checkpoint --output-dir]

2. Training of per-scene MLPs
>>> python -m scripts.evaluation train-embeddings [--pred-root]

3. Run RANSAC to extract planar meshes
>>> python -m scripts.evaluation run-ransac-ours [--pred-root --dest-planar-meshes]

4. Evaluate results on the meshing benchmark
>>> python -m scripts.evaluation meshing-benchmark [--pred-root --gt-root]

5. Evaluate results on the segmentation benchmark
>>> python -m scripts.evaluation segmentation-benchmark [--pred-root --gt-root]

6. Evaluate results on the planar benchmark (part of the meshing benchmark)
>>> python -m scripts.evaluation planar-benchmark [--pred-root --gt-root]
"""
import subprocess
from pathlib import Path

import click
from loguru import logger


@click.group()
def run():
    pass


@run.command()
@click.option(
    "--checkpoint",
    help="Path to the model checkpoint to use",
    type=Path,
    default=Path("checkpoints/airplanes_model.ckpt"),
)
@click.option(
    "--output-dir",
    help="Path to the output directory where meshes and 2D predictions will be saved",
    type=Path,
    default=Path("results"),
)
def model_inference(checkpoint: str, output_dir: Path):
    logger.info("Generating TSDFs and 2D embeddings using our model for each scan")
    subprocess.run(
        [
            "python",
            "-m",
            "airplanes.test_2D_network",
            f"load_weights_from_checkpoint={str(checkpoint)}",
            f"output_base_path={output_dir}",
            "cache_depths=True",  # we want to save 2D embeddings and plane probabilities
        ]
    )


@run.command()
@click.option(
    "--pred-root",
    help="Path to the directory where non-planar meshes were saved",
    type=str,
    default=Path("results/depth_planes_embeddings/scannet/meshes"),
)
def train_embeddings(pred_root: Path):
    logger.info("Training 3D Embeddings (MLPs)")
    subprocess.run(
        [
            "python",
            "-m",
            "airplanes.scene_embeddings_optimisation",
            "--pred-root",
            f"{str(pred_root)}",
            "--embedding-dim",
            "3",
            "--max-depth",
            "3.0",
            "--normal-pull-threshold",  ## t_n in the paper
            "0.8",
            "--embedding-pull-threshold",  ## t_e in the paper
            "0.9",
            "--push-threshold",  ## t_p in the paper
            "1.0",
            "--pixels-per-iteration",
            "400",
            "--frames-per-iteration",
            "50",
            "--num-iterations",
            "10",
        ]
    )


@run.command()
@click.option(
    "--pred-root",
    help="Path to the directory where non-planar meshes were saved",
    type=str,
    default=Path("results/depth_planes_embeddings/scannet/meshes"),
)
@click.option(
    "--dest-planar-meshes",
    help="Path to the output directory where the planar meshes produced by our model will be saved",
    type=str,
    default=Path("results/depth_planes_embeddings/scannet/meshes/planar_meshes_ours"),
)
def run_ransac_ours(pred_root: Path, dest_planar_meshes: Path):
    logger.warning("Running RANSAC using Embeddings")
    subprocess.run(
        [
            "python",
            "-m",
            "airplanes.clustering.sequential_ransac",
            "--pred_root",
            f"{str(pred_root)}",
            "--output-dir",
            f"{dest_planar_meshes}",
            "--file-type",
            "tsdf",
            "--embeddings-usage",
            "ransac",
            "--embeddings-scale-factor",
            "1.0",
            "--embedding-dim",
            "3",
            "--normal-inlier-threshold",
            "0.0",
            "--embeddings-inlier-threshold",  ## r_e in the paper
            "0.5",
            "--voxel-planar-threshold",
            "0.25",
            "--force-assign-points",
            "--merge-planes-with-similar-embeddings",
        ]
    )


@run.command()
@click.option(
    "--pred-root",
    help="Path to the directory where non-planar meshes were saved",
    type=str,
    default=Path("results/depth_planes_embeddings/scannet/meshes"),
)
@click.option(
    "--dest-planar-meshes",
    help="Path to the output directory where the planar meshes produced by the baseline will be saved",
    type=str,
    default=Path("results/depth_planes_embeddings/scannet/meshes/planar_meshes_baseline"),
)
def run_ransac_baseline(pred_root: Path, dest_planar_meshes: Path):
    logger.warning("Running RANSAC on top of SimpleRecon")
    subprocess.run(
        [
            "python",
            "-m",
            "airplanes.clustering.sequential_ransac",
            "--pred_root",
            f"{str(pred_root)}",
            "--output-dir",
            f"{dest_planar_meshes}",
            "--file-type",
            "mesh",
            "--embeddings-usage",
            "none",
            "--embeddings-inlier-threshold",
            "0.0",
            "--normal-inlier-threshold",
            "0.8",
            "--voxel-planar-threshold",
            "0.0",
        ]
    )


@run.command()
@click.option(
    "--pred-root",
    help="Path to the directory where the estimated planar meshes were saved",
    type=str,
    default=Path("results/depth_planes_embeddings/scannet/meshes/planar_meshes_ours"),
)
@click.option(
    "--gt-root",
    help="Path to the ground truth planar meshes",
    type=str,
    default=Path("/mnt/scannet-planes-meshes"),
)
@click.option(
    "--output-score-dir",
    help="Path to the output directory where scores will be saved",
    type=str,
    default=Path("results/depth_planes_embeddings/scannet/scores"),
)
def meshing_benchmark(pred_root: Path, gt_root: Path, output_score_dir: Path):
    logger.warning("Running meshing benchmark")
    subprocess.run(
        [
            "python",
            "-m",
            "airplanes.benchmarks.meshing",
            "--pred-root",
            f"{str(pred_root)}",
            "--gt-root",
            f"{str(gt_root)}",
            "--output-score-dir",
            f"{str(output_score_dir)}",
        ]
    )


@run.command()
@click.option(
    "--pred-root",
    help="Path to the directory where the estimated planar meshes were saved",
    type=str,
    default=Path("results/depth_planes_embeddings/scannet/meshes/planar_meshes_ours"),
)
@click.option(
    "--gt-root",
    help="Path to the ground truth planar meshes",
    type=str,
    default=Path("/mnt/scannet-planes-meshes"),
)
@click.option(
    "--output-score-dir",
    help="Path to the output directory where scores will be saved",
    type=str,
    default=Path("results/depth_planes_embeddings/scannet/scores"),
)
def segmentation_benchmark(pred_root: Path, gt_root: Path, output_score_dir: Path):
    logger.warning("Running segmentation benchmark")
    subprocess.run(
        [
            "python",
            "-m",
            "airplanes.benchmarks.segmentation",
            "run-instance-benchmark",
            "--pred_root",
            f"{str(pred_root)}",
            "--gt_root",
            f"{str(gt_root)}",
            "--output-score-dir",
            f"{str(output_score_dir)}",
        ]
    )


@run.command()
@click.option(
    "--pred-root",
    help="Path to the directory where the estimated planar meshes were saved",
    type=str,
    default=Path("results/depth_planes_embeddings/scannet/meshes/planar_meshes_ours"),
)
@click.option(
    "--gt-root",
    help="Path to the ground truth planar meshes",
    type=str,
    default=Path("/mnt/scannet-planes-meshes"),
)
@click.option(
    "--output-score-dir",
    help="Path to the output directory where scores will be saved",
    type=str,
    default=Path("results/depth_planes_embeddings/scannet/scores"),
)
def planar_benchmark(pred_root: Path, gt_root: Path, output_score_dir: Path):
    logger.warning("Running planar benchmark")
    subprocess.run(
        [
            "python",
            "-m",
            "airplanes.benchmarks.meshing",
            "--pred-root",
            f"{str(pred_root)}",
            "--gt-root",
            f"{str(gt_root)}",
            "--output-score-dir",
            f"{str(output_score_dir)}",
            "--use-planar-metrics",
            "--k",
            "20",
        ]
    )


if __name__ == "__main__":
    run()  # type:ignore

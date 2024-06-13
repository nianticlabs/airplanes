"""
Script to process arbitrary captures.

Usage:

1. Prepare test tuples for arbitrary captures.
>>> python -m scripts.inference_on_captures prepare-captures [--captures /path/to/captures]

2. Run inference on arbitrary captures with the 2D network.
>>> python -m scripts.inference_on_captures model-inference [--checkpoint --output-dir]

4. Training of per-scene MLPs
>>> python -m scripts.inference_on_captures train-embeddings [--pred-root --captures]

3. Run RANSAC to extract planar meshes using our method
>>> python -m scripts.inference_on_captures run-ransac-ours [--pred-root --dest-planar-meshes --captures]

4. Run RANSAC to extract planar meshes using the baseline method
>>> python -m scripts.inference_on_captures run-ransac-baseline [--pred-root --dest-planar-meshes --captures]
"""

import subprocess
import click
from pathlib import Path
from loguru import logger


@click.group()
def run():
    pass


@run.command()
@click.option(
    "--captures", help="Path to captures", default=Path("arbitrary_captures/vdr"), type=Path
)
def prepare_captures(captures: Path):
    logger.info("Generating test tuples for arbitrary captures")
    subprocess.run(
        [
            "python",
            "-m",
            "airplanes.keyframe_finder.generate_test_tuples",
            "--dataset",
            str(captures),
        ]
    )

    logger.info("Building the file with all the scans names")
    sequences = []
    for sequence in (captures / "scans").iterdir():
        sequence_name = sequence.name
        sequences.append(sequence_name)
    with open(captures / "scans.txt", "w") as f:
        for s in sequences:
            f.write(f"{s}\n")


@run.command()
@click.option(
    "--checkpoint",
    help="Path to the model checkpoint",
    type=Path,
    default=Path("checkpoints/airplanes_model.ckpt"),
)
@click.option(
    "--output-dir",
    help="Path to the output directory where meshes and 2D predictions will be saved",
    type=Path,
    default=Path("results"),
)
@click.option(
    "--config",
    help="Name of the config file to use",
    type=str,
    default="inference_on_captures.yaml",
)
def model_inference(checkpoint: str, output_dir: Path, config: str):
    logger.info("Generating TSDFs and 2D embeddings using our model for each capture")
    subprocess.run(
        [
            "python",
            "-m",
            "airplanes.test_2D_network",
            "--config-name",
            config,
            f"load_weights_from_checkpoint={str(checkpoint)}",
            f"output_base_path={output_dir}",
        ]
    )


@run.command()
@click.option(
    "--pred-root",
    help="Path to the directory where non-planar meshes were saved",
    type=str,
    default=Path("results/depth_planes_embeddings/vdr/meshes"),
)
@click.option(
    "--captures", help="Path to captures", default=Path("arbitrary_captures/vdr"), type=Path
)
def train_embeddings(pred_root: Path, captures: Path):
    logger.warning("Training 3D Embeddings (MLPs)")
    subprocess.run(
        [
            "python",
            "-m",
            "airplanes.scene_embeddings_optimisation",
            "--validation-file",
            str(captures / "scans.txt"),
            "--pred-root",
            f"{str(pred_root)}",
            "--embedding-dim",
            "3",
            "--normal-pull-threshold",
            "0.8",
            "--max-depth",
            "3.0",
            "--pixels-per-iteration",
            "400",
            "--frames-per-iteration",
            "50",
            "--num-iterations",
            "10",
            "--embedding-pull-threshold",
            "0.9",
        ]
    )


@run.command()
@click.option(
    "--pred-root",
    help="Path to the directory where non-planar meshes were saved",
    type=str,
    default=Path("results/depth_planes_embeddings/vdr/meshes"),
)
@click.option(
    "--dest-planar-meshes",
    help="Path to the output directory where the planar meshes produced by our model will be saved",
    type=str,
    default=Path("results/depth_planes_embeddings/vdr/meshes/planar_meshes_ours"),
)
@click.option(
    "--captures", help="Path to the captures", default=Path("arbitrary_captures/vdr"), type=Path
)
def run_ransac_ours(pred_root: Path, dest_planar_meshes: Path, captures: Path):
    logger.warning("Running RANSAC using Embeddings")
    subprocess.run(
        [
            "python",
            "-m",
            "airplanes.baselines.sequential_ransac",
            "--validation-file",
            str(captures / "scans.txt"),
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
            "--embeddings-inlier-threshold",
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
    default=Path("results/depth_planes_embeddings/vdr/meshes"),
)
@click.option(
    "--dest-planar-meshes",
    help="Path to the output directory where the planar meshes produced by the baseline will be saved",
    type=str,
    default=Path("results/depth_planes_embeddings/vdr/meshes/planar_meshes_baseline"),
)
@click.option(
    "--captures", help="Path to the captures", default=Path("arbitrary_captures/vdr"), type=Path
)
def run_ransac_baseline(pred_root: Path, dest_planar_meshes: Path, captures: Path):
    logger.warning("Running RANSAC on top of SimpleRecon")
    subprocess.run(
        [
            "python",
            "-m",
            "airplanes.baselines.sequential_ransac",
            "--validation-file",
            str(captures / "scans.txt"),
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


if __name__ == "__main__":
    run()

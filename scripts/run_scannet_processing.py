"""Script to generate planar data for ScanNet scenes.

>>> python -m scripts.run_scannet_processing --scannet /mnt/scannet --output-dir /mnt/scannet-planes-meshes
"""

import subprocess
import click
from pathlib import Path
from loguru import logger


@click.command()
@click.option("--scannet", help="Path to ScanNetv2", default=Path("/mnt/scannet"), type=Path)
@click.option(
    "--output-dir",
    help="Where do you want to save outcomes?",
    default=Path("/mnt/scannet-planes-meshes"),
    type=Path,
)
def cli(scannet: Path, output_dir: Path):
    logger.info("Generating planar data for ScanNetv2")

    # check that we are pointing to ScanNetv2, and not ScanNetv2/scans
    if scannet.name == "scans":
        raise ValueError(
            "Please provide the path to the root of ScanNetv2, not to the 'scans' folder."
        )

    subprocess.run(
        [
            "python",
            "-m",
            "airplanes.data_preparation.generate_ground_truth",
            "--scannet",
            str(scannet / "scans"),
            "--output",
            str(output_dir),
        ]
    )


if __name__ == "__main__":
    cli()  # type:ignore

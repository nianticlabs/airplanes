"""Script to generate visibility volumes for testing scenes in ScanNetv2.

>>> python -m scripts.run_creation_of_visibility_volumes --scannet /mnt/scannet --output-dir /mnt/scannet-planes-meshes --renders /mnt/scannet-planes-renders
"""

import subprocess
import click
from pathlib import Path
from loguru import logger


@click.command()
@click.option("--scannet", help="Path to ScanNetv2", default=Path("/mnt/scannet"), type=Path)
@click.option(
    "--renders",
    help="Where do you want to save outcomes?",
    default=Path("/mnt/scannet-planes-renders"),
    type=Path,
)
@click.option(
    "--output-dir",
    help="Where do you want to save outcomes?",
    default=Path("/mnt/scannet-planes-meshes"),
    type=Path,
)
def cli(scannet: Path, renders: Path, output_dir: Path):
    logger.info("Generating visibility volumes for ScanNetv2")

    # check that we are pointing to ScanNetv2, and not ScanNetv2/scans
    if scannet.name == "scans":
        raise ValueError(
            "Please provide the path to the root of ScanNetv2, not to the 'scans' folder."
        )

    subprocess.run(
        [
            "python",
            "-m",
            "airplanes.data_preparation.generate_visibility_volumes",
            "--scan_data_root",
            str(scannet / "scans"),
            "--output_dir",
            str(output_dir),
            "--rendered_depths_root",
            str(renders),
        ]
    )


if __name__ == "__main__":
    cli()

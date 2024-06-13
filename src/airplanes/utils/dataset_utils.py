from pathlib import Path
from typing import Optional, Type, Union

from airplanes.datasets.generic_mvs_dataset import GenericMVSDataset
from airplanes.datasets.scannet_dataset import ScannetDataset
from airplanes.datasets.scannet_planes_dataset import ScannetPlaneDataset
from airplanes.utils.generic_utils import read_scannetv2_filename, readlines
from src.airplanes.datasets.vdr_dataset import VDRDataset


def get_dataset_depth(
    dataset_name: str,
    split_filepath: Union[str, Path],
    single_debug_scan_id: Optional[str] = None,
    verbose: bool = False,
) -> tuple[Type[GenericMVSDataset], list[str]]:
    """Helper function for passing back the right dataset class, and helps with
    itentifying the scans in a split file.

    dataset_name: a string pointing to the right dataset name, allowed names
        are:
            - scannet
    split_filepath: a path to a text file that contains a list of scans that
        will be passed back as a list called scans.
    single_debug_scan_id: if not None will override the split file and will
        be passed back in scans as the only item.
    verbose: if True will print the dataset name and number of scans.

    Returns:
        dataset_class: A handle to the right dataset class for use in
            creating objects of that class.
        scans: a lit of scans in the split file.
    """
    dataset_class: Type[GenericMVSDataset]

    if dataset_name == "scannet":
        scans = read_scannetv2_filename(filepath=split_filepath)

        if single_debug_scan_id is not None:
            scans = [single_debug_scan_id]

        dataset_class = ScannetDataset
        if verbose:
            print(f"".center(80, "#"))
            print(f" ScanNet Dataset, number of scans: {len(scans)} ".center(80, "#"))
            print(f"".center(80, "#"))
            print("")

    elif dataset_name == "vdr":
        with open(split_filepath) as file:
            scans = file.readlines()
            scans = [scan.strip() for scan in scans]

        if single_debug_scan_id is not None:
            scans = [single_debug_scan_id]

        dataset_class = VDRDataset

        if verbose:
            print(f"".center(80, "#"))
            print(f" VDR Dataset, number of scans: {len(scans)} ".center(80, "#"))
            print(f"".center(80, "#"))
            print("")

    else:
        raise ValueError(f"Not a recognized dataset: {dataset_name}")

    return dataset_class, scans


def get_dataset_plane(
    dataset_name: str,
    split_filepath: Union[str, Path],
    single_debug_scan_id: Optional[str] = None,
    verbose: bool = True,
):
    """Helper function for passing back the right dataset class, and helps with
    itentifying the scans in a split file.

    dataset_name: a string pointing to the right dataset name, allowed names
        are:
            - scannet
    split_filepath: a path to a text file that contains a list of scans that
        will be passed back as a list called scans.
    single_debug_scan_id: if not None will override the split file and will
        be passed back in scans as the only item.
    verbose: if True will print the dataset name and number of scans.

    Returns:
        dataset_class: A handle to the right dataset class for use in
            creating objects of that class.
        scans: a lit of scans in the split file.
    """
    if dataset_name == "scannet":
        scans = read_scannetv2_filename(filepath=split_filepath)

        if single_debug_scan_id is not None:
            scans = [single_debug_scan_id]

        dataset_class = ScannetPlaneDataset
        if verbose:
            print(f"".center(80, "#"))
            print(f" ScanNet Plane Dataset, number of scans: {len(scans)} ".center(80, "#"))
            print(f"".center(80, "#"))
            print("")
    else:
        raise ValueError(f"Not a recognized dataset: {dataset_name}")

    return dataset_class, scans


__AVAILABLE_DATASET_TYPES__ = {"depth": get_dataset_depth, "planes": get_dataset_plane}


def get_dataset(
    dataset_mode: str,
    dataset_name: str,
    split_filepath: Union[str, Path],
    single_debug_scan_id: Optional[str] = None,
    verbose: bool = True,
):
    if dataset_mode not in __AVAILABLE_DATASET_TYPES__.keys():
        raise ValueError(
            f"Cannot find a valid dataset for mode {dataset_mode}. Options are {__AVAILABLE_DATASET_TYPES__.keys()}"
        )
    return __AVAILABLE_DATASET_TYPES__[dataset_mode](
        dataset_name=dataset_name,
        split_filepath=split_filepath,
        single_debug_scan_id=single_debug_scan_id,
        verbose=verbose,
    )


def split_scans_into_separate_files(scan_txtfile: Path, num_subfiles: int) -> list[Path]:
    """
    Loads a txtfile and splits it amongst num_subfiles separate files.
    These will be saved in the parent folder of scan_txtfile
    """
    lines = readlines(scan_txtfile)

    # Calculate the number of lines to write to each output file
    lines_per_file = len(lines) // num_subfiles
    remainder = len(lines) % num_subfiles

    # Create and write to the output files
    output_filepaths = []

    for subfile_idx in range(num_subfiles):
        # Calculate the start and end indices for lines in this output file
        start = subfile_idx * lines_per_file
        end = (
            (subfile_idx + 1) * lines_per_file
            if subfile_idx < num_subfiles - 1
            else (subfile_idx + 1) * lines_per_file + remainder
        )

        # Create the output file path
        output_file_path = scan_txtfile.parent / f"{scan_txtfile.stem}_{subfile_idx}.txt"

        # Write lines to the output file
        with output_file_path.open("w") as output_file:
            for line in lines[start:end]:
                output_file.write(line + "\n")

        output_filepaths.append(output_file_path)

    return output_filepaths

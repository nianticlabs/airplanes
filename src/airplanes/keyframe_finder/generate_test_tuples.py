"""
Script that computes tuples for arbitrary captured scans.

From SimpleRecon
https://github.com/nianticlabs/simplerecon/blob/main/data_scripts/generate_test_tuples.py

Usage:
>>> python -m scripts.data_scripts.generate_test_tuples -dataset $SEQUENCE
"""


import os
import random
from multiprocessing import Manager
from pathlib import Path

import click
import numpy as np
from omegaconf import DictConfig

from airplanes.datasets import vdr_dataset
from airplanes.keyframe_finder import keyframe_buffer as kfb
from airplanes.keyframe_finder.keyframe_buffer import DVMVS_Config


def compute_offline_tuple(
    poses,
    n_measurement_frames,
    current_keyframe_index,
    reference_pose,
):
    """
    Computes an offline tuple by scanning keyframes backwards and forwards in
    time.

    Args:
        poses: poses for the scan
        n_measurement_frames: number of measurement (source) frames required for
            each tuple. Total number of frames in the tuple is
            n_measurement_frames+1
        current_keyframe_index: the current frame
        reference_pose: the pose of the current frame at current_keyframe_index

        Returns:
        sample: a dictionary with the key indices whose values are the indices
            of the tuple.
    """
    sample = {"indices": [current_keyframe_index]}

    tuple_keyframe_buffer = kfb.OfflineKeyframeBuffer(
        buffer_size=DVMVS_Config.test_keyframe_buffer_size * 2,
        keyframe_pose_distance=DVMVS_Config.test_keyframe_pose_distance,
        optimal_t_score=DVMVS_Config.test_optimal_t_measure,
        optimal_R_score=DVMVS_Config.test_optimal_R_measure,
        store_return_indices=True,
    )

    # prime the buffer with the reference pose
    response = tuple_keyframe_buffer.try_new_keyframe(
        reference_pose.copy(),
        None,
        index=current_keyframe_index,
    )

    current_backwards_index = current_keyframe_index - 1
    current_forwards_index = current_keyframe_index + 1
    direction = True
    count_added = 0
    exhausted_forward = False
    exhausted_backward = False

    while not exhausted_forward or not exhausted_backward:
        # step away from the current frame alternating in both directions
        # fill the buffer with keyframes from both direcitons as we go.
        if exhausted_forward and exhausted_backward:
            break

        # check which way we should be looking
        if direction:
            # going forward, so get the forward pose to update the buffer
            direction = False
            if current_forwards_index >= len(poses):
                exhausted_forward = True
                continue
            new_frame_index = current_forwards_index
            new_frame_pose = poses[new_frame_index].copy()
            current_forwards_index += 1

        else:
            # going backward, so get the forward pose to update the buffer
            direction = True
            if current_backwards_index < 0:
                exhausted_backward = True
                continue
            new_frame_index = current_backwards_index
            new_frame_pose = poses[new_frame_index].copy()
            current_backwards_index -= 1

        # poll the buffer,
        response = tuple_keyframe_buffer.try_new_keyframe(
            new_frame_pose, None, index=new_frame_index
        )

        if response == 1:
            count_added += 1

        # if we have enough frames, then stop.
        if count_added >= DVMVS_Config.test_keyframe_buffer_size * 2:
            break

    # the buffer is full, now get source frames.
    measurement_frames = tuple_keyframe_buffer.get_best_measurement_frames_for_0index(
        n_measurement_frames
    )

    for _, _, measurement_index in measurement_frames:
        sample["indices"].append(measurement_index)

    return sample


def default_dvmvs_tuples(scan, poses, dists_to_last_valid, n_measurement_frames):
    """
    Creates a list of default DVMVS tuples for a scan's poses. Only tuples at
    keyframes will be returned. Each tuple will be online (frames will be behind
    the current frame in time.)

    This will not return a tuple for the first frame.

    Args:
        scan: scan id string.
        poses: a list of poses.
        dists_to_last_valid: list with distances from each current valid frame
            to the last valid frame. Pass Nones if you you're passing invalid
            poses and want the DVMVS keyframe handler to take care of them.
        n_measurement_frames: number of measurement (source) frames required for
            each tuple. Total number of frames in the tuple is
            n_measurement_frames+1

    Returns:
        samples: a list of dictionaries where each item dicitonary contians
            a scan's id and a list of indices for the tuple's frames.
    """
    keyframe_buffer = kfb.KeyframeBuffer(
        buffer_size=DVMVS_Config.test_keyframe_buffer_size,
        keyframe_pose_distance=DVMVS_Config.test_keyframe_pose_distance,
        optimal_t_score=DVMVS_Config.test_optimal_t_measure,
        optimal_R_score=DVMVS_Config.test_optimal_R_measure,
        store_return_indices=True,
    )

    samples = []
    for i in range(0, len(poses)):
        sample = {"scan": scan, "indices": [i]}

        reference_pose = poses[i].copy()

        # POLL THE KEYFRAME BUFFER
        response = keyframe_buffer.try_new_keyframe(
            reference_pose, None, dists_to_last_valid[i], index=i
        )
        if response == 3:
            print("Tracking lost!")

        elif response == 1:
            measurement_frames = keyframe_buffer.get_best_measurement_frames(n_measurement_frames)

            for _, _, measurement_index in measurement_frames:
                sample["indices"].append(measurement_index)

            samples.append(sample)

    return samples


def offline_dvmvs_tuples(scan, poses, n_measurement_frames):
    """
    Creates a list of offline DVMVS tuples for a scan's poses. Only tuples at
    keyframes will be returned. This will provide frames both ahead of and
    behind the current frame.

    This will not return a tuple for the first frame.

    Args:
        scan: scan id string.
        poses: a list of poses.
        n_measurement_frames: number of measurement (source) frames required for
            each tuple. Total number of frames in the tuple is
            n_measurement_frames+1

    Returns:
        samples: a list of dictionaries where each item dicitonary contians
            a scan's id and a list of indices for the tuple's frames.
    """

    samples = []

    keyframe_buffer = kfb.KeyframeBuffer(
        buffer_size=DVMVS_Config.test_keyframe_buffer_size,
        keyframe_pose_distance=DVMVS_Config.test_keyframe_pose_distance,
        optimal_t_score=DVMVS_Config.test_optimal_t_measure,
        optimal_R_score=DVMVS_Config.test_optimal_R_measure,
        store_return_indices=True,
    )

    for i in range(0, len(poses)):
        reference_pose = poses[i].copy()

        response = keyframe_buffer.try_new_keyframe(pose=reference_pose.copy(), image=None, index=i)

        if response != 1:
            continue

        sample = compute_offline_tuple(poses, n_measurement_frames, i, reference_pose.copy())
        sample["scan"] = scan

        if len(sample["indices"]) == 1 and i == 0:
            # first frame and no indices available.
            continue
        else:
            samples.append(sample)
    return samples


def dense_dvmvs_tuples(scan, poses, n_measurement_frames):
    """
    Creates a list of DVMVS online tuples for a scan's poses at each frame.

    This will not return a tuple for the first frame.

    Args:
        scan: scan id string.
        poses: a list of poses.
        n_measurement_frames: number of measurement (source) frames required for
            each tuple. Total number of frames in the tuple is
            n_measurement_frames+1

    Returns:
        samples: a list of dictionaries where each item dicitonary contians
            a scan's id and a list of indices for the tuple's frames.
    """

    samples = []

    for i in range(0, len(poses)):
        sample = {"scan": scan, "indices": [i]}

        reference_pose = poses[i]

        keyframe_buffer = kfb.OfflineKeyframeBuffer(
            buffer_size=DVMVS_Config.test_keyframe_buffer_size,
            keyframe_pose_distance=DVMVS_Config.test_keyframe_pose_distance,
            optimal_t_score=DVMVS_Config.test_optimal_t_measure,
            optimal_R_score=DVMVS_Config.test_optimal_R_measure,
            store_return_indices=True,
        )

        response = keyframe_buffer.try_new_keyframe(reference_pose, None, index=i)

        num_available_measurement_frames = 0
        current_backwards_index = i - 1
        exhausted_backward = False
        count_added = 0
        while num_available_measurement_frames != n_measurement_frames:
            # POLL THE KEYFRAME BUFFER
            if exhausted_backward:
                break
            if current_backwards_index < 0:
                exhausted_backward = True
                break

            new_frame_index = current_backwards_index
            new_frame_pose = poses[new_frame_index]
            current_backwards_index -= 1

            response = keyframe_buffer.try_new_keyframe(new_frame_pose, None, index=new_frame_index)

            if response == 1:
                count_added += 1

            if count_added >= DVMVS_Config.test_keyframe_buffer_size:
                break

        measurement_frames = keyframe_buffer.get_best_measurement_frames_for_0index(
            n_measurement_frames
        )
        for _, _, measurement_index in measurement_frames:
            sample["indices"].append(measurement_index)

        if len(sample["indices"]) == 1 and i == 0:
            # first frame and no indices available.
            continue
        else:
            samples.append(sample)

    return samples


def offline_dense_dvmvs_tuples(scan, poses, n_measurement_frames):
    """
    Creates a list of DVMVS offline tuples for a scan's poses at each frame.

    This will not return a tuple for the first frame. For frames where there are
    not enough keyframes to build the tuple (near the beginning of the sequency)
    frame indices will be repeated to compensate and pad the tuple.

    Args:
        scan: scan id string.
        poses: a list of poses.
        n_measurement_frames: number of measurement (source) frames required for
            each tuple. Total number of frames in the tuple is
            n_measurement_frames+1

    Returns:
        samples: a list of dictionaries where each item dicitonary contians
            a scan's id and a list of indices for the tuple's frames.
    """

    samples = []

    for i in range(0, len(poses)):
        sample = {"scan": scan, "indices": [i]}

        reference_pose = poses[i]

        sample = compute_offline_tuple(poses, n_measurement_frames, i, reference_pose)
        sample["scan"] = scan

        if len(sample["indices"]) == 1 and i == 0:
            # first frame and no indices available.
            continue
        else:
            samples.append(sample)
    return samples


def crawl_subprocess_long(opts, scan, count, progress):
    """
    Returns a list of tuples according to the options at opts_temp_filepath.

    This will not return a tuple for the first frame. For frames where there are
    not enough keyframes to build the tuple (near the beginning of the sequency)
    frame indices will be repeated to compensate and pad the tuple.

    Args:
        opts_temp_filepath: filepath for an options config file.
        scan: scan to operate on.
        count: total count of multi process scans.
        progress: a Pool() progress value for tracking progress. For debugging
            you can pass
                multiprocessing.Manager().Value('i', 0)
            for this.

    Returns:
        item_list: a list of strings where each string is the cocnatenated
            scan id and frame_ids for every tuple.
                scan_id frame_id_0 frame_id_1 ... frame_id_N-1

    """

    item_list = []

    # get dataset
    dataset_class = vdr_dataset.VDRDataset

    ds = dataset_class(
        dataset_path=opts.data.dataset_path,
        mv_tuple_file_suffix=None,
        data_opts=opts.data,
        split=opts.split,
        tuple_info_file_location=opts.data.tuple_info_file_location,
        pass_frame_id=True,
        verbose_init=False,
    )

    valid_frames = ds.get_valid_frame_ids(opts.split, scan)

    try:
        check_valid_dist = int(valid_frames[0].strip().split(" ")[2])
    except:
        if opts.data.frame_tuple_type == "default":
            print(
                f"\nWARNING: Couldn't find max valid distances in the valid_frames "
                f"file for scan {scan}. Please delete existing valid_frames.txt "
                f"and rerun to regenerate valid frames. "
                f"There's a difference of 9 extra frames out of 25599 for the "
                f"ScanNetv2 test set.\n"
            )

    dists_to_last_valid = []

    frame_ind_to_frame_id = {}
    for frame_ind, frame_line in enumerate(valid_frames):
        frame_ind_to_frame_id[frame_ind] = frame_line.strip().split(" ")[1]
        try:
            dists_to_last_valid.append(int(frame_line.strip().split(" ")[2]))
        except:
            # just add Noness
            dists_to_last_valid.append(None)

    poses = []
    for frame_ind in range(len(valid_frames)):
        frame_id = frame_ind_to_frame_id[frame_ind]
        world_T_cam_44, _ = ds.load_pose(scan.rstrip("\n"), frame_id)
        poses.append(world_T_cam_44)

    subsequence_length = opts.data.num_images_in_tuple
    n_measurement_frames = subsequence_length - 1

    if opts.data.frame_tuple_type == "default":
        samples = default_dvmvs_tuples(scan, poses, dists_to_last_valid, n_measurement_frames)
    elif opts.data.frame_tuple_type == "offline":
        samples = offline_dvmvs_tuples(scan, poses, n_measurement_frames)
    elif opts.data.frame_tuple_type == "dense":
        samples = dense_dvmvs_tuples(scan, poses, n_measurement_frames)
    elif opts.data.frame_tuple_type == "dense_offline":
        samples = offline_dense_dvmvs_tuples(scan, poses, n_measurement_frames)
    else:
        ValueError(f"Not a recognized tuple frame type: " f"{opts.data.frame_tuple_type}")

    num_repeats = 0
    for sample in samples:
        sampled_indices = sample["indices"]

        if len(sampled_indices) != subsequence_length:
            # not enough frames in the buffer.

            # get all frames seen so far that aren't keyframes
            available_indices = list(range(0, sampled_indices[0]))
            available_indices = [
                frame_ind for frame_ind in available_indices if frame_ind not in sampled_indices
            ]

            # pick from frames that haven't been touched yet (available_indices).
            diff = subsequence_length - len(sampled_indices)
            if diff > len(available_indices):
                diff = len(available_indices)

            # preferably pick from recent frames
            back_search_dist = 30 if len(available_indices) >= 30 else len(available_indices)

            sampled_indices += random.sample(available_indices[-back_search_dist:], k=diff)

            # check again in case we still don't have enough and random sample
            # repeat if we don't
            if len(sampled_indices) != subsequence_length:
                diff = subsequence_length - len(sampled_indices)
                num_repeats += diff
                sampled_indices += random.choices(sampled_indices[1:], k=diff)

        assert len(sampled_indices) == subsequence_length

        chosen_frame_ids = [frame_ind_to_frame_id[frame_ind] for frame_ind in sampled_indices]

        cat_ids = " ".join([str(frame_id) for frame_id in chosen_frame_ids])
        item_list.append(f"{scan} {cat_ids}")

    progress.value += 1
    print(
        f"Completed scan {scan}, {progress.value} of total {count}. "
        f"# Samples: {len(samples)} of {len(poses)} poses. "
        f"Samples with repeated frames: {num_repeats}.\r"
    )

    return item_list


@click.command()
@click.option("--dataset", type=Path, help="Path to the dataset", required=True)
def cli(dataset: Path):
    np.random.seed(42)
    random.seed(42)

    item_list = []
    for scan in (dataset / "scans").iterdir():
        if not scan.is_dir():
            continue

        opts = DictConfig(
            {
                "split": "test",
                "single_debug_scan_id": scan.name,
                "data": {
                    "dataset_path": str(dataset),
                    "mv_tuple_file_suffix": "_eight_view_deepvmvs.txt",
                    "tuple_info_file_location": str(dataset),
                    "frame_tuple_type": "default",
                    "num_images_in_tuple": 8,
                },
            }
        )

        split_filename = f"{opts.split}{opts.data.mv_tuple_file_suffix}"
        split_filepath = os.path.join(opts.data.tuple_info_file_location, split_filename)
        print(f"Saving to {split_filepath}\n")

        item_list += crawl_subprocess_long(
            opts,
            opts.single_debug_scan_id,
            0,
            Manager().Value("i", 0),
        )

    with open(split_filepath, "w") as f:
        for line in item_list:
            f.write(line + "\n")

    print(f"Saved to {split_filepath}")


if __name__ == "__main__":
    cli()

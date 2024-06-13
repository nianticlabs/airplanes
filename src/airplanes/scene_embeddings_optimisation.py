import time
from pathlib import Path

import click
import numpy as np
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

from airplanes.utils.generic_utils import read_scannetv2_filename
from airplanes.utils.geometry_utils import BackprojectDepth, NormalGenerator
from airplanes.utils.io_utils import AssetFileNames


class MLP(nn.Module):
    def __init__(self, num_outputs: int = 3, num_harmonics: int = 24, num_inputs: int = 3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_harmonics * num_inputs * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_outputs),
        )
        self.harmonic_embeddor = HarmonicEmbedding(num_harmonics)

    def forward(self, x: torch.Tensor):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x).cuda()
        x = self.harmonic_embeddor(x)
        return self.mlp(x)


class HarmonicEmbedding(torch.nn.Module):
    def __init__(self, n_harmonic_functions: int = 24, omega0: float = 0.1):
        super().__init__()
        self.register_buffer(
            "frequencies",
            omega0 * (2.0 ** torch.arange(n_harmonic_functions)),
        )

    def forward(self, x: torch.Tensor):
        embed = (x[..., None] * self.frequencies).reshape(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)


class SceneOptimiser:
    def __init__(
        self,
        num_harmonics: int,
        num_iterations: int,
        learning_rate: float = 0.01,
        embedding_dim: int = 3,
        pixels_per_iteration: int = 200,
        frames_per_iteration: int = 50,
        embedding_pull_threshold: float = 0.5,
        normal_pull_threshold: float = 0.8,
        max_depth: float = 5.0,
        push_threshold=1.0,
    ) -> None:
        self.num_harmonics = num_harmonics
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim
        self.pixels_per_iteration = pixels_per_iteration
        self.frames_per_iteration = frames_per_iteration
        self.embedding_pull_threshold = embedding_pull_threshold
        self.normal_pull_threshold = normal_pull_threshold
        self.max_depth = max_depth
        self.push_threshold = push_threshold

        logger.info(f"{'embedding_pull_threshold':<30}: {self.embedding_pull_threshold}")
        logger.info(f"{'normal_pull_threshold':<30}: {self.normal_pull_threshold}")
        logger.info(f"{'max_depth':<30}: {self.max_depth}")
        logger.info(f"{'push_threshold':<30}: {self.push_threshold}")
        logger.info(f"{'num_harmonics':<30}: {self.num_harmonics}")

    def initialise(self):
        np.random.seed(10)
        torch.manual_seed(10)
        torch.cuda.manual_seed(10)

        self.mlp = MLP(num_outputs=self.embedding_dim, num_harmonics=self.num_harmonics).cuda()
        self.optimiser = torch.optim.Adam(self.mlp.parameters(), lr=self.learning_rate)
        self.world_points_M3N = []
        self.embeddings_M3N = []
        self.normals_M3N = []
        self.depths_M1N = []

        self.backprojector = None
        self.normal_generator = None

    def load_frame_predictions(self, frame_predictions_path: Path):
        """Load predictions of our model for each keyframe in the sequence.
        For each 3D point we memorise the world coordinates, the embedding, the normal and the depth
        value.
        """
        self.keyframe_ids = []
        files = sorted(frame_predictions_path.glob("*.pickle"))
        assert len(files) > 0, f"No files found in {frame_predictions_path}"
        for file in files:
            self.keyframe_ids.append(file.stem)
            frame_predictions = np.load(file, allow_pickle=True)

            depth = torch.tensor(frame_predictions["depth_pred_s0_b1hw"])

            if self.backprojector is None:
                self.backprojector = BackprojectDepth(depth.shape[-2], depth.shape[-1])
                self.normal_generator = NormalGenerator(depth.shape[-2], depth.shape[-1])

            world_T_cam = torch.tensor(frame_predictions["world_T_cam_b44"])
            invK = torch.linalg.inv(torch.tensor(frame_predictions["K_s0_b44"]))
            normal = self.normal_generator(depth, invK)
            self.normals_M3N.append(normal)
            self.depths_M1N.append(depth)

            cam_points = self.backprojector.forward(depth, invK)
            world_points_13N = (world_T_cam @ cam_points)[:, :3]
            self.world_points_M3N.append(world_points_13N)

            embedding = torch.tensor(frame_predictions["embedding_pred_s0_b3hw"][0])
            self.embeddings_M3N.append(embedding)

        self.world_points_M3N = torch.cat(self.world_points_M3N).cuda()
        self.embeddings_M3N = torch.stack(self.embeddings_M3N).flatten(2).cuda()
        self.normals_M3N = torch.cat(self.normals_M3N).flatten(2).cuda()
        self.depths_M1N = torch.cat(self.depths_M1N).flatten(1).cuda()

    def optimise_scene(self, frame_predictions_path: Path, savepath: Path, scene: str):
        """Optimise the MLP on a given scene.
        We impose  a push-pull loss, forcing 3D embeddings to be close when their 2D counterparts are
        similar, and to be distant if 2D embeddings are different.
        """
        self.initialise()
        self.load_frame_predictions(frame_predictions_path)

        frames_in_buffer = 0
        total_iterations = self.num_iterations * len(self.world_points_M3N)
        start = time.time()
        for iteration in range(total_iterations):
            if iteration % self.num_iterations == 0 and frames_in_buffer < len(
                self.world_points_M3N
            ):
                frames_in_buffer += 1

            if frames_in_buffer < self.frames_per_iteration:
                im_ids = torch.arange(frames_in_buffer, device="cuda:0")
            else:
                near_ids = torch.arange(
                    start=frames_in_buffer - 10, end=frames_in_buffer, device="cuda:0"
                )
                far_ids = torch.randperm(frames_in_buffer - 10, device="cuda:0")[
                    : self.frames_per_iteration - 10
                ]
                im_ids = torch.cat((near_ids, far_ids))

            pix_ids = torch.randperm(self.world_points_M3N.shape[-1], device="cuda:0")[
                : self.pixels_per_iteration
            ]

            depth_MN1 = self.depths_M1N[im_ids][..., pix_ids].unsqueeze(-1)
            input_coords_MN3 = self.world_points_M3N[im_ids][..., pix_ids].permute(0, 2, 1)
            embed_3d = self.mlp(input_coords_MN3) * (depth_MN1 < self.max_depth).float()
            embed_2d = (
                self.embeddings_M3N[im_ids][..., pix_ids].permute(0, 2, 1)
                * (depth_MN1 < self.max_depth).float()
            )

            normal_2d = self.normals_M3N[im_ids][..., pix_ids]

            diff_2d = torch.norm(embed_2d.unsqueeze(1) - embed_2d.unsqueeze(2), dim=-1)
            diff_3d = torch.norm(embed_3d.unsqueeze(1) - embed_3d.unsqueeze(2), dim=-1)
            diff_normal_2d = (normal_2d.unsqueeze(2) * normal_2d.unsqueeze(3)).sum(1)

            # pull-push loss
            mask = torch.logical_and(
                diff_2d < self.embedding_pull_threshold,
                diff_normal_2d > self.normal_pull_threshold,
            )
            mask = mask.float()
            inv_mask = 1 - mask
            pull_loss = (diff_3d * mask).sum() / mask.sum()
            push_loss = (F.relu(self.push_threshold - diff_3d) * inv_mask).sum() / inv_mask.sum()
            loss = pull_loss + push_loss

            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

        logger.info(
            f"Optimisation took {time.time() - start:.3f}s. Per iteration {(time.time() - start) / total_iterations:.3f}s"
        )


def optimise_one_scene(
    pred_root: Path,
    savepath: Path,
    num_harmonics: int,
    embedding_dim: int,
    scene_optimiser: SceneOptimiser,
    scene: str,
) -> None:
    """
    Optimise an MLP to predict scene embeddings given a 3d coordinate as input.
    Save the MLP weights.
    params:
        pred_root: path to the predicted tsdfs
        savepath: folder that contains final results
        num_harmonics: number of harmonics to use for the MLP input
        embedding_dim: dimension of the embedding
        scene_optimiser: SceneOptimiser object
        scene: scene name
    """

    # run optimisation
    frame_predictions_path = pred_root.parent / "depths" / scene

    mlp_weights_path = savepath / scene
    mlp_weights_path.mkdir(parents=True, exist_ok=True)

    scene_optimiser.optimise_scene(frame_predictions_path, savepath, scene)

    # save the MLP weights
    torch.save(
        scene_optimiser.mlp.state_dict(),
        mlp_weights_path
        / AssetFileNames.get_embeddings_mlp_filename(
            scene_name=scene, num_harmonics=num_harmonics, embedding_dim=embedding_dim
        ),
    )


def optimise_scenes_multigpu(
    scenes: list,
    pred_root: Path,
    savepath: Path,
    num_harmonics: int,
    embedding_dim: int,
    scene_optimiser: SceneOptimiser,
    num_workers_per_gpu: int,
) -> None:
    """
    Optimise an MLP to predict scene embeddings given a 3d coordinate as input using ray for optimisation with multiple GPUs.
    params:
        scenes: list of scene names
        pred_root: path to the predicted tsdfs
        savepath: folder that contains final results
        num_harmonics: number of harmonics to use for the MLP input
        embedding_dim: dimension of the embedding
        scene_optimiser: SceneOptimiser object
        num_workers_per_gpu: number of workers to run per GPU when using ray
    """

    # Initialize ray with GPUs
    ray.init(ignore_reinit_error=True, num_cpus=None, num_gpus=None)

    # Define a remote function
    # NOTE: fractional num_gpus allow for multiple tasks to run on a single GPU
    # num_gpus=0.5 means that upto 2 tasks can run on a single GPU simultaneously
    @ray.remote(num_gpus=1 / num_workers_per_gpu)
    def gpu_worker_function(
        pred_root: Path,
        savepath: Path,
        num_harmonics: int,
        embedding_dim: int,
        scene_optimiser: SceneOptimiser,
        scene: str,
    ):
        optimise_one_scene(
            pred_root=pred_root,
            savepath=savepath,
            num_harmonics=num_harmonics,
            embedding_dim=embedding_dim,
            scene_optimiser=scene_optimiser,
            scene=scene,
        )

    # Create a list of futures
    gpu_futures = []
    for scene_name in scenes:
        gpu_futures.append(
            gpu_worker_function.remote(
                pred_root=pred_root,
                savepath=savepath,
                num_harmonics=num_harmonics,
                embedding_dim=embedding_dim,
                scene_optimiser=scene_optimiser,
                scene=scene_name,
            )
        )

    # Progress bar with tqdm
    pbar = tqdm(total=len(gpu_futures), desc="All scans", dynamic_ncols=True, unit="scan")

    # Loop until all futures are done
    while len(gpu_futures) > 0:
        # Check if any future is ready
        ready_futures, remaining_futures = ray.wait(gpu_futures, timeout=0.1)

        # Get the results from the ready futures
        ray.get(ready_futures)

        # Update progress bar
        pbar.update(len(ready_futures))

        # Update the list of futures
        gpu_futures = remaining_futures

    # Close progress bar
    pbar.close()

    # Shutdown ray
    ray.shutdown()


@click.command()
@click.option("--pred-root", type=Path, help="Path to predicted TSDFs", required=True)
@click.option(
    "--output-dir",
    type=Path,
    help="folder that contains final results",
    default=None,
)
@click.option(
    "--validation-file",
    type=Path,
    default=Path("src/airplanes/data_splits/ScanNetv2/standard_split/scannetv2_test_planes.txt"),
    help="Path to the file that contains the test scenes",
)
@click.option(
    "--num-harmonics",
    type=int,
    default=24,
    help="Number of harmonics to use for the MLP input",
)
@click.option(
    "--num-iterations",
    type=int,
    default=1000,
    help="Number of iterations to run the optimiser for",
)
@click.option(
    "--pixels-per-iteration",
    type=int,
    default=400,
    help="Number of pixels to sample per iteration",
)
@click.option(
    "--frames-per-iteration",
    type=int,
    default=50,
    help="Number of frames to sample per iteration",
)
@click.option(
    "--embedding-dim",
    type=int,
    default=3,
)
@click.option(
    "--embedding-pull-threshold",
    type=float,
    default=0.9,
)
@click.option(
    "--normal-pull-threshold",
    type=float,
    default=0.8,
)
@click.option(
    "--max-depth",
    type=float,
    default=3.0,
)
@click.option(
    "--push-threshold",
    type=float,
    default=1.0,
)
@click.option(
    "--num-workers-per-gpu",
    type=int,
    default=1,
    show_default=True,
    help="Number of workers to run per GPU when using ray",
)
def cli(
    pred_root: Path,
    output_dir: Path,
    validation_file: Path,
    num_harmonics: int,
    num_iterations: int,
    pixels_per_iteration: int,
    frames_per_iteration: int,
    embedding_dim: int,
    embedding_pull_threshold: float,
    normal_pull_threshold: float,
    max_depth: float,
    push_threshold: float,
    num_workers_per_gpu: int,
):
    """
    Optimise an MLP to predict consistent scene embeddings.
    """

    scene_optimiser = SceneOptimiser(
        num_harmonics=num_harmonics,
        num_iterations=num_iterations,
        embedding_dim=embedding_dim,
        pixels_per_iteration=pixels_per_iteration,
        frames_per_iteration=frames_per_iteration,
        embedding_pull_threshold=embedding_pull_threshold,
        normal_pull_threshold=normal_pull_threshold,
        max_depth=max_depth,
        push_threshold=push_threshold,
    )

    savepath = pred_root if output_dir is None else output_dir
    savepath.mkdir(parents=True, exist_ok=True)

    # select only eval scenes
    scenes = read_scannetv2_filename(filepath=validation_file)

    optimise_scenes_multigpu(
        scenes=scenes,
        pred_root=pred_root,
        savepath=savepath,
        num_harmonics=num_harmonics,
        embedding_dim=embedding_dim,
        scene_optimiser=scene_optimiser,
        num_workers_per_gpu=num_workers_per_gpu,
    )


if __name__ == "__main__":
    cli()

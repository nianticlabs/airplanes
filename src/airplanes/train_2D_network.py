import os

import hydra
import pytorch_lightning as pl
import torch
from loguru import logger as loguru_logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from airplanes.utils.dataset_utils import get_dataset
from airplanes.utils.generic_utils import copy_code_state
from src.airplanes.experiment_modules.depth_planes_embeddings_model import (
    DepthPlanesEmbeddingsModel,
)


@hydra.main(
    version_base=None, config_path=os.getcwd() + "/configs", config_name="train_2D_network.yaml"
)
def hydra_main_wrapper(opts: DictConfig):
    main(opts)


def main(opts: DictConfig):
    if opts.gpus == 0:
        print("Setting precision to 32 bits since --gpus is set to 0.")
        opts.precision = 32

    # set seed
    pl.seed_everything(opts.random_seed)

    model_class = DepthPlanesEmbeddingsModel
    if opts.load_weights_from_checkpoint is not None:
        model = model_class.load_from_checkpoint(
            opts.load_weights_from_checkpoint, opts=opts, args=None
        )
    if opts.get("lazy_load_weights_from_checkpoint", None) is not None:
        model = model_class(opts)
        state_dict = torch.load(opts.lazy_load_weights_from_checkpoint)["state_dict"]
        available_keys = list(state_dict.keys())
        for param_key, param in model.named_parameters():
            if param_key in available_keys:
                try:
                    if isinstance(state_dict[param_key], torch.nn.Parameter):
                        # backwards compatibility for serialized parameters
                        param = state_dict[param_key].data
                    else:
                        param = state_dict[param_key]

                    model.state_dict()[param_key].copy_(param)
                except:
                    print(f"WARNING: could not load weights for {param_key}")
    else:
        # load model using read options
        model = model_class(opts)

    # load dataset and dataloaders
    dataset_class, _ = get_dataset(
        dataset_mode=opts.data.type,
        dataset_name=opts.data.dataset,
        split_filepath=opts.data.dataset_scan_split_file,
        single_debug_scan_id=opts.single_debug_scan_id,
    )

    train_dataset = dataset_class(
        opts.data.dataset_path,
        split="train",
        mv_tuple_file_suffix=opts.data.mv_tuple_file_suffix,
        data_opts=opts.data,
        num_images_in_tuple=opts.data.num_images_in_tuple,
        tuple_info_file_location=opts.data.tuple_info_file_location,
        image_width=opts.image_width,
        image_height=opts.image_height,
        shuffle_tuple=opts.shuffle_tuple,
        load_src_depth=opts.data.load_src_depth,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=opts.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    val_dataset = dataset_class(
        opts.data.dataset_path,
        split="val",
        mv_tuple_file_suffix=opts.data.mv_tuple_file_suffix,
        data_opts=opts.data,
        num_images_in_tuple=opts.data.num_images_in_tuple,
        tuple_info_file_location=opts.data.tuple_info_file_location,
        image_width=opts.image_width,
        image_height=opts.image_height,
        include_full_res_depth=opts.high_res_validation,
        load_src_depth=opts.data.load_src_depth,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=opts.val_batch_size,
        shuffle=False,
        num_workers=opts.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    # set up a tensorboard logger through lightning
    logger = TensorBoardLogger(save_dir=opts.log_dir, name=opts.models.name)

    # This will copy a snapshot of the code (minus whatever is in .gitignore)
    # into a folder inside the main log directory. We also save the config
    copy_code_state(path=os.path.join(logger.log_dir, "code"))

    with open(os.path.join(logger.log_dir, "config.yaml"), "w") as fp:
        OmegaConf.save(opts, fp)
        loguru_logger.info(f"saving config at: {logger.log_dir}/config.yaml")

    # set a checkpoint callback for lignting to save model checkpoints
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_last=True,
        save_top_k=3,
        verbose=True,
        monitor="val/embedding_loss",
        mode="min",
    )

    # keep track of changes in learning rate
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        devices=opts.gpus,
        log_every_n_steps=opts.log_interval,
        val_check_interval=opts.val_interval,
        limit_val_batches=opts.val_batches,
        max_steps=opts.max_steps,
        precision=opts.precision,
        benchmark=True,
        logger=logger,
        sync_batchnorm=False,
        callbacks=[checkpoint_callback, lr_monitor],
        num_sanity_val_steps=opts.num_sanity_val_steps,
        strategy="ddp_find_unused_parameters_true",
    )

    # start training
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=opts.resume)


if __name__ == "__main__":
    hydra_main_wrapper()  # type: ignore

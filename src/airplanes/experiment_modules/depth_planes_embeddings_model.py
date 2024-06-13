import logging

import pytorch_lightning as pl
import timm
import torch
import torch.nn.functional as F
from torch import nn

from airplanes.losses import hinge_embedding_loss
from airplanes.modules.cost_volume import CostVolumeManager, FeatureVolumeManager
from airplanes.modules.fast_networks import SkipDecoderRegression
from airplanes.modules.layers import TensorFormatter
from airplanes.modules.networks import (
    CVEncoder,
    DepthDecoderPP,
    ResnetMatchingEncoder,
    UNetMatchingEncoder,
)
from airplanes.utils.augmentation_utils import CustomColorJitter
from airplanes.utils.generic_utils import (
    reverse_imagenet_normalize,
    tensor_B_to_bM,
    tensor_bM_to_B,
)
from airplanes.utils.geometry_utils import NormalGenerator
from airplanes.utils.metrics_utils import compute_depth_metrics
from airplanes.utils.segmentation_utils import compute_iou
from airplanes.utils.visualization_utils import colormap_image, colormap_planes

logger = logging.getLogger(__name__)

from itertools import chain


class DepthPlanesEmbeddingsModel(pl.LightningModule):
    """Class for SimpleRecon depth estimators + planes heads.

    This class handles training and inference for SR + planes models.

    Depth maps will be predicted

    It houses experiments for a vanilla cost volume that uses dot product
    reduction and the full feature volume.

    It also allows for experimentation on the type of image encoder.

    The opts the model is first initialized with will be saved as part of
    hparams. On load from checkpoint the model will use those stores
    options. It's generally a good idea to use those directly, unless you
    want to do something fancy with changing resolution, since some
    projective modules use initialized spatial grids for
    projection/backprojection.

    Attributes:
        run_opts: options object with flags.
        matching_encoder_type: the model used for generating image features for
            matching in the cost volume.
        cost_volume: the cost volume module.
        encoder: the image encoder used to enforce an image prior.
        cost_volume_net: the first half of the U-Net for encoding cost
            volume features and image prior features.
        depth_decoder: second half of the U-Net for decoding feautures into
            depth maps at multiple resolutions.
        compute_normals: module for computing normals
        tensor_formatter: helper class for reshaping tensors - batching and
            unbatching views.

    A note on losses: we report more losses than we actually use for
    backprop.

    We use the term cur/current for the refernce frame (from the paper)whose
    depth we predict and src/soruce for source (neighborhood) frames used
    for matching.

    """

    def __init__(self, opts):
        """Inits a depth model.

        Args: opts: Options config object with:

        opts.image_encoder_name: the type of image encoder used for the
            image prior. Supported in this version is EfficientNet.
        opts.cv_encoder_type: the type of cost volume encoder to use. The
            only supported version here is a multi_scale encoder that takes
            in features from the cost volume and features at multiple scales
            from the encoder used for image priors.
        opts.matching_num_depth_bins: number of depth planes used for
            MVS in the cost volume.
        opts.matching_scale: w.r.t to the predicted depth map, this the
            scale at which we match in the cost volume. 0 indicates matching
            at the resolution of the depth map. 1 indicates matching at half
            that resolution.
        opts.depth_decoder_name: the type of decoder to use for decoding
            features into depth. We're using a U-Net++ like architure in
            SimpleRecon.
        opts.image_width, opts.image_height: incoming image width and
            height.
        opts.loss_type: type of loss to use at multiple scales. Final
            supported verison here is log_l1.
        opts.feature_volume_type: the type of cost volume to use. Supported
            types are simple_cost_volume for a dot product based reduction
            or an mlp_feature_volume for metadata laced feature reduction.
        opts.matching_model: type of matching model to use. 'resnet' and
            'fpn' are supported.
        opts.matching_feature_dims: number of features dimensions output
            from the matching encoder.
        opts.model_num_views: number of views to expect in each tuple of
            frames (refernece/current frame + source frames)

        """
        super().__init__()
        self.save_hyperparameters()

        self.run_opts = opts

        # iniitalize the encoder for strong image priors
        if "efficientnet" in self.run_opts.models.image_encoder_name:
            self.encoder = timm.create_model(
                "tf_efficientnetv2_s_in21ft1k", pretrained=True, features_only=True
            )

            self.encoder.num_ch_enc = self.encoder.feature_info.channels()
        else:
            raise ValueError("Unrecognized option for image encoder type!")

        # iniitalize the first half of the U-Net, encoding the cost volume
        # and image prior image feautres
        if self.run_opts.models.cv_encoder_type == "multi_scale_encoder":
            self.cost_volume_net = CVEncoder(
                num_ch_cv=self.run_opts.models.matching_num_depth_bins,
                num_ch_enc=self.encoder.num_ch_enc[self.run_opts.models.matching_scale :],
                num_ch_outs=[64, 128, 256, 384],
            )
            dec_num_input_ch = (
                self.encoder.num_ch_enc[: self.run_opts.models.matching_scale]
                + self.cost_volume_net.num_ch_enc
            )
        else:
            raise ValueError("Unrecognized option for cost volume encoder type!")

        # iniitalize the final depth decoder
        if self.run_opts.models.depth_decoder_name == "unet_pp":
            self.depth_decoder = DepthDecoderPP(dec_num_input_ch)
        else:
            raise ValueError("Unrecognized option for depth decoder name!")

        # plane decoders
        self.planar_mask_decoder = SkipDecoderRegression(dec_num_input_ch, num_output_channels=1)
        self.embeddings_decoder = SkipDecoderRegression(dec_num_input_ch, num_output_channels=3)

        # used for normals loss
        self.compute_normals = NormalGenerator(
            self.run_opts.image_height // 2,
            self.run_opts.image_width // 2,
        )

        # planes losses
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.sigmoid = nn.Sigmoid()

        # what type of cost volume are we using?
        if self.run_opts.models.feature_volume_type == "simple_cost_volume":
            cost_volume_class = CostVolumeManager
        elif self.run_opts.models.feature_volume_type == "mlp_feature_volume":
            cost_volume_class = FeatureVolumeManager
        else:
            raise ValueError(
                f"Unrecognized option {self.run_opts.models.feature_volume_type} "
                f"for feature volume type!"
            )

        self.cost_volume = cost_volume_class(
            matching_height=self.run_opts.image_height
            // (2 ** (self.run_opts.models.matching_scale + 1)),
            matching_width=self.run_opts.image_width
            // (2 ** (self.run_opts.models.matching_scale + 1)),
            num_depth_bins=self.run_opts.models.matching_num_depth_bins,
            matching_dim_size=self.run_opts.models.matching_feature_dims,
            num_source_views=opts.models.model_num_views - 1,
        )

        # init the matching encoder. resnet is fast and is the default for
        # results in the paper, fpn is more accurate but much slower.
        if "resnet" == self.run_opts.models.matching_encoder_type:
            self.matching_model = ResnetMatchingEncoder(
                18, self.run_opts.models.matching_feature_dims
            )
        elif "unet_encoder" == self.run_opts.models.matching_encoder_type:
            self.matching_model = UNetMatchingEncoder()
        else:
            raise ValueError(
                f"Unrecognized option {self.run_opts.models.matching_encoder_type} "
                f"for matching encoder type!"
            )

        self.tensor_formatter = TensorFormatter()

        self.color_aug = CustomColorJitter(0.2, 0.2, 0.2, 0.2)

    def compute_matching_feats(
        self,
        cur_image,
        src_image,
        unbatched_matching_encoder_forward,
    ):
        """
        Computes matching features for the current image (reference) and
        source images.

        Unfortunately on this PyTorch branch we've noticed that the output
        of our ResNet matching encoder is not numerically consistent when
        batching. While this doesn't affect training (the changes are too
        small), it does change and will affect test scores. To combat this
        we disable batching through this module when testing and instead
        loop through images to compute their feautures. This is stable and
        produces exact repeatable results.

        Args:
            cur_image: image tensor of shape B3HW for the reference image.
            src_image: images tensor of shape BM3HW for the source images.
            unbatched_matching_encoder_forward: disable batching and loops
                through iamges to compute feaures.
        Returns:
            matching_cur_feats: tensor of matching features of size bchw for
                the reference current image.
            matching_src_feats: tensor of matching features of size BMcHW
                for the source images.
        """
        if unbatched_matching_encoder_forward:
            all_frames_bm3hw = torch.cat([cur_image.unsqueeze(1), src_image], dim=1)
            batch_size, num_views = all_frames_bm3hw.shape[:2]
            all_frames_B3hw = tensor_bM_to_B(all_frames_bm3hw)
            matching_feats = [self.matching_model(f) for f in all_frames_B3hw.split(1, dim=0)]

            matching_feats = torch.cat(matching_feats, dim=0)
            matching_feats = tensor_B_to_bM(
                matching_feats,
                batch_size=batch_size,
                num_views=num_views,
            )

        else:
            # Compute matching features and batch them to reduce variance from
            # batchnorm when training.
            matching_feats = self.tensor_formatter(
                torch.cat([cur_image.unsqueeze(1), src_image], dim=1),
                apply_func=self.matching_model,
            )

        matching_cur_feats = matching_feats[:, 0]
        matching_src_feats = matching_feats[:, 1:].contiguous()

        return matching_cur_feats, matching_src_feats

    def forward(
        self,
        phase,
        cur_data,
        src_data,
        unbatched_matching_encoder_forward=False,
        return_mask=False,
    ):
        """
        Computes a forward pass through the depth model.

        This function is used for both training and inference.

        During training, a flip horizontal augmentation is used on images
        with a random chance of 50%. If you do plan on changing this flip,
        be careful on where its done. When we do need to flip, we only want
        it to apply through image encoders, but not through the cost volume.
        When we use a flip, we apply it to images before the matching
        encoder, flipping back when we pass those feautres through the cost
        volume, and then flip the cost volume's output so that they align
        with the current image's features (flipped when we use the image
        prior encoder) when we use our final U-Net.

        Args:
            phase: str defining phase of training. When phase is "train,"
                flip augmentation is used on images.
            cur_data: a dictionary with tensors for the current view. These
                include
                    "image_b3hw" for an RGB image,
                    "K_si_b44" intrinsics tensor for projecting points to
                        image space where i starts at 0 and goes up to the
                        maximum divisor scale,
                    "cam_T_world_b44" a camera extrinsics matrix for
                        transforming world points to camera coordinates,
                    and "world_T_cam_b44" a camera pose matrix for
                        transforming camera points to world coordinates.
            src_data: also a dictionary with elements similar to cur_data.
                All tensors here are expected to have batching shape B...
                instead of bM where M is the number of source images.
            unbatched_matching_encoder_forward: disable batching and loops
                through iamges to compute matching feaures, used for stable
                inference when testing. See compute_matching_feats for more
                information.
            return_mask: return a 2D mask from the cost volume for areas
                where there is source view information.
        Returns:
            depth_outputs: a dictionary with outputs including
                "log_depth_pred_s{i}_b1hw" log depths where i is the
                resolution at which this depth map is. 0 represents the
                highest resolution depth map predicted at opts.depth_width,
                opts.depth_height,
                "log_depth_pred_s{i}_b1hw" depth maps in linear scale where
                    is the resolution at which this depth map is. 0
                    represents the highest resolution depth map predicted at
                    opts.depth_width, opts.depth_height,
                "lowest_cost_bhw" the argmax for likelihood along depth
                    planes from the cost volume, representing the best
                    matched depth plane at each spatial resolution,
                and "overall_mask_bhw" returned when return_mask is True and
                    is a 2D mask from the cost volume for areas where there
                    is source view information from the current view's point
                    of view.
            plane_outputs: a dictionary with outputs including
                "plane_mask_pred_s{i}_b1hw" a plane mask prediction where i
                is the resolution at which the plane mask is. 0 represents
                the highest resolution plane mask.
                "embedding_pred_s{i}_b3hw" plane embeddings predicted at the
                i-th resolution.

        """

        # get all tensors from the batch dictioanries.
        # NOTE: we train only the new decoders
        with torch.no_grad():
            cur_feats, depth_outputs, flip = self.run_depth_branch(
                phase, cur_data, src_data, unbatched_matching_encoder_forward, return_mask
            )

        # predict plane output
        plane_mask_outputs = self.planar_mask_decoder(cur_feats)
        embeddings_outputs = self.embeddings_decoder(cur_feats)

        plane_outputs = {}
        for i in range(4):
            # handle flipping
            if flip:
                plane_mask_outputs[f"output_s{i}_b1hw"] = torch.flip(
                    plane_mask_outputs[f"output_s{i}_b1hw"], (-1,)
                )
                embeddings_outputs[f"output_s{i}_b3hw"] = torch.flip(
                    embeddings_outputs[f"output_s{i}_b3hw"], (-1,)
                )

            plane_outputs[f"plane_mask_pred_s{i}_b1hw"] = plane_mask_outputs[f"output_s{i}_b1hw"]
            plane_outputs[f"embedding_pred_s{i}_b3hw"] = embeddings_outputs[f"output_s{i}_b3hw"]

        all_outputs = {**depth_outputs, **plane_outputs}

        return all_outputs

    def run_depth_branch(
        self, phase, cur_data, src_data, unbatched_matching_encoder_forward, return_mask
    ):
        cur_image = cur_data["image_b3hw"]
        src_image = src_data["image_b3hw"]
        src_K = src_data[f"K_s{self.run_opts.models.matching_scale}_b44"]
        cur_invK = cur_data[f"invK_s{self.run_opts.models.matching_scale}_b44"]
        src_cam_T_world = src_data["cam_T_world_b44"]
        src_world_T_cam = src_data["world_T_cam_b44"]

        cur_cam_T_world = cur_data["cam_T_world_b44"]
        cur_world_T_cam = cur_data["world_T_cam_b44"]

        with torch.cuda.amp.autocast(False):
            # Compute src_cam_T_cur_cam, a transformation for going from 3D
            # coords in current view coordinate frame to source view coords
            # coordinate frames.
            src_cam_T_cur_cam = src_cam_T_world @ cur_world_T_cam.unsqueeze(1)

            # Compute cur_cam_T_src_cam the opposite of src_cam_T_cur_cam. From
            # source view to current view.
            cur_cam_T_src_cam = cur_cam_T_world.unsqueeze(1) @ src_world_T_cam

        # flip transformation! Figure out if we're flipping. Should be true if
        # we are training and a coin flip says we should.
        flip_threshold = 0.5 if phase == "train" else 0.0
        flip = torch.rand(1).item() < flip_threshold

        if flip:
            # flip all images.
            cur_image = torch.flip(cur_image, (-1,))
            src_image = torch.flip(src_image, (-1,))

        # Compute image features for the current view. Used for a strong image
        # prior.
        cur_feats = self.encoder(cur_image)

        # Compute matching features
        matching_cur_feats, matching_src_feats = self.compute_matching_feats(
            cur_image, src_image, unbatched_matching_encoder_forward
        )

        if flip:
            # now (carefully) flip matching features back for correct MVS.
            matching_cur_feats = torch.flip(matching_cur_feats, (-1,))
            matching_src_feats = torch.flip(matching_src_feats, (-1,))

        # Get min and max depth to the right shape, device and dtype
        min_depth = (
            torch.tensor(self.run_opts.models.min_matching_depth).type_as(src_K).view(1, 1, 1, 1)
        )
        max_depth = (
            torch.tensor(self.run_opts.models.max_matching_depth).type_as(src_K).view(1, 1, 1, 1)
        )

        # Compute the cost volume. Should be size bdhw.
        cost_volume, lowest_cost, _, overall_mask_bhw = self.cost_volume(
            cur_feats=matching_cur_feats,
            src_feats=matching_src_feats,
            src_extrinsics=src_cam_T_cur_cam,
            src_poses=cur_cam_T_src_cam,
            src_Ks=src_K,
            cur_invK=cur_invK,
            min_depth=min_depth,
            max_depth=max_depth,
            return_mask=return_mask,
        )

        if flip:
            # OK, we've computed the cost volume, now we need to flip the cost
            # volume to have it aligned with flipped image prior features
            cost_volume = torch.flip(cost_volume, (-1,))

        # Encode the cost volume and current image features
        if self.run_opts.models.cv_encoder_type == "multi_scale_encoder":
            cost_volume_features = self.cost_volume_net(
                cost_volume,
                cur_feats[self.run_opts.models.matching_scale :],
            )
            cur_feats = cur_feats[: self.run_opts.models.matching_scale] + cost_volume_features

        # Decode into depth at multiple resolutions.
        depth_outputs = self.depth_decoder(cur_feats)

        # loop through depth outputs, flip them if we need to and get linear
        # scale depths.
        for k in list(depth_outputs.keys()):
            log_depth = depth_outputs[k].float()

            if flip:
                # now flip the depth map back after final prediction
                log_depth = torch.flip(log_depth, (-1,))

            depth_outputs[k] = log_depth
            depth_outputs[k.replace("log_", "")] = torch.exp(log_depth)

        # include argmax likelihood depth estimates from cost volume and
        # overall source view mask.
        depth_outputs["lowest_cost_bhw"] = lowest_cost
        depth_outputs["overall_mask_bhw"] = overall_mask_bhw

        return cur_feats, depth_outputs, flip

    def _compute_plane_losses(
        self, cur_data: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]
    ):
        planar_mask_b1hw = ~cur_data["non_planar_mask"]
        plane_ids_masked_b1hw = cur_data["planes_id_masked"]

        found_scale = False
        seg_loss = 0.0
        embedding_loss = 0.0
        for i in range(4):
            plane_mask_pred_resized_b1hw = F.interpolate(
                outputs[f"plane_mask_pred_s{i}_b1hw"],
                size=planar_mask_b1hw.shape[-2:],
                mode="nearest",
            )
            seg_loss += self.bce_loss(plane_mask_pred_resized_b1hw, planar_mask_b1hw.float()).mean()

            embedding_pred_resized_b3hw = F.interpolate(
                outputs[f"embedding_pred_s{i}_b3hw"],
                size=planar_mask_b1hw.shape[-2:],
                mode="nearest",
            )
            if planar_mask_b1hw.sum():
                for idx in range(len(embedding_pred_resized_b3hw)):
                    embedding_pred_resized_3hw = embedding_pred_resized_b3hw[idx]
                    plane_ids_1hw = plane_ids_masked_b1hw[idx]
                    embedding_loss += (
                        hinge_embedding_loss(embedding_pred_resized_3hw, plane_ids_1hw)
                        / embedding_pred_resized_b3hw.shape[0]
                    )

            found_scale = True

        if not found_scale:
            raise Exception("Could not find a valid scale to compute the loss!")

        loss = seg_loss * 1.0 + embedding_loss * 1.0

        losses = {
            "loss": loss,
            "seg_loss": seg_loss,
            "embedding_loss": embedding_loss,
        }
        return losses

    def compute_losses(
        self,
        cur_data: dict[str, torch.Tensor],
        src_data: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute the final set of losses"""
        planar_losses = self._compute_plane_losses(cur_data=cur_data, outputs=outputs)

        losses = {}
        losses["loss"] = planar_losses["loss"]
        planar_losses.pop("loss")
        losses.update(planar_losses)

        return losses

    def step(self, phase, batch, batch_idx):
        """Takes a training/validation step through the model.

        phase: "train" or "val". "train" will signal this function and
            others log results and use flip augmentation.
        batch: (cur_data, src_data) where cur_data is a dict with data on
            the current (reference) view and src_data is a dict with data on
            source views.
        """
        cur_data, src_data = batch

        if phase == "train":
            cur_data["image_b3hw"] = self.color_aug(cur_data["image_b3hw"], denormalize_first=True)
            for src_ind in range(src_data["image_b3hw"].shape[1]):
                src_data["image_b3hw"][:, src_ind] = self.color_aug(
                    src_data["image_b3hw"][:, src_ind], denormalize_first=True
                )

        # forward pass through the model.
        outputs = self(phase, cur_data, src_data)

        # predicted depth
        depth_pred = outputs["depth_pred_s0_b1hw"]
        depth_pred_lr = outputs["depth_pred_s3_b1hw"]
        cv_min = outputs["lowest_cost_bhw"]

        # gt depth
        depth_gt = cur_data["depth_b1hw"]
        depth_mask = cur_data["mask_b1hw"]
        depth_mask_b = cur_data["mask_b_b1hw"]

        # predicted plane channels
        plane_mask_pred_b1hw = torch.sigmoid(outputs["plane_mask_pred_s0_b1hw"])
        embedding_pred_b3hw = outputs["embedding_pred_s0_b3hw"].detach().cpu().float()

        # gt plane data
        planar_mask_b1hw = ~cur_data["non_planar_mask"]
        plane_ids_gt_b1hw = cur_data["planes_id"]

        # estimate normals for groundtruth
        normals_gt = self.compute_normals(depth_gt, cur_data["invK_s0_b44"])
        cur_data["normals_b3hw"] = normals_gt

        # estimate normals for depth
        normals_pred = self.compute_normals(depth_pred, cur_data["invK_s0_b44"])
        outputs["normals_pred_b3hw"] = normals_pred

        # compute losses
        losses = self.compute_losses(cur_data=cur_data, src_data=src_data, outputs=outputs)

        # is it train?
        is_train = phase == "train"
        batch_size = depth_pred.shape[0]

        # logging and validation
        with torch.inference_mode():
            # log images for train.
            if is_train and self.global_step % self.trainer.log_every_n_steps == 0:
                for i in range(min(4, batch_size)):
                    # log image
                    image_i = reverse_imagenet_normalize(cur_data["image_b3hw"][i])

                    self.logger.experiment.add_image(f"image/{i}", image_i, self.global_step)

                    # log depths
                    mask_i = depth_mask[i].float().cpu()
                    depth_gt_viz_i, vmin, vmax = colormap_image(
                        depth_gt[i].float().cpu(), mask_i, return_vminvmax=True
                    )
                    depth_pred_viz_i = colormap_image(
                        depth_pred[i].float().cpu(), vmin=vmin, vmax=vmax
                    )
                    cv_min_viz_i = colormap_image(
                        cv_min[i].unsqueeze(0).float().cpu(), vmin=vmin, vmax=vmax
                    )
                    depth_pred_lr_viz_i = colormap_image(
                        depth_pred_lr[i].float().cpu(), vmin=vmin, vmax=vmax
                    )

                    self.logger.experiment.add_image(
                        f"depth_gt/{i}", depth_gt_viz_i, self.global_step
                    )
                    self.logger.experiment.add_image(
                        f"depth_pred/{i}", depth_pred_viz_i, self.global_step
                    )
                    self.logger.experiment.add_image(
                        f"depth_pred_lr/{i}", depth_pred_lr_viz_i, self.global_step
                    )
                    self.logger.experiment.add_image(
                        f"normals_gt/{i}", 0.5 * (1 + normals_gt[i]), self.global_step
                    )
                    self.logger.experiment.add_image(
                        f"normals_pred/{i}", 0.5 * (1 + normals_pred[i]), self.global_step
                    )
                    self.logger.experiment.add_image(f"cv_min/{i}", cv_min_viz_i, self.global_step)

                    # log plane embeddings - only works for k = 3!
                    embedding_pred_viz_i = embedding_pred_b3hw[i]
                    mins, maxs = embedding_pred_viz_i.min(), embedding_pred_viz_i.max()
                    embedding_pred_viz_i = (embedding_pred_viz_i - mins) / (maxs - mins + 1e-3)

                    self.logger.experiment.add_image(  # type:ignore
                        f"plane_embedding_pred/{i}", embedding_pred_viz_i, self.global_step
                    )

                    planar_mask_i = planar_mask_b1hw[i].float().cpu()
                    plane_ids_gt_viz_i = colormap_planes(
                        plane_ids_gt_b1hw[i, 0], mask_hw=planar_mask_i[0]
                    )
                    plane_mask_pred_viz_i = colormap_image(
                        plane_mask_pred_b1hw[i].detach().cpu().float(), flip=False
                    )

                    self.logger.experiment.add_image(  # type:ignore
                        f"plane_ids_gt/{i}", plane_ids_gt_viz_i, self.global_step
                    )
                    self.logger.experiment.add_image(  # type:ignore
                        f"plane_mask_pred_viz/{i}", plane_mask_pred_viz_i, self.global_step
                    )

                self.logger.experiment.flush()

            # log losses
            for loss_name, loss_val in losses.items():
                self.log(
                    f"{phase}/{loss_name}",
                    loss_val,
                    sync_dist=True,
                    on_step=is_train,
                    on_epoch=not is_train,
                    prog_bar=loss_name == "loss",
                )

            # high_res_validation: it isn't always wise to load in high
            # resolution depth maps so this is an optional flag.
            if phase == "train" or not self.run_opts.high_res_validation:
                # compute metrics at low res depth resolution for train or if
                # validation isn't set to use high res depth
                depth_metrics = compute_depth_metrics(
                    depth_gt[depth_mask_b], depth_pred[depth_mask_b]
                )
            else:
                # if we are validating or testing, we want to upscale our
                # predictions to full-size and compare against the GT depth map,
                # so that metrics are comparable across resolutions
                full_size_depth_gt = cur_data["full_res_depth_b1hw"]
                full_size_mask_b = cur_data["full_res_mask_b_b1hw"]
                # this should be nearest to reflect test, but keeping it for
                # backwards comparison reasons.
                full_size_pred = F.interpolate(
                    depth_pred,
                    full_size_depth_gt.size()[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                depth_metrics = compute_depth_metrics(
                    full_size_depth_gt[full_size_mask_b],
                    full_size_pred[full_size_mask_b],
                )

            # plane metrics
            plane_metrics = {}
            if phase != "train":
                plane_mask_pred = plane_mask_pred_b1hw.detach().cpu().numpy().squeeze()
                plane_mask_gt = planar_mask_b1hw.cpu().numpy().squeeze()
                iou_plane_mask = compute_iou(pred=plane_mask_pred, gt=plane_mask_gt)
                plane_metrics["planarity_iou"] = iou_plane_mask

            all_metrics = {**depth_metrics, **plane_metrics}

            for metric_name, metric_val in all_metrics.items():
                self.log(
                    f"{phase}_metrics/{metric_name}",
                    metric_val,
                    sync_dist=True,
                    on_step=is_train,
                    on_epoch=not is_train,
                )

        return losses["loss"]

    def training_step(self, batch, batch_idx):
        """Runs a training step."""
        return self.step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        """Runs a validation step."""
        return self.step("val", batch, batch_idx)

    def configure_optimizers(self):
        """Configuring optmizers and learning rate schedules.

        By default we use a stepped learning rate schedule with steps at
        70000 and 80000.

        """

        params = chain(self.embeddings_decoder.parameters(), self.planar_mask_decoder.parameters())

        optimizer = torch.optim.AdamW(
            params, lr=self.run_opts.models.lr, weight_decay=self.run_opts.models.wd
        )

        def lr_lambda(step):
            if step < self.run_opts.models.lr_steps[0]:
                return 1
            elif step < self.run_opts.models.lr_steps[1]:
                return 0.1
            else:
                return 0.01

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }

import math
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models.resnet as resnet
from loguru import logger
from torchvision import transforms

from ..nnutils import net_blocks as nb
from ..utils import geometry_utils
from . import diff_operators, resnetfc


class ResNet(nn.Module):
    def __init__(self, model="resnet18", nlayers=4, pretrained=True):
        super().__init__()
        if model == "resnet18":
            net = resnet.resnet18(pretrained=pretrained)
        elif model == "resnet34":
            net = resnet.resnet34(pretrained=pretrained)
        elif model == "resnet50":
            net = resnet.resnet50(pretrained=pretrained)
        else:
            raise NameError("Unknown Fan Filter setting!")

        self.nlayers = nlayers
        nb.turnBNoff(net)

        self.conv1 = net.conv1
        self.pool = net.maxpool
        self.layer0 = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.avgpool = net.avgpool

    def forward(self, image):
        """
        :param image: [BxC_inxHxW] tensor of input image
        :return: list of [BxC_outxHxW] tensors of output features
        """
        y = image
        feat_pyramid = []
        y = self.layer0(y)
        feat_pyramid.append(y)
        if self.nlayers > 0:
            y = self.layer1(self.pool(y))
            feat_pyramid.append(y)
        if self.nlayers > 1:
            y = self.layer2(y)
            feat_pyramid.append(y)
        if self.nlayers > 2:
            y = self.layer3(y)
            feat_pyramid.append(y)
        if self.nlayers > 3:
            y = self.layer4(y)
            feat_pyramid.append(y)

        gfeat = self.avgpool.forward(y)
        gfeat = gfeat.squeeze(3).squeeze(2)
        return feat_pyramid, gfeat


def build_backbone(model="resnet18", nlayers=4, pretrained=True):
    resnet = ResNet(model, nlayers=nlayers, pretrained=pretrained)
    if model == "resnet18":
        return resnet, 1024
    elif model == "resnet34":
        return resnet, 1024
    else:
        assert False, "Model backone not available"


def encoder_3d(nlayers, nc_input, nc_l1, nc_max):
    modules = []
    nc_output = nc_l1
    nc_step = 1
    for nl in range(nlayers):
        if (nl >= 1) and (nl % nc_step == 0) and (nc_output <= nc_max * 2):
            nc_output *= 2

        modules.append(nb.conv3d(False, nc_input, nc_output, stride=1))
        nc_input = nc_output
        # modules.append(nb.conv3d(False, nc_input, nc_output, stride=1))
        # modules.append(torch.nn.MaxPool3d(kernel_size=2, stride=2))

    encoder = nn.Sequential(*modules)
    return encoder, nc_output


class SpatialEncoder(nn.Module):
    def __init__(self, opts, pretrained=True):
        super().__init__()
        self.image_encoder, feat_dim = build_backbone(
            model="resnet34", pretrained=pretrained
        )
        self.latent_scaling = torch.FloatTensor([0, 0])
        self.d_out = feat_dim

    def forward(self, image):
        latents, _ = self.image_encoder.forward(image)
        latent_sz = latents[0].shape[-2:]
        for i in range(len(latents)):
            latents[i] = F.interpolate(
                latents[i],
                latent_sz,
                mode="bilinear",
                align_corners=True,
            )
        self.latent = torch.cat(latents, dim=1)
        return self.latent


# def last_op_

# def last_op_tanh(x, max_value):
#     return F.tanh(x) * max_value


class ScaledTanh(nn.Module):
    def __init__(self, max_value):
        super().__init__()
        self.max_value = max_value

    def forward(self, x):
        return self.max_value * torch.tanh(x)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class VolSDF(nn.Module):
    def __init__(self, opts):
        super().__init__()
        if opts.MODEL.DECODER == "pixNerf":
            self.image_encoder = SpatialEncoder(opts)
        else:
            raise NotImplementedError("Unknown decoder!")

        if opts.MODEL.DEPTH_INPUT:
            self.depth_image_encoder = SpatialEncoder(opts, pretrained=False)

        self.opts = opts

        self.point_emb = resnetfc.PositionalEncoding()
        self.dir_emb = resnetfc.PositionalEncoding()
        self.last_op = None
        if opts.MODEL.ALLOW_CLIPPING:
            self.last_op = Identity()
            if opts.MODEL.CLIP_ACTIVATION == "tanh":
                self.last_op = ScaledTanh(max_value=1)
        else:
            self.last_op = ScaledTanh(max_value=8)
        if opts.MODEL.MLP_ACTIVATION == "leakyReLU":
            activation = nn.LeakyReLU(0.2)
        else:
            activation = None
        self.project_pos_emb = False
        point_emb_d_out = self.point_emb.d_out
        dir_emb_d_out = self.dir_emb.d_out
        if opts.MODEL.PROJECT_POS_EMB > 0:
            self.project_pos_emb = True
            self.point_emb_proj_layer = resnetfc.ResnetFC(
                d_in=self.point_emb.d_out,
                n_blocks=3,
                d_out=opts.MODEL.PROJECT_POS_EMB,
                activation=activation,
                use_batch_norm=opts.MODEL.MLP_BATCH_NORM,
            )
            point_emb_d_out = opts.MODEL.PROJECT_POS_EMB
            self.dir_emb_proj_layer = resnetfc.ResnetFC(
                d_in=self.point_emb.d_out,
                n_blocks=3,
                d_out=opts.MODEL.PROJECT_POS_EMB,
                activation=activation,
                use_batch_norm=opts.MODEL.MLP_BATCH_NORM,
            )
            dir_emb_d_out = opts.MODEL.PROJECT_POS_EMB

        mlp_input_size = point_emb_d_out + dir_emb_d_out + self.image_encoder.d_out

        if opts.MODEL.DEPTH_INPUT:
            mlp_input_size += self.depth_image_encoder.d_out

        self.surface_classifier = resnetfc.ResnetFC(
            d_in=mlp_input_size,
            d_out=1,
            last_op=self.last_op,
            activation=activation,
            use_batch_norm=opts.MODEL.MLP_BATCH_NORM,
        )
        self.surface_classifier = self.surface_classifier
        self.projection = geometry_utils.prespective_transform
        self.img_size = (opts.DATALOADER.IMG_SIZE, opts.DATALOADER.IMG_SIZE)
        self.loss_fun = torch.nn.functional.l1_loss

        self.init_transforms()
        return

    def init_transforms(
        self,
    ):
        base_transform = torch.nn.Sequential(
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )
        self.base_transform = torch.jit.script(base_transform)
        return

    def transform(self, images):
        images = self.base_transform(images)
        return images

    def filter_images(self, images, depths=None):
        _, _, image_height, image_width = images.shape
        images = self.transform(images)
        self.feats = self.image_encoder(images)
        if self.opts.MODEL.DEPTH_INPUT and depths is not None:
            depths = depths[:, None, :, :]
            depths = depths.expand(-1, 3, -1, -1)
            self.depth_feats = self.depth_image_encoder(depths)
            self.feats = torch.cat([self.feats, self.depth_feats], axis=1)
        else:
            self.depth_feats = None
        return self.feats

    def convert_points_to_ndc(self, points, RT, kNDC):
        points = geometry_utils.transform_points(points, RT)
        xyz_ndc = geometry_utils.convert_to_ndc(points, kNDC, self.projection)
        return points, xyz_ndc

    def get_depth_points(self, points, ray_dir, RT, kNDC):
        points, xyz_ndc = self.convert_points_to_ndc(points, RT, kNDC)
        depth = points[:, 2:3, :]
        return depth, xyz_ndc

    def query(self, defaults=None, points=None, ray_dir=None, kNDC=None, RT=None):
        opts = self.opts
        # project 3d points to image plane
        image_width, image_height = self.img_size[0], self.img_size[1]
        assert kNDC is not None, "please supply a  calibndc"

        if RT is None:  ## transform using extrinsics
            assert False, "where is the RT matrix"

        self.points_cam, self.xyz_ndc = self.convert_points_to_ndc(points, RT, kNDC)

        xyz_valids = (self.xyz_ndc > -1.0) * (self.xyz_ndc < 1.0)
        xyz_valids = xyz_valids[:, 0, :] * xyz_valids[:, 1, :]
        # logger.info('xyz_valids {}'.format(xyz_valids.float().mean()))
        # if xyz_valids.float().mean() < 1.0:
        self.project_points = (
            self.xyz_ndc[
                :,
                0:2,
            ]
            * 0
        )
        self.project_points[:, 0, :] = (
            (
                self.xyz_ndc[
                    :,
                    0,
                ]
                + 1
            )
            * image_width
            / 2
        )
        self.project_points[:, 1, :] = (
            (
                self.xyz_ndc[
                    :,
                    1,
                ]
                + 1
            )
            * image_height
            / 2
        )

        self.valid_points = xyz_valids
        xyz_ndc = self.xyz_ndc.permute(0, 2, 1).contiguous()
        feat_points = F.grid_sample(
            self.feats, xyz_ndc[:, :, None, 0:2], align_corners=True, mode="bilinear"
        )

        # feat_points = grid_sample.grid_sample_2d(
        #     self.feats,
        #     self.xyz_ndc.permute(0, 2, 1)[:, :, None, 0:2],
        # )
        feat_points = feat_points.squeeze(3)
        B, feat_z, npoints = feat_points.shape
        feat_points = feat_points.permute(0, 2, 1).contiguous()
        feat_points = feat_points.reshape(B * npoints, feat_z)

        xyz_feat = xyz_ndc * 1
        xyz_feat = xyz_feat.reshape(-1, 3)
        xyz_feat = self.point_emb(xyz_feat)

        ray_dir = ray_dir.permute(0, 2, 1).contiguous()

        if not opts.MODEL.DIR_ENCODING:
            ray_dir_feat = xyz_feat * 0  ## zero out the direction encoding
        else:
            ray_dir_feat = self.dir_emb(ray_dir.reshape(B * npoints, 3))

        if self.project_pos_emb:  ## project the pos_emb to higher dimensions.
            xyz_feat = self.point_emb_proj_layer(xyz_feat)
            if not opts.MODEL.DIR_ENCODING:
                ray_dir_feat = xyz_feat * 0
            else:
                ray_dir_feat = 0 * self.dir_emb_proj_layer(ray_dir_feat)

        if not opts.MODEL.USE_POINT_FEATURES:
            xyz_feat = xyz_feat * 0  ## zero out the point encoding

        point_concat_feat = torch.cat([feat_points, xyz_feat, ray_dir_feat], dim=1)

        self.surface_prediction = self.get_surface_prediction(
            point_concat_feat, B, npoints
        )
        point_concat_detach = torch.cat(
            [feat_points.data, xyz_feat, ray_dir_feat], dim=1
        )
        self.surface_pred_detached = self.get_surface_prediction(
            point_concat_detach, B, npoints
        )
        return self.surface_prediction, self.surface_pred_detached

    def get_surface_prediction(
        self,
        point_feat,
        batch_size,
        npoints,
    ):
        surface_prediction = self.surface_classifier.forward(point_feat)
        surface_prediction = surface_prediction.reshape(
            batch_size, npoints, 1
        )  ## B x npoints x 1
        surface_prediction = surface_prediction.permute(0, 2, 1)  ## B x 1 x npoints
        return surface_prediction

    def get_preds(self, gt=False):
        opts = self.opts
        if gt:
            raise Exception("Not implemented")
        return self.surface_prediction

    def get_error(self, **kwargs):
        opts = self.opts
        surfacePred = self.surface_prediction
        surfacePredDetached = self.surface_prediction_detached
        valid_points = self.valid_points
        gt_dist = self.penalty_regions.unsqueeze(1)
        penalty_types = self.penalty_types.unsqueeze(1)
        grad_penalty_masks = self.grad_penalty_masks.unsqueeze(1)
        points = self.points
        loss, tracking_scalars = self.forward_loss(
            points,
            gt_targets=gt_dist,
            target_types=penalty_types,
            pred_targets=surfacePred,
            pred_targets_detach=surfacePredDetached,
            grad_penalty_mask=grad_penalty_masks,
            validity=valid_points,
            **kwargs
        )
        return loss, tracking_scalars

    def forward_loss(
        self,
        points,
        gt_targets,
        target_types,
        pred_targets,
        pred_targets_detach,
        grad_penalty_mask,
        validity=None,
        use_grad_penalty=True,
        ray_dir=None,
    ):
        opts = self.opts
        device = points.device

        use_clipping = opts.MODEL.ALLOW_CLIPPING
        clipping_meta = {
            "allow": use_clipping,
            "threshold": opts.MODEL.CLIPPED_DISTANCE_VALUE,
        }
        ii_loss = self.loss_II(pred_targets, gt_targets, clipping_meta)
        oo_loss = self.loss_OO(
            pred_targets=pred_targets,
            gt_targets=gt_targets,
            clipping_meta=clipping_meta,
        )
        io_loss = self.loss_IO(
            pred_targets=pred_targets,
            gt_targets=gt_targets,
            clipping_meta=clipping_meta,
        )
        oi_loss = self.loss_OI(
            pred_targets=pred_targets,
            gt_targets=gt_targets,
            clipping_meta=clipping_meta,
        )
        oi_start_loss = self.loss_OI_start(
            pred_targets=pred_targets,
            gt_targets=gt_targets,
            clipping_meta=clipping_meta,
        )
        entropy_loss = self.loss_entropy(
            pred_targets=pred_targets,
            gt_targets=gt_targets,
            target_types=target_types,
            all_occ_pts=opts.MODEL.ENTROPY_ALL_OCCCLUDED_PTS,
            use_std=opts.MODEL.ENTROPY_TEMP_STD,
        )

        ii_masked_loss, ii_elements = apply_masked_loss(ii_loss, target_types == 1)
        oi_start_masked_loss, oi_start_elements = apply_masked_loss(
            oi_start_loss, target_types == 5
        )
        oo_masked_loss, oo_elements = apply_masked_loss(oo_loss, target_types == 2)
        io_masked_loss, io_elements = apply_masked_loss(io_loss, target_types == 3)
        oi_masked_loss, oi_elements = apply_masked_loss(oi_loss, target_types == 4)

        # segment_loss = ii_loss * (target_types == 1) + \
        #        oo_loss * (target_types == 2) + \
        #        io_loss * (target_types == 3) + \
        #        oi_loss * (target_types == 4) + \
        #        oi_loss_start * (target_types == 5)
        # loss_lst = []
        # batch_size = len(points)
        # for bx in range(batch_size):
        #     validity = (target_types[bx] > 0)
        #     if validity.sum() == 0:
        #         loss_lst.append(torch.FloatTensor([0.]).to(device).mean())
        #     else:
        #         loss_lst.append(segment_loss[bx][validity].mean())
        # segment_loss = torch.stack(loss_lst)
        losses = {}
        losses["ii"] = torch.stack(ii_masked_loss)
        losses["oi_start"] = torch.stack(oi_start_masked_loss)
        losses["oo"] = torch.stack(oo_masked_loss)
        losses["io"] = torch.stack(io_masked_loss)
        losses["oi"] = torch.stack(oi_masked_loss)

        if opts.MODEL.USE_ENTROPY_LOSS:
            losses["entropy"] = entropy_loss
        else:
            losses["entropy"] = torch.zeros(1).cuda()
        segment_loss = (
            losses["ii"]
            + losses["oi_start"]
            + losses["oo"]
            + losses["io"]
            + losses["oi"]
        )

        tracked_scalars = {}
        tracked_scalars["ii"] = (target_types == 1).float().mean()
        tracked_scalars["oo"] = (target_types == 2).float().mean()
        tracked_scalars["io"] = (target_types == 3).float().mean()
        tracked_scalars["oi"] = (target_types == 4).float().mean()
        tracked_scalars["oi_start"] = (target_types == 5).float().mean()

        total_per = 0
        if True:
            for key in tracked_scalars.keys():
                tracked_scalars[key] = tracked_scalars[key].item()
                if not (tracked_scalars[key] == tracked_scalars[key]):
                    tracked_scalars[key] = 0
                total_per += tracked_scalars[key]

        for key in tracked_scalars.keys():
            losses[key] = losses[key] * tracked_scalars[key] / (total_per + 1e-4)

        is_grad_loss_valid = (grad_penalty_mask.squeeze(1) > 0).sum() > 100

        if is_grad_loss_valid and use_grad_penalty:
            assert (
                ray_dir is not None
            ), "ray dir is None, cannot implemented grad penalty"
            # breakpoint()  # check if the ray dir is accurate.
            # "verfication needed here -- "
            if opts.MODEL.DF_GRAD_PENALTY or opts.MODEL.JJ_GRAD_PENALTY:
                if opts.MODEL.DF_GRAD_PENALTY:
                    grad_loss = self.gradient_penalty_df(
                        points, pred_targets_detach, ray_dir=ray_dir
                    )
                elif opts.MODEL.JJ_GRAD_PENALTY:
                    grad_loss = self.gradient_penalty_jj(
                        points, pred_targets_detach, ray_dir=ray_dir
                    )
                else:
                    assert False, "grad function not defined"

                if opts.MODEL.GRAD_PENALTY_UNSATURATED_ONLY:
                    clip_threshold = opts.MODEL.CLIPPED_DISTANCE_VALUE
                    grad_norm = (grad_penalty_mask.squeeze(1) > 0) * 1
                    saturated_regions = (
                        pred_targets_detach.data < -clip_threshold + 1e-3
                    ) + (pred_targets_detach.data > clip_threshold - 1e-3)
                    unsaturated_regions = torch.logical_not(saturated_regions)
                    grad_loss = grad_loss * unsaturated_regions
                    temp = []
                    batch_size = len(grad_loss)
                    for b in range(batch_size):
                        temp_g_loss = grad_loss[b][unsaturated_regions[b]]
                        if len(temp_g_loss) > 0:
                            temp_g_loss = temp_g_loss.mean()
                        else:
                            temp_g_loss = torch.zeros((1), device=device).float()
                        temp.append(temp_g_loss)
                    # print(temp)
                    grad_loss = torch.stack(temp)
                else:
                    grad_loss = grad_loss * (grad_penalty_mask.squeeze(1) > 0)
                    grad_loss = grad_loss.mean(1)
            else:
                grad_norm = self.gradient_penalty(
                    points, pred_targets_detach, ray_dir=ray_dir
                )
                grad_loss = torch.abs(1 - grad_norm) * (
                    grad_penalty_mask.squeeze(1) > 0
                )
                grad_loss = grad_loss.mean(1)
            losses["grad"] = grad_loss
        else:
            grad_loss = losses["ii"] * 0
            losses["grad"] = grad_loss

        # if not (grad_loss.sum().item() == grad_loss.sum().item()):
        if torch.isnan(grad_loss).sum().item() > 0:
            pdb.set_trace()

        if not (segment_loss.sum().item() == segment_loss.sum().item()):
            pdb.set_trace()
        return losses, tracked_scalars

    @staticmethod
    def gradient_penalty(points, pred_targets, ray_dir):
        pred_normal = diff_operators.gradient(pred_targets.squeeze(1), points)
        dot_p = (pred_normal * ray_dir).sum(1)[
            :,
            None,
        ]
        projection = ray_dir * dot_p
        grad_norm = torch.norm(projection, dim=1)
        return grad_norm

    @staticmethod
    def gradient_penalty_df(points, pred_targets, ray_dir):
        pred_normal = diff_operators.gradient(pred_targets.squeeze(1), points)
        dot_p = (pred_normal * ray_dir).sum(1)[
            :,
            None,
        ]
        # projection = ray_dir * dot_p
        # grad_norm = torch.norm(projection, dim=1)
        penalty = 2 * (((dot_p + 1) / 0.25) ** 2) / (((dot_p + 1) / 0.25) ** 2 + 4)
        return penalty

    @staticmethod
    def gradient_penalty_jj(points, pred_targets, ray_dir):
        pred_normal = diff_operators.gradient(pred_targets.squeeze(1), points)
        dot_p = (pred_normal * ray_dir).sum(1)[
            :,
            None,
        ]
        penalty1 = 2 * (((dot_p + 1) / 0.25) ** 2) / (((dot_p + 1) / 0.25) ** 2 + 4)
        penalty2 = 2 * (1 / (1 + torch.exp(2 * (dot_p - 5))))
        penalty = torch.min(penalty1, penalty2)
        return penalty

    @staticmethod
    def loss_entropy(
        pred_targets,
        gt_targets,
        target_types,
        detached_var=True,
        all_occ_pts=False,
        use_std=False,
    ):
        if all_occ_pts:
            admissible_preds = pred_targets[(target_types <= 4)]
        else:
            admissible_preds = pred_targets[(target_types <= 4) * (target_types > 0)]
        if detached_var:
            if use_std:
                var = torch.std(admissible_preds.detach())
            else:
                var = torch.var(admissible_preds.detach())
        else:
            if use_std:
                var = torch.std(admissible_preds.detach())
            else:
                var = torch.var(admissible_preds)
        temp = var + 1e-3
        sigmod_output = VolSDF.tempsigmoid(temp, x=admissible_preds)
        N = len(sigmod_output)
        x = sigmod_output.sum() / N
        entropy = -1 * VolSDF.entropy_value(x)
        return entropy

    @staticmethod
    def entropy_value(x):
        return -(x) * torch.log(x + 1e-4) - (1 - x) * torch.log(1 - x + 1e-4)

    @staticmethod
    def tempsigmoid(temp, x):
        return torch.sigmoid(x / (temp))

    @staticmethod
    def loss_II(pred_targets, gt_targets, clipping_meta):
        above = gt_targets[..., 1]
        below = gt_targets[..., 0]
        sCloser = torch.abs(below) < torch.abs(above)
        if clipping_meta["allow"]:
            threshold = clipping_meta["threshold"]
            min_th, max_th = -threshold, +threshold
            below = torch.clamp(below, min=min_th)
            above = torch.clamp(above, max=max_th)
        i1_loss = F.l1_loss(pred_targets, below, reduce=False)
        i2_loss = F.l1_loss(pred_targets, above, reduce=False)
        loss_ii = i1_loss * (sCloser == True) + i2_loss * (sCloser == False)
        # loss_ii = loss_ii.squeeze(1).mean(1)
        return loss_ii

    @staticmethod
    def loss_OO(pred_targets, gt_targets, clipping_meta):
        above = gt_targets[..., 1]
        below = gt_targets[..., 0]
        if clipping_meta["allow"]:
            threshold = clipping_meta["threshold"]
            min_th, max_th = -threshold, +threshold
            below = torch.clamp(below, min=min_th)
            above = torch.clamp(above, max=max_th)

        mid = (below + above) / 2  ## ((s -z) + (e - z) )/2
        halfWidth = (above - below) / 2  ##  ((s-z) - (e-z))/2
        loss_oo = F.relu(halfWidth - torch.abs(pred_targets - mid))
        # loss_oo = loss_oo.squeeze(1).mean(1)
        return loss_oo

    @staticmethod
    def loss_IO(pred_targets, gt_targets, clipping_meta):
        above = gt_targets[..., 1]
        below = gt_targets[..., 0]
        sCloser = torch.abs(below) < torch.abs(above)
        if clipping_meta["allow"]:
            threshold = clipping_meta["threshold"]
            min_th, max_th = -threshold, +threshold
            below = torch.clamp(below, min=min_th)
            above = torch.clamp(above, max=max_th)
        i1_loss = F.l1_loss(pred_targets, below, reduce=False)
        o2_loss = torch.min(
            F.relu(above - pred_targets), F.l1_loss(pred_targets, below, reduce=False)
        )
        loss_io = i1_loss * (sCloser == True) + o2_loss * (sCloser == False)
        # loss_io = loss_io.squeeze(1).mean(1)
        return loss_io

    @staticmethod
    def loss_OI(pred_targets, gt_targets, clipping_meta):
        above = gt_targets[..., 1]
        below = gt_targets[..., 0]
        sCloser = torch.abs(below) < torch.abs(above)
        if clipping_meta["allow"]:
            threshold = clipping_meta["threshold"]
            min_th, max_th = -threshold, +threshold
            below = torch.clamp(below, min=min_th)
            above = torch.clamp(above, max=max_th)
        i2_loss = F.l1_loss(pred_targets, above, reduce=False)
        o1_loss = torch.min(
            F.relu(pred_targets - below), F.l1_loss(pred_targets, above, reduce=False)
        )
        loss_oi = o1_loss * (sCloser == True) + i2_loss * (sCloser == False)
        # loss_oi = loss_oi.squeeze(1).mean(1)
        return loss_oi

    @staticmethod
    def loss_OI_start(pred_targets, gt_targets, clipping_meta):
        above = gt_targets[..., 1]
        if clipping_meta["allow"]:
            threshold = clipping_meta["threshold"]
            min_th, max_th = -threshold, +threshold
            above = torch.clamp(above, max=max_th)
        i2_loss = F.l1_loss(pred_targets, above, reduce=False)
        # i2_loss = i2_loss.squeeze(1).mean(1)
        return i2_loss

    def predict(
        self,
        images,
    ):
        self.filter_images(
            images,
        )
        return

    def forward(
        self,
        images,
        points,
        ray_dir,
        kNDC,
        RT,
        penalty_regions,
        penalty_types,
        grad_penalty_masks,
        use_grad_penalty=True,
        depths=None,
    ):
        opts = self.opts
        self.filter_images(images, depths=depths)
        self.penalty_regions = penalty_regions
        self.penalty_types = penalty_types
        self.grad_penalty_masks = grad_penalty_masks
        self.points = points
        self.surface_prediction, self.surface_prediction_detached = self.query(
            points=points, ray_dir=ray_dir, kNDC=kNDC, RT=RT
        )
        predictions = {}
        predictions["surface"] = self.surface_prediction
        predictions["valid_points"] = self.valid_points
        predictions["project_points"] = self.project_points
        predictions["xyz_ndc"] = self.xyz_ndc
        predictions["points_cam"] = self.points_cam
        # predictions['depth'] = self.depth
        # predictions = self.surface_prediction
        losses = {}
        error, tracked_scalars = self.get_error(
            use_grad_penalty=use_grad_penalty, ray_dir=ray_dir
        )

        if type(error) is dict:
            losses.update(error)
        else:
            losses["surface"] = error

        predictions["tracked_scalars"] = tracked_scalars
        return predictions, losses

    def forward_drdf_loss(self, predictions, gt_targets):
        opts = self.opts
        if opts.MODEL.ALLOW_CLIPPING:
            clamp_threshold = opts.MODEL.CLIPPED_DISTANCE_VALUE
            gt_targets = torch.clamp(
                gt_targets,
                min=-1 * clamp_threshold,
                max=clamp_threshold,
            )

        if opts.MODEL.APPLY_LOG_TRANSFORM:
            gt_targets = geometry_utils.apply_log_transform(gt_targets)
            predictions = geometry_utils.apply_log_transform(predictions)
            loss_distance = torch.nn.functional.l1_loss(
                predictions, gt_targets, reduce=False
            ).squeeze()
        else:
            loss_distance = torch.nn.functional.l1_loss(
                predictions, gt_targets, reduce=False
            ).squeeze()
        losses = {"surface": loss_distance.mean(1)}
        return losses

    def forward_drdf(self, images, points, ray_dir, kNDC, RT, gt_targets):
        opts = self.opts
        self.filter_images(images)

        self.surface_prediction, self.surface_prediction_detached = self.query(
            points=points, ray_dir=ray_dir, kNDC=kNDC, RT=RT
        )

        losses = self.forward_drdf_loss(self.surface_prediction, gt_targets[:, None, :])
        predictions = {}
        predictions["surface"] = self.surface_prediction
        predictions["valid_points"] = self.valid_points
        predictions["project_points"] = self.project_points
        predictions["xyz_ndc"] = self.xyz_ndc
        predictions["points_cam"] = self.points_cam
        return predictions, losses


def apply_masked_loss(loss, mask):
    ## scale it with number of items in the mask.
    masked_losses = []
    device = loss.device
    for l, m in zip(loss, mask):
        l = torch.masked_select(l, m)
        if len(l) == 0:
            masked_losses.append(torch.FloatTensor([0]).to(device))
        else:
            masked_losses.append(l)

    masked_losses_avg = [l.mean() for l in masked_losses]
    return masked_losses_avg, masked_losses

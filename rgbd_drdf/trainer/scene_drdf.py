import copy
import os.path as osp
import pdb

import imageio
import numpy as np
import ray
import torch
import torch.nn as nn

from ..data_utils import matterport3d_data
from ..html_utils import visdom_visualizer as visdom_utils
from ..nnutils import drdf_model
from ..renderer import render_utils
from ..utils import intersection_finder_utils, utils_3d
from ..utils.scene_visuals import SceneVisuals
from . import base_trainer


class SceneDRDFTrainer(base_trainer.BaseTrainer, SceneVisuals):
    def __init__(self, opts):
        super().__init__(opts)

    def init_model(
        self,
    ):
        opts = self.opts
        model = drdf_model.DRDFModel(opts)
        self.model = model.to(self.device)

        self.visdom_logger = visdom_utils.Visualizer(opts)
        self.render_func = eval(opts.RENDERER.RENDER_FUNC)

        if opts.TRAIN.NUM_PRETRAIN_EPOCHS > 0:
            self.load_network(
                self.model,
                network_label="model",
                epoch_label=f"epoch_{opts.TRAIN.NUM_PRETRAIN_EPOCHS}",
            )
        self.intersection_finder = intersection_finder_utils.intersection_finder_drdf
        return

    def init_dataset(
        self,
    ):
        opts = self.opts
        if opts.DATALOADER.DATASET_TYPE == "matterport":
            self.dataloader = matterport3d_data.matterport_dataloader(
                opts, shuffle=True
            )
            opts_val = copy.deepcopy(opts)
            opts_val.DATALOADER.SPLIT = "val"
            opts_val.DATALOADER.MODE = "train"
            self.val_dataloader = matterport3d_data.matterport_dataloader(
                opts_val, shuffle=True
            )
        else:
            assert False, "dataset not available"

        return

    def init_optimizer(
        self,
    ):
        opts = self.opts
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=opts.OPTIM.LEARNING_RATE,
            betas=(opts.OPTIM.BETA1, opts.OPTIM.BETA2),
        )
        len_dataloader = len(self.dataloader)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=opts.OPTIM.LEARNING_RATE,
            epochs=opts.TRAIN.NUM_EPOCHS,
            steps_per_epoch=len_dataloader,
            cycle_momentum=False,
            pct_start=0.005,
        )
        return

    def set_input(self, batch):

        if batch["empty"] == True:
            self.invalid_batch = True
            return

        opts = self.opts
        input_imgs = batch["image"]
        batch_size = input_imgs.size(0)
        self.input_imgs = input_imgs.to(self.device)
        self.RT = batch["RT"].to(self.device)
        self.points = batch["points"][:, 0:3, :].to(self.device)
        self.ray_dirs = batch["points"][:, 3:, :].to(self.device)
        self.Kndc = batch["kNDC"].to(self.device)
        self.valid_intersect = batch["valid_intersect"].to(self.device)
        self.index = batch["index"]
        self.extents = batch["extents"]
        # self.camera_params = batch['camera_params']
        # self.depths = batch['depth']

        # self.validity_mask = batch['validity_mask'].to(self.device
        #                                                ).unsqueeze(1).float()

        if False:
            self.meshes = batch["mesh"]
            self.mp3d_int = batch["mp3d_int"].numpy()
            self.mp3d_img_size = batch["mp3d_img_size"].numpy()
            self.mp3d_2d_bbox = batch["mp3d_2d_bbox"].numpy()

        distance = batch["ray_dist"].to(self.device)
        self.distance = distance

        return

    def forward(
        self,
    ):

        predictions, losses = self.model.forward_drdf(
            images=self.input_imgs,
            points=self.points,
            ray_dir=self.ray_dirs,
            kNDC=self.Kndc,
            RT=self.RT,
            gt_targets=self.distance,
        )

        self.loss_factors = {}
        total_loss = 0
        self.losses = losses
        weight_dict = {}
        weight_dict["surface"] = 1.0
        self.weight_dict = weight_dict
        for key in losses.keys():
            if key not in weight_dict.keys():
                self.loss_factors[key] = losses[key].mean()
            else:
                self.loss_factors[key] = weight_dict[key] * losses[key].mean()
                total_loss += self.loss_factors[key]

        if not (total_loss.item() == total_loss.item()):
            pdb.set_trace()
        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            predictions["xyz_ndc"] = self.model.module.xyz_ndc
        else:
            predictions["xyz_ndc"] = self.model.xyz_ndc

        for k in self.smoothed_factor_losses.keys():
            if k in self.loss_factors.keys():
                self.smoothed_factor_losses[k] = (
                    0.99 * self.smoothed_factor_losses[k]
                    + 0.01 * self.loss_factors[k].item()
                )

        self.smoothed_total_loss = (
            self.smoothed_total_loss * 0.99 + 0.01 * total_loss.item()
        )
        self.predictions = predictions
        return total_loss

    def define_criterion(self):
        opts = self.opts
        self.smoothed_factor_losses = {
            "surface": 0.0,
        }
        return

    def val(self, num_batches=5):
        opts = self.opts
        val_losses = []
        for i, batch in enumerate(self.val_dataloader):
            self.set_input(batch)
            self.total_loss = self.forward()
            val_loss = self.total_loss.item()
            val_losses.append(val_loss)
            if i >= num_batches and num_batches > 0:
                break
        val_loss = np.mean(np.array(val_losses))
        return val_loss

    def backward(self, total_loss):
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.total_loss = total_loss.item()
        self.optimizer.step()
        self.scheduler.step()
        return

    def get_current_scalars(
        self,
    ):
        time_per_batch = self.time_per_batch / self.epoch_iter
        lr = self.scheduler.get_last_lr()[0]
        loss_dict = {
            "total_loss": self.smoothed_total_loss,
            "iter_frac": self.real_iter * 1.0 / self.total_steps,
            "valid_points": self.predictions["valid_points"].float().mean().item(),
            "lr": lr,
            "val_loss": self.val_loss,
            "time_per_batch": time_per_batch,
        }

        for k in self.smoothed_factor_losses.keys():
            loss_dict["loss_" + k] = self.smoothed_factor_losses[k]
        return loss_dict

    def log_step(self, total_steps, epoch, epoch_iter):
        opts = self.opts
        scalars = self.get_current_scalars()
        self.visdom_logger.print_current_scalars(epoch, epoch_iter, scalars)
        if opts.LOGGING.PLOT_SCALARS:
            for key, value in scalars.items():
                self.tensorboard_writer.add_scalar(f"train/{key}", value, total_steps)
        return

    def visuals_to_save(self, visual_count):
        batch_visuals = []
        opts = self.opts
        img_size = np.array([opts.DATALOADER.IMG_SIZE, opts.DATALOADER.IMG_SIZE])
        predictions = self.predictions
        project_points = predictions["project_points"].data.cpu().numpy()
        # project_points_depth = predictions['xyz_ndc'].data.cpu().numpy()[:,2,:]
        project_points_depth = predictions["points_cam"].data.cpu().numpy()[:, 2, :]
        # project_points = utils_3d.convert_ndc_to_image(project_points, img_size)
        valid_points = predictions["valid_points"]
        for bx in range(visual_count):
            visuals = {}
            visuals["image"] = (
                self.input_imgs[bx].detach().cpu().numpy().transpose(1, 2, 0)
            )
            visuals["points_proj"] = utils_3d.render_mesh_cv2(
                project_points[bx], project_points_depth[bx], img_size, color_depth=True
            )
            visuals["depth_pred"] = SceneVisuals.colored_depthmap(
                self.generate_depth(bx, catch_exceptions=True),
                d_min=0.0,
                d_max=10.0,
            )
            batch_visuals.append(visuals)
        return batch_visuals

    def get_visuals(self, visual_count):
        visuals = self.visuals_to_save(visual_count)
        return visuals

    def log_visuals(self, total_steps):
        opts = self.opts
        visuals = self.get_visuals(visual_count=1)
        # self.visdom_logger.display_current_results(visuals[0], total_steps)
        prefix = "train"
        for key in visuals[0].keys():
            self.tensorboard_writer.add_image(
                f"{prefix}/{key}", visuals[0][key], total_steps, dataformats="HWC"
            )
        return


if __name__ == "__main__":
    from ..config import defaults
    from ..utils import parse_args

    cmd_args = parse_args.parse_args()
    cfg = defaults.get_cfg_defaults()
    if cmd_args.cfg_file is not None:
        cfg.merge_from_file(cmd_args.cfg_file)
    if cmd_args.set_cfgs is not None:
        cfg.merge_from_list(cmd_args.set_cfgs)

    cfg.RESULT_DIR = osp.join(cfg.RESULT_DIR, f"{cfg.NAME}")
    ray.init(local_mode=False, num_cpus=4)
    trainer = SceneTrainer(cfg)
    trainer.initialize()
    trainer.train()

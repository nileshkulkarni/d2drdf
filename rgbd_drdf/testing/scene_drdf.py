import copy
import os
import os.path as osp
import pickle as pkl

import imageio
import numpy as np
import trimesh
from loguru import logger

from rgbd_drdf.renderer import (
    pyt3d_renderer as pyt3d_pointcloud_wrapper,
)

from ..data_utils import matterport3d_data
from ..html_utils import scene_html

# from ..html_utils import visdom_visualizer as visdom_utils
from ..nnutils import drdf_model
from ..renderer import render_utils
from ..utils import intersection_finder_utils, tensor_utils, utils_3d
from ..utils.scene_visuals import SceneVisuals
from . import base_tester

c2c = tensor_utils.copy2cpu


class SceneDRDFTester(base_tester.BaseTester, SceneVisuals):
    def __init__(self, opts):
        super().__init__(opts)
        self.opts = opts

    def init_model(
        self,
    ):
        opts = self.opts
        model = drdf_model.DRDFModel(opts)
        self.model = model.to(self.device)
        self.render_func = eval(opts.RENDERER.RENDER_FUNC)
        self.is_latest = False

        if opts.TEST.LATEST_CKPT:
            self.load_network(
                self.model,
                network_label="model",
                epoch_label="epoch_{}".format("latest"),
            )
            self.is_latest = True
            logger.info("Loading the latest checkpoint")
        else:
            self.load_network(
                self.model,
                network_label="model",
                epoch_label=f"epoch_{opts.TRAIN.NUM_EPOCHS}",
            )
            logger.info(f"Loading the checkpoint from epoch {opts.TRAIN.NUM_EPOCHS}")
            self.is_latest = False

        self.intersection_finder = intersection_finder_utils.intersection_finder_drdf
        # self.pyt3d_wrapper = pyt3d_wrapper.Pyt3DWrapper(image_size=(1024, 1280))
        self.model.eval()
        return

    def init_dataset(
        self,
    ):
        opts = self.opts
        if opts.DATALOADER.DATASET_TYPE == "matterport":
            # self.dataloader = matterport3d_data.matterport_dataloader(opts, shuffle=False)
            opts_val = copy.deepcopy(opts)
            opts_val.TRAIN.NUM_WORKERS = 2
            opts_val.TRAIN.BATCH_SIZE = 1
            # opts_val.DATALOADER.SPLIT = "val"
            opts_val.DATALOADER.SHUFFLE_ONCE = True
            self.val_dataloader = matterport3d_data.matterport_dataloader(
                opts_val, shuffle=False
            )
            # self.val_dataloader = self.dataloader
            self.dataloader = self.val_dataloader
            self.val_batch = False
            self.prev_loss = 0.0
            self.opts = opts_val
            self.split = opts.DATALOADER.SPLIT
        else:
            assert False, "dataset not available"

        return

    def set_input(self, batch):
        if batch["empty"] == True:
            self.invalid_batch = True
            return
        opts = self.opts
        input_imgs = batch["image"]
        batch_size = input_imgs.size(0)
        self.input_imgs = input_imgs.to(self.device, non_blocking=True)
        self.RT = batch["RT"].to(self.device)
        self.uuid = batch["uuid"]
        self.Kndc = batch["kNDC"].to(self.device)
        self.index = batch["index"]
        self.img_hr = batch["img_hr"]
        self.image_names = batch["image_name"]
        self.house_ids = batch["house_id"]
        if "mesh" in batch.keys():
            self.meshes = batch["mesh"]

        if False:
            self.mp3d_int = batch["mp3d_int"].numpy()
            self.mp3d_img_size = batch["mp3d_img_size"].numpy()
            self.mp3d_2d_bbox = batch["mp3d_2d_bbox"].numpy()

        if not opts.DATALOADER.INFERENCE_ONLY:
            points = batch["points"].reshape(batch_size, -1, 6)
            points = points.permute(0, 2, 1).to(self.device, non_blocking=True)

            self.points = points[:, :3, :]
            self.points = self.points.requires_grad_(True)
            self.ray_dirs = points[:, :3, :]

            validity = batch["distance_validity"].reshape(batch_size, -1)

            # grad_penalty_mask = torch.stack(grad_penalty_mask_lst).reshape(batch_size, -1)
            grad_penalty_mask = batch["grad_penalty_regions"]
            self.grad_penalty_mask = grad_penalty_mask.to(self.device)
            penalty_types = batch["penalty_types"].reshape(batch_size, -1)
            penalty_types = penalty_types * (validity == 1) - 1 * (validity == 0)
            self.distance = (
                batch["penalty_regions"].reshape(batch_size, -1, 3).to(self.device)
            )
            self.penalty_types = penalty_types.to(self.device)
        return

    def test_create_visuals(
        self,
    ):
        opts = self.opts
        data_iterator = iter(self.dataloader)
        self.html_vis = html_vis = scene_html.HTMLWriter(opts)
        self.total_steps = 0

        for i in range(len(self.dataloader)):
            if (i + 1) > opts.TEST.NUM_ITER:
                break
            batch = next(data_iterator)
            self.set_input(batch)
            # self.predict()
            self.visuals_to_save(visual_count=1)
            self.total_steps += 1
        return

    def test_eval_model(
        self,
    ):
        opts = self.opts
        data_iterator = iter(self.dataloader)
        self.html_vis = html_vis = scene_html.HTMLWriter(opts)
        self.total_steps = 0

        eval_dir = opts.TEST.EVAL_DIR

        if "eval2" not in eval_dir:
            eval_dir = eval_dir.replace("eval", "eval2")

        if opts.TEST.HIGH_RES:
            eval_dir = eval_dir.replace("eval", "high_res_eval")
            opts.MODEL.RESOLUTION = 256

        if self.is_latest:
            self.eval_save_dir = osp.join(eval_dir, "raw", opts.NAME, self.split)
        else:
            epoch_number = opts.TRAIN.NUM_EPOCHS
            self.eval_save_dir = osp.join(
                eval_dir, "raw", opts.NAME, f"{self.split}_{epoch_number}"
            )

        self.overwrite_allowed = opts.TEST.OVERWRITE
        if osp.exists(self.eval_save_dir):
            if not self.overwrite_allowed:
                logger.info(
                    f"Raw dir for {opts.NAME} already exists, not overwriting existing files"
                )
                num_of_files = len(os.listdir(self.eval_save_dir))
                logger.info(f"Already existing files {num_of_files}")
            else:
                logger.info(
                    f"Raw dir for {opts.NAME} already exists, **overwriting** existing files"
                )

        self.select_uuids = False
        if opts.TEST.UUID_SELECT_FILE != "" and opts.TEST.HIGH_RES:
            opts.TEST.NUM_ITER = len(self.dataloader)
            self.select_uuids = True
            uuid_select_list = []
            with open(opts.TEST.UUID_SELECT_FILE) as f:
                uuid_select_list = {l.strip() for l in f.readlines()}
        breakpoint()
        os.makedirs(self.eval_save_dir, exist_ok=True)
        for i in range(len(self.dataloader)):
            if (i + 1) > opts.TEST.NUM_ITER:
                break
            batch = next(data_iterator)
            if self.select_uuids:
                uuid = batch["uuid"][0]
                if uuid not in uuid_select_list:
                    continue
            print(
                "{} , {} , {} ".format(
                    batch["index"], batch["house_id"], batch["image_name"]
                )
            )
            self.set_input(batch)
            self.save_prediction_single_sample()

        return

    def predict(self, use_grad_penalty=None):
        opts = self.opts
        if use_grad_penalty is None:
            use_grad_penalty = opts.MODEL.USE_GRAD_PENALTY

        predictions, _ = self.model.forward(
            images=self.input_imgs,
            points=self.points,
            ray_dir=self.ray_dirs * 0,
            kNDC=self.Kndc,
            RT=self.RT,
            penalty_regions=self.distance,
            penalty_types=self.penalty_types,
            grad_penalty_masks=self.grad_penalty_mask,
            use_grad_penalty=use_grad_penalty,
        )
        self.predictions = predictions
        return

    def visuals_to_save(self, visual_count):
        batch_visuals = []
        opts = self.opts
        # predictions = self.predictions
        # project_points = predictions['project_points'].data.cpu().numpy()
        # project_points_depth = predictions['points_cam'].data.cpu().numpy()[:, 2, :]
        img_size = np.array([opts.DATALOADER.IMG_SIZE, opts.DATALOADER.IMG_SIZE])
        pcl_log_dir = osp.join(opts.PCL_LOG_DIR, opts.NAME, opts.DATALOADER.SPLIT)
        os.makedirs(pcl_log_dir, exist_ok=True)
        pcl_render_dir = osp.join(opts.PCL_RENDER_DIR, opts.NAME, opts.DATALOADER.SPLIT)
        os.makedirs(pcl_render_dir, exist_ok=True)
        for bx in range(visual_count):
            image_name = self.image_names[bx]
            house_id = self.house_ids[bx]
            uuid = "{}_{}".format(house_id, image_name.replace(".jpg", ""))
            index = self.index[bx].item()
            visuals = {}
            visuals["image"] = (
                self.input_imgs[bx].detach().cpu().numpy().transpose(1, 2, 0)
            )
            breakpoint()
            RT = self.RT[bx].detach().cpu().numpy()
            gen_struct = self.generate_mesh(bx)
            meta_store = {}
            if True:
                if gen_struct is not None:
                    meta_store["mesh"] = gen_struct.mesh
                    meta_store["mesh_path"] = gen_struct.mesh_path
                    meta_store["pcl"] = gen_struct.pcl
                    meta_store["visibility"] = gen_struct.visibility
                    meta_store["kndc"] = self.Kndc[bx].data.cpu().numpy()
                    meta_store["image"] = visuals["image"]
                    meta_store["RT"] = RT
                    np.savez(f"{pcl_log_dir}/{uuid}.npz", **meta_store)

            if False:
                pcl_renderer_wrapper = pyt3d_pointcloud_wrapper.Pyt3DWrapperPointCloud(
                    image_size=(1080, 1280), device="cuda:0"
                )
                save_dir = f"{index}_pred"
                os.makedirs(save_dir, exist_ok=True)
                # pcl, visibility, image_point_colors, RT
                rendered_images = (
                    pyt3d_pointcloud_wrapper.render_pcl_w_normals_w_visiblity(
                        meta_store["pcl"],
                        gen_struct.visibility,
                        RT,
                        pcl_renderer_wrapper,
                    )
                )

                pcl_render_dir_bx = osp.join(pcl_render_dir, uuid)
                os.makedirs(pcl_render_dir_bx, exist_ok=True)

                imageio.imsave(
                    osp.join(pcl_render_dir_bx, "image.png"), visuals["image"]
                )

                for rx, rendered_img in enumerate(rendered_images):
                    # visuals['pred_view_{}'.format(rx)] = rendered_img[rx]
                    imageio.imsave(osp.join(pcl_render_dir_bx, f"pred_view_{rx}.img"))

        return

    def save_prediction_single_sample(
        self,
    ):
        ## save the predictions as .npz file
        opts = self.opts
        uuid = self.uuid[0]
        save_file = osp.join(self.eval_save_dir, uuid + ".pkl")

        if osp.exists(save_file) and (not self.overwrite_allowed):
            logger.info(f"Skipping already exisiting file {uuid}")
            return
        image_name = self.image_names[0]
        house_id = self.house_ids[0]
        index = self.index[0].item()
        visuals = {}
        # source_mesh = self.meshes[0]
        visuals["image"] = self.input_imgs[0].detach().cpu().numpy().transpose(1, 2, 0)
        RT = self.RT[0].detach().cpu().numpy()
        gen_struct = self.generate_mesh(0)
        kwargs = {}
        kwargs["image"] = visuals["image"]
        if opts.TEST.HIGH_RES:
            kwargs["image_hr"] = self.img_hr[0]
        # kwargs["gt_mesh"] = source_mesh
        kwargs["RT"] = c2c(RT)
        kwargs["Kndc"] = c2c(self.Kndc[0])
        kwargs["visibility"] = c2c(gen_struct.visibility)
        kwargs["pcl"] = c2c(gen_struct.pcl)
        kwargs["raw_intersections"] = c2c(gen_struct.raw_intersections)
        # kwargs["query_coords"] = c2c(gen_struct.query_coords)
        # kwargs["ray_dir"] = c2c(gen_struct.ray_dir)
        # kwargs["distances"] = c2c(gen_struct.distances)
        # kwargs["ray_intersection_coords"] = c2c(gen_struct.ray_intersection_coords)
        with open(save_file, "wb") as f:
            pkl.dump(kwargs, f)
        return

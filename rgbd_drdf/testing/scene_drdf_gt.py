import copy
import os
import os.path as osp
import pickle as pkl

import imageio
import numpy as np
import trimesh
from loguru import logger
from ..data_utils import matterport3d_data
from ..html_utils import scene_html
from ..utils import intersection_finder_utils, tensor_utils, utils_3d
from ..utils.scene_visuals import SceneVisuals
from . import base_tester

c2c = tensor_utils.copy2cpu


class SceneGT(base_tester.BaseTester, SceneVisuals):
    def __init__(self, opts):
        super().__init__(opts)
        self.opts = opts

    def init_model(
        self,
    ):
        opts = self.opts
       
        self.intersection_finder = intersection_finder_utils.intersection_finder_drdf
        return

    def init_dataset(
        self,
    ):
        opts = self.opts
        self.dataset_name = opts.DATALOADER.DATASET_TYPE
        if opts.DATALOADER.DATASET_TYPE == "matterport":
            # self.dataloader = matterport3d_data.matterport_dataloader(opts, shuffle=False)
            opts_val = copy.deepcopy(opts)
            opts_val.TRAIN.NUM_WORKERS = 1
            opts_val.TRAIN.BATCH_SIZE = 1
            # opts_val.DATALOADER.SPLIT = "val"
            opts_val.DATALOADER.SHUFFLE_ONCE = True
            self.val_dataloader = matterport3d_data.matterport_dataloader(
                opts_val, shuffle=False
            )
            self.split = opts.DATALOADER.SPLIT
            self.dataloader = self.val_dataloader
            self.val_batch = False
            self.prev_loss = 0.0
            self.opts = opts_val

        elif opts.DATALOADER.DATASET_TYPE == "taskonomy":
            opts_val = copy.deepcopy(opts)
            opts_val.TRAIN.NUM_WORKERS = 1
            opts_val.TRAIN.BATCH_SIZE = 1
            opts_val.DATALOADER.SPLIT = "medium/val"
            self.val_dataloader = taskonomy_data.taskonomy_dataloader(
                opts_val, shuffle=False
            )
            self.dataloader = self.val_dataloader
            self.val_batch = False
            self.prev_loss = 0.0
            self.opts = opts_val
            self.split = "val"
        elif opts.DATALOADER.DATASET_TYPE == "threedf":

            opts_val = copy.deepcopy(opts)
            opts_val.TRAIN.NUM_WORKERS = 1
            opts_val.TRAIN.BATCH_SIZE = 1
            opts_val.DATALOADER.SPLIT = "new/val"
            self.val_dataloader = threedf_data.threedf_dataloader(
                opts_val, shuffle=False
            )
            self.dataloader = self.val_dataloader
            self.val_batch = False
            self.prev_loss = 0.0
            self.opts = opts_val
            self.split = "val"
        elif opts.DATALOADER.DATASET_TYPE == "scannet":
            opts_val = copy.deepcopy(opts)
            opts_val.TRAIN.NUM_WORKERS = 1
            opts_val.TRAIN.BATCH_SIZE = 1
            opts_val.DATALOADER.SPLIT = "val"
            self.val_dataloader = scannet_data.scannet_dataloader(
                opts_val, shuffle=False
            )
            self.dataloader = self.val_dataloader
            self.val_batch = False
            self.prev_loss = 0.0
            self.opts = opts_val
            self.split = "val"
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
        # self.depth_gt = batch["depth"]
        self.RT = batch["RT"].to(self.device)
        self.uuid = batch["uuid"]
        self.Kndc = batch["kNDC"].to(self.device)
        self.index = batch["index"]
        self.image_names = batch["image_name"]
        self.house_ids = batch["house_id"]
        self.meshes = batch["mesh"]
        self.img_hr = batch["img_hr"]
        if "mp3d_int" in batch.keys():
            self.mp3d_int = batch["mp3d_int"].numpy()
            self.mp3d_img_size = batch["mp3d_img_size"].numpy()
            self.mp3d_2d_bbox = batch["mp3d_2d_bbox"].numpy()
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

    def dump_eval_data(
        self,
    ):
        opts = self.opts
        data_iterator = iter(self.dataloader)
        self.html_vis = html_vis = scene_html.HTMLWriter(opts)
        self.total_steps = 0

        if opts.TEST.HIGH_RES:
            opts.TEST.EVAL_DIR = opts.TEST.EVAL_DIR.replace("eval", "high_res_eval")
            opts.MODEL.RESOLUTION = 256
        self.eval_save_dir = osp.join(opts.TEST.EVAL_DIR, "raw", opts.NAME, self.split)
        os.makedirs(self.eval_save_dir, exist_ok=True)
        gt_order = []
        gt_order_file = osp.join(
            self.eval_save_dir, f"{self.dataset_name}_{self.split}.lst"
        )

        self.select_uuids = False
        if opts.TEST.UUID_SELECT_FILE != "" or opts.TEST.HIGH_RES:
            opts.TEST.NUM_ITER = len(self.dataloader)
            self.select_uuids = True
            uuid_select_list = []
            with open(opts.TEST.UUID_SELECT_FILE) as f:
                uuid_select_list = {l.strip() for l in f.readlines()}

        # uuid_select_list = ['sKLMLpTHeUy_affe7c5613254862af987bc30a966e31_i1_5']
        # breakpoint()
        for i in range(len(self.dataloader)):
            if (i + 1) > opts.TEST.NUM_ITER:
                break

            batch = next(data_iterator)
            if self.select_uuids:
                uuid = batch["uuid"][0]
                if uuid not in uuid_select_list:
                    continue
            gt_order.append(batch["uuid"][0])
            print(
                "{} , {} , {} ".format(
                    batch["index"], batch["house_id"], batch["image_name"]
                )
            )
            self.set_input(batch)
            self.save_gt_single_sample()

        with open(gt_order_file, "w") as f:
            for lx, line in enumerate(gt_order):
                f.write(line)
                f.write("\n")
        return

    def save_gt_single_sample(
        self,
    ):
        ## save the predictions as .npz file
        opts = self.opts
        uuid = self.uuid[0]

        save_file = osp.join(self.eval_save_dir, uuid + ".pkl")
        image_name = self.image_names[0]
        house_id = self.house_ids[0]
        index = self.index[0].item()

        if not opts.TEST.OVERWRITE and osp.exists(save_file):
            logger.info(f"Not overwriting already exists {image_name}")
            return
        gen_struct = self.generate_gt_point_cloud(0)
        visuals = {}
        source_mesh = self.meshes[0]
        visuals["image"] = self.input_imgs[0].detach().cpu().numpy().transpose(1, 2, 0)
        RT = self.RT[0].detach().cpu().numpy()
        kwargs = {}

        kwargs["image_hr"] = self.img_hr[0]
        kwargs["image"] = visuals["image"]
        kwargs["gt_mesh"] = source_mesh
        kwargs["RT"] = c2c(RT)
        kwargs["Kndc"] = c2c(self.Kndc[0])
        kwargs["visibility"] = c2c(gen_struct.visibility)
        kwargs["pcl"] = c2c(gen_struct.pcl)
        kwargs["raw_intersections"] = c2c(gen_struct.raw_intersections)
        kwargs["query_coords"] = c2c(gen_struct.query_coords)
        kwargs["ray_dir"] = c2c(gen_struct.ray_dir)
        kwargs["distances"] = c2c(gen_struct.distances)
        kwargs["ray_intersection_coords"] = c2c(gen_struct.ray_intersection_coords)
        with open(save_file, "wb") as f:
            pkl.dump(kwargs, f)
        return
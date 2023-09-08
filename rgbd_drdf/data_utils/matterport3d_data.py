import collections
import io
import os
import os.path as osp
import pdb
import pickle as pkl
import random
import time

# from ..utils import grid_utils
import zipfile
from typing import Any, DefaultDict, Dict, List, Tuple

import cv2
import imageio
import numpy as np
import torch
import trimesh
from fvcore.common.config import CfgNode
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

from ..utils import (
    data_sampling,
    geometry_utils,
    matterport_parse,
    mesh_utils,
    ray_utils,
)
from ..utils.default_imports import *
from . import base_data
from .base_data import BaseData, collate_fn, worker_init_fn

curr_dir = osp.dirname(osp.abspath(__file__))

cachedir = osp.join(osp.dirname(osp.abspath(__file__)), "..", "cachedir")


class MatterportData(BaseData):
    def __init__(self, opts: CfgNode):
        super().__init__(opts)
        self.dataset_dir = opts.MATTERPORT_PATH

        self.filter_keys = [
            "_i1_",  # "_i2_", "_i0_"
        ]  # we are using only these as the i0 and i2 point to the bottom
        _house_ids = os.listdir(
            osp.join(self.dataset_dir, "undistorted_camera_parameters")
        )
        split_dir = osp.join(cachedir, "splits")
        split_file = osp.join(split_dir, "mp3d_split.pkl")
        # # house_ids = os.listdir(
        # #     osp.join(self.dataset_dir, 'undistorted_camera_parameters')
        # # )
        with open(split_file, "rb") as f:
            splits = pkl.load(f)
        # splits = read_house_splits(split_file, house_ids)
        # house_ids = house_ids[0:10]
        self.dataloader_mode = opts.DATALOADER.MODE
        _house_ids = splits[opts.DATALOADER.SPLIT]
        if opts.DATALOADER.NUM_HOUSES == 1:
            _house_ids = ["mJXqzFtmKg4"]
        else:
            nhouse = opts.DATALOADER.NUM_HOUSES
            _house_ids = _house_ids[:nhouse]

        self.crop_kwargs = {
            "img_size": opts.DATALOADER.IMG_SIZE,
            "scaling": 1.0,
            "split": opts.DATALOADER.SPLIT,
            "no_crop": opts.DATALOADER.NO_CROP,
        }
        self.meshes = DefaultDict(dict)
        self.mp3d_cachedir = osp.join(cachedir, "mp3d_data_zmx8")
        self.split = split = opts.DATALOADER.SPLIT
        # _house_ids = _house_ids[10:11]
        group_by_region = False

        _indexed_items = matterport_parse.preload_conf_files(
            matterport_dir=self.dataset_dir,
            house_ids=_house_ids,
            group_by_region=group_by_region,
        )
        _indexed_items = matterport_parse.filter_indices(
            _indexed_items, self.filter_keys, keep_valid_regions=False
        )

        sample_frac = opts.DATALOADER.SAMPLE_FRAC

        val_uuids_file = "./matterport_val_uuid_file.lst"
        if split == "train":
            _indexed_items = data_sampling.matterport_create_subsample(
                indexed_items=_indexed_items,
                sample_frac=sample_frac,
                split=opts.DATALOADER.SPLIT,
            )
            good_inds = self.mark_good_inds(_indexed_items, split=opts.DATALOADER.SPLIT)
            _indexed_items = np.array(_indexed_items)[np.array(good_inds)].tolist()

        elif split == "val" and osp.exists(val_uuids_file):
            logger.info("dropping indices that are not needed for evaluation. ")
            logger.info(f"Loading uuids from {val_uuids_file}")
            with open(val_uuids_file) as f:
                val_uuids = [k.strip() for k in f.readlines()]
            val_indexed_items = []
            val_uuids = set(val_uuids)

            for indexed_item in _indexed_items:
                house_id = indexed_item["house_id"]
                image_names = indexed_item["image_names"].replace(".jpg", "")
                uuid = f"{house_id}_{image_names}"
                if uuid in val_uuids:
                    val_indexed_items.append(indexed_item)
            random.seed(time.time())
            _indexed_items = val_indexed_items
            random.shuffle(_indexed_items)
        else:
            good_inds = self.mark_good_inds(_indexed_items, split=opts.DATALOADER.SPLIT)
            _indexed_items = np.array(_indexed_items)[np.array(good_inds)].tolist()

        if opts.DATALOADER.SHUFFLE_ONCE:
            random.seed(0)
            random.shuffle(_indexed_items)
        elif opts.DATALOADER.SHUFFLE:
            random.shuffle(_indexed_items)
        else:
            logger.info("Not shuffling")


        self.houses2mesh = matterport_parse.preload_meshes(
            indexed_items=_indexed_items, matterport_dir=self.dataset_dir
        )
        if split == "train":
            house_ids = self.houses2mesh.keys()
            self.sampled_house2mesh_data = data_sampling.matterport_mesh_sample(
                house_ids, sample_frac, split
            )
        else:
            self.sampled_house2mesh_data = {}

        self.indexed_items = _indexed_items
        self.num_items = len(self.indexed_items)

        return

    def check_example_validity(self, meta_data):
        img_data = self.forward_img(meta_data)
        depth = img_data["depth"]
        if self.opts.DATALOADER.FILTER_DEPTH:
            depth1d = depth.reshape(-1)
            if (depth1d < 3.0).mean() > 0.6:
                elem = {"empty": True}
                return elem
            if (depth1d < 0.001).mean() > 0.5:
                elem = {"empty": True}
                # print('end {} '.format(index))
                return elem

        return {"empty": False}

    def mark_good_inds(self, indexed_items, split):
        opts = self.opts
        checkpoint_dir = osp.join(cachedir, "matterport_good_list", "all")
        # checkpoint_dir = osp.join(cachedir, 'snapshots', opts.NAME)
        if split == "val":
            good_inds_file = osp.join(checkpoint_dir, f"good_inds_file_{split}.pkl")
        else:
            good_inds_file = osp.join(checkpoint_dir, "good_inds_file.pkl")
        logger.info(f"Good Inds file {good_inds_file}")
        if osp.exists(good_inds_file):
            logger.info("Fast loading good inds from file")
            with open(good_inds_file, "rb") as f:
                good_list = pkl.load(f)
        else:
            good_list = []
            for ix in range(len(indexed_items)):
                meta_data = indexed_items[ix]
                elem = self.check_example_validity(meta_data)
                if ix % 1000 == 0:
                    logger.info(f"Checking {ix}/{len(indexed_items)}")
                if not elem["empty"]:
                    uuid = meta_data["uuid"]
                    good_list.append(uuid)

            with open(good_inds_file, "wb") as f:
                pkl.dump(good_list, f)

        good_inds = []
        good_list = set(good_list)
        for ix in range(len(indexed_items)):
            meta_data = indexed_items[ix]
            uuid = meta_data["uuid"]
            if uuid in good_list:
                good_inds.append(ix)

        return good_inds

    def load_mesh(self, meta_data):
        house_id = meta_data["house_id"]

        mesh = self.houses2mesh[house_id]
        mesh = {
            "faces": mesh["faces"] * 1,
            "vertices": mesh["vertices"] * 1,
        }
        if self.opts.DATALOADER.USE_POISSON_MESH and self.split == "train":
            fid_data = np.load(osp.join(self.mp3d_cachedir, meta_data["uuid"] + ".npz"))
            fids = fid_data["fids"]
        else:
            (
                fids,
                face_labels,
                face_segments,
                fids_c,
                fids_w,
            ) = matterport_parse.load_face_metadata(
                metadata=meta_data, face_cachedir=self.mp3d_cachedir
            )

        if fids is not None:
            # mesh['face_labels'] =
            if house_id in self.sampled_house2mesh_data.keys():
                sampled_house_mesh = self.sampled_house2mesh_data[house_id]
                fid_set = set(sampled_house_mesh["faces_ids"])
                # fids = [fid  for fid in fids if fid in sampled_house_mesh['faces_ids_strict']]
                fids = [fid for fid in fids if fid in fid_set]
                fids = np.array(fids)
            mesh["faces"] = mesh["faces"][fids]
            mesh = matterport_parse.delete_unused_verts(
                mesh,
            )
            mesh = trimesh.Trimesh(vertices=mesh["vertices"], faces=mesh["faces"])
            return mesh
        else:
            return None

    def forward_mesh(self, meta_data):
        house_id = meta_data["house_id"]
        region_id = meta_data["region_id"]
        region_mesh_pth = osp.join(
            self.dataset_dir,
            "region_segmentations",
            house_id,
            "region_segmentations",
            f"region{region_id}.ply",
        )
        mesh = None  ## caching meshes!
        if house_id in self.meshes.keys():
            if region_id in self.meshes["house_id"].keys():
                mesh = self.meshes["house_id"]["region_id"]
        if mesh is None:
            with open(region_mesh_pth, "rb") as f:
                mesh = trimesh.exchange.ply.load_ply(f)
            self.meshes[house_id][region_id] = mesh

        simple_mesh = trimesh.Trimesh(mesh["vertices"], mesh["faces"])
        return simple_mesh

    def forward_img(self, meta_data) -> Dict[str, Any]:
        opts = self.opts
        cam2mp, intrinsics = self.forward_camera(meta_data)
        mp3d_int = intrinsics * 1
        # Rx = trimesh.transformations.euler_matrix(np.pi, 0, 0, 'sxyz')
        # cam2mp = np.matmul(cam2mp, Rx)
        img_name = meta_data["image_names"]
        house_id = meta_data["house_id"]
        img_dir = osp.join(
            self.dataset_dir,
            "undistorted_color_images",
            house_id,
            "undistorted_color_images",
        )
        depth_dir = osp.join(
            self.dataset_dir,
            "undistorted_depth_images",
            house_id,
            "undistorted_depth_images",
        )
        # depth_dir = '/scratch/justincj_root/justincj/nileshk/SceneTSDF/Matterport3d/raw/{}/undistorted_depth_images'.format(house_id)
        depth_path = osp.join(
            depth_dir, img_name.replace("_i", "_d").replace(".jpg", ".png")
        )
        img_path = osp.join(img_dir, img_name)
        # print(img_path)
        img = imageio.imread(img_path)
        depth = imageio.imread(depth_path)
        og_img_size = (img.shape[0], img.shape[1])
        bbox = np.array([0, 0, img.shape[1], img.shape[0]])

        crop_kwargs = self.crop_kwargs
        (
            img,
            depth,
            intrinsics,
            K_crop,
            bbox,
            resize_function,
            high_res_img,
            depth_high_res,
        ) = self.center_crop_image(
            img,
            depth,
            intrinsics,
            **crop_kwargs,
        )
        # depth = depth.astype(float) * 0.25
        # depth = depth / 1000
        img = img.astype(np.float32) / 255
        img = img.transpose(2, 0, 1)
        data = {}
        data["img"] = img
        data["depth"] = depth
        data["cam2mp"] = cam2mp
        data["intrinsics"] = intrinsics
        data["mp3d_int"] = mp3d_int
        data["mp3d_img_size"] = og_img_size
        data["mp3d_2d_bbox"] = bbox
        data["img_hr"] = high_res_img
        return data


    def project_points_ndc(self, points, img_size, K, img=None):
        if True:  # Filter points.2
            valid_inds = np.where(points[2, :] > -6)[0]
            points = points[:, valid_inds]

        pdb.set_trace()
        projected_points = geometry_utils.prespective_transform(points, K)
        # img_points = np.matmul(K, points).transpose(1, 0)
        # # pdb.set_trace()
        # img_points[:, 0:2] = img_points[:, 0:2] / img_points[:, 2:3]

        x = projected_points[0, :] * img_size[0] / 2 + img_size[0] / 2
        y = projected_points[1, :] * img_size[1] / 2 + img_size[1] / 2

        x = np.round(np.array(x)).astype(int)
        y = np.round(np.array(y)).astype(int)

        valid = np.logical_and(
            np.logical_and(x >= 0, x < img_size[0]),
            np.logical_and(y >= 0, y < img_size[1]),
        )
        valid_inds = np.where(valid)[0]
        # project to image plane
        x = x[valid_inds]
        y = y[valid_inds]

        mask_img = np.zeros((img_size[1], img_size[0], 3))
        mask_img[y, x, :] = 255

        if img is not None:
            mask_img = np.concatenate([img, mask_img], axis=1)
        imageio.imsave("mask.png", mask_img)
        return mask_img

    def __len__(
        self,
    ):
        return self.num_items

    def __getitem__(self, index):
        opts = self.opts
        if opts.DATALOADER.SINGLE_INSTANCE:
            # index = 5100
            house_id = "mJXqzFtmKg4"
            img_name = "93537c1826904116b02ed6019b20c57d_i1_2.jpg"

            # index = 1700

            found = False
            for ind, ref_item in enumerate(self.indexed_items):
                if (house_id in ref_item["house_id"]) and (
                    img_name in ref_item["image_names"]
                ):
                    found = True
                    break

            assert found, "single instance not found"
            index = ind
        meta_data = self.indexed_items[index]
        # mesh_region = self.forward_mesh(meta_data)
        mesh_house = self.load_mesh(meta_data)
        # trimesh.exchange.export.export_mesh(mesh_house, 'house.ply')
        # trimesh.exchange.export.export_mesh(mesh_region, 'region.ply')
        mesh = mesh_house
        # trimesh.exchange.export.export_mesh(mesh, 'mesh.ply')
        img_data = self.forward_img(meta_data)  ## contains img, depth, cam_ext, cam_int
        image = img_data["img"]
        # imageio.imsave('test.png', image.transpose(1,2,0))
        extrinsics = np.linalg.inv(img_data["cam2mp"])

        img_size = (image.shape[2], image.shape[1])
        elem = {"empty": True}
        K = img_data["intrinsics"] * 1
        if True:
            K[0, 0] = 1 * K[0, 0]  # Flip the x-axis.
            K[1, 1] = 1 * K[1, 1]  # Flip the y-axis.
            K[2, 2] = 1  # Flip the z-axis.

            Kndc_mult = np.array(
                [[2.0 / img_size[0], 0, -1], [0, 2.0 / img_size[1], -1], [0, 0, 1]]
            )

            Kndc = np.matmul(Kndc_mult, K)

        image_names = meta_data["image_names"]
        house_id = meta_data["house_id"]
        uuid = f"{house_id}_{image_names.replace('.jpg','')}"
        elem["empty"] = False
        elem["house_id"] = house_id
        elem["image_name"] = image_names
        elem["image"] = image
        elem["RT"] = extrinsics.astype(np.float32)
        elem["kNDC"] = Kndc.astype(np.float32)
        elem["index"] = index
        elem["uuid"] = uuid

        elem["img_hr"] = img_data["img_hr"]
        if not opts.DATALOADER.INFERENCE_ONLY:

            def is_point_inside(points_temp):
                RT = extrinsics
                points_temp = geometry_utils.transform_points(
                    points_temp.transpose(), RT
                )
                xyz_ndc = geometry_utils.prespective_transform(points_temp, Kndc)
                xyz_valids = (xyz_ndc >= -1) * (xyz_ndc <= 1)

                xyz_valids = xyz_valids[0, :] * xyz_valids[1, :]
                return xyz_valids

            if False:
                points = self.sample_points_mesh(mesh, 100000)

                points = points.transpose() * 1
                points = geometry_utils.transform_points(points, extrinsics)
                proj_img = self.project_points_ndc(points, img_size, Kndc, img=image)
                imageio.imsave("test.png", proj_img)
                pdb.set_trace()

            ## Ray based sampling startegy!
            points = self.sample_rays(
                mesh,
                RT=extrinsics,
                Kndc=Kndc,
                z_max=opts.DATALOADER.SAMPLING.Z_MAX,
                ray_dir_lst=opts.DATALOADER.SAMPLING.RAY_DIR_LST,
                npoints_lst=opts.DATALOADER.SAMPLING.N_RAYS_LST,
            )

            points_shape = points.shape  ## N, N_per_ray, 6
            points = points.reshape(-1, 6)  ## first 3 are point, next 3 are direc

            if False:
                points = points[:, 0:3]
                points = points.transpose() * 1
                points = geometry_utils.transform_points(points, extrinsics)
                proj_img = self.project_points_ndc(
                    points, img_size, Kndc, img=(image * 255).transpose(1, 2, 0)
                )
                imageio.imsave("test.png", proj_img)

            ray_dist, int_loc, valid_int, tri_ids = ray_utils.get_special_ray_distances(
                mesh,
                points=points[:, 0:3],
                rays=points[:, 3:],
                unsigned_ray_dist=opts.DATALOADER.UNSIGNED_RAY_DIST,
                signed=opts.DATALOADER.SIGNED_RAY_DIST,
            )
            # temp = points.reshape(20, -1, 6)
            # mesh_utils.save_point_cloud(temp[-1,256:,0:3], 'ray.ply')

            elem["points"] = points.transpose(1, 0).astype(
                np.float32
            )  ## array of points and directions.
            elem["ray_dist"] = ray_dist.astype(np.float32)

            elem["extents"] = np.array(mesh.extents)
            elem["valid_intersect"] = valid_int.astype(np.float32)
            elem["img_size"] = img_size
            elem["mp3d_int"] = img_data["mp3d_int"]
            elem["mp3d_img_size"] = np.array(img_data["mp3d_img_size"])
            elem["mp3d_2d_bbox"] = img_data["mp3d_2d_bbox"]

            elem["empty"] = False

        if self.split == "train" or self.dataloader_mode == "train":
            del mesh
        else:
            elem["mesh"] = mesh
        return elem


def matterport_dataloader(opts, shuffle=False):
    dataset = MatterportData(opts)
    shuffle = "train" in opts.DATALOADER.SPLIT or shuffle
    persistent_workers = opts.TRAIN.NUM_WORKERS > 0
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opts.TRAIN.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=opts.TRAIN.NUM_WORKERS,
        collate_fn=base_data.collate_fn,
        worker_init_fn=base_data.worker_init_fn,
        persistent_workers=persistent_workers,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader


if __name__ == "__main__":
    from ..config import defaults
    from ..utils import parse_args

    cmd_args = parse_args.parse_args()
    cfg = defaults.get_cfg_defaults()
    if cmd_args.cfg_file is not None:
        cfg.merge_from_file(cmd_args.cfg_file)
    if cmd_args.set_cfgs is not None:
        cfg.merge_from_list(cmd_args.set_cfgs)

    dataset = MatterportData(cfg)
    dataset[100]

import os.path as osp
import pickle as pkl

import numpy as np
from loguru import logger
from tqdm import tqdm

curr_dir = osp.dirname(osp.abspath(__file__))
cachedir = osp.join(osp.dirname(osp.abspath(__file__)), "..", "cachedir")

matterport_sample_files = {
    "0.03": "matterport_3",
    "0.06": "matterport_6",
    "0.13": "matterport_13",
    "0.25": "matterport_25",
    "0.5": "matterport_50",
    "0.75": "matterport_75",
    "1.0": "matterport_100",
}


taskonomy_sample_files = {
    "0.03": "taskonomy_3",
    "0.06": "taskonomy_6",
    "0.13": "taskonomy_13",
    "0.25": "taskonomy_25",
    "0.5": "taskonomy_50",
    "0.75": "taskonomy_75",
    "1.0": "taskonomy_100",
}

tdf_sample_files = {
    "0.25": "threedf_25",
    "0.13": "threedf_13",
    "0.06": "threedf_6",
    "0.03": "threedf_3",
    "0.5": "threedf_50",
    "1.0": "threedf_100",
}

scannet_sample_files = {
    "0.25": "scannet_25",
    "0.13": "scannet_13",
    "0.06": "scannet_6",
    "0.5": "scannet_50",
    "1.0": "scannet_100",
}


def matterport_create_subsample(indexed_items, sample_frac, split):
    sample_file = matterport_sample_files[str(sample_frac)]

    if sample_file != "matterport_100" and split == "train":
        sample_file = osp.join(cachedir, "sampling_splits", sample_file, f"{split}.pkl")
        sampled_indexes = select_sampled_items(
            indexed_items=indexed_items, sample_file=sample_file
        )
        if len(sampled_indexes) > 0:
            indexed_items = np.array(indexed_items)[np.array(sampled_indexes)].tolist()
        else:
            indexed_items = []
    return indexed_items


def tdf_create_subsample(indexed_items, sample_frac, split):

    sample_file = tdf_sample_files[str(sample_frac)]

    if sample_file != "tdf_100" and split == "train":
        sample_file = osp.join(cachedir, "sampling_splits", sample_file, f"{split}.pkl")
        sampled_indexes = select_sampled_items(
            indexed_items=indexed_items, sample_file=sample_file
        )
        if len(sampled_indexes) > 0:
            indexed_items = np.array(indexed_items)[np.array(sampled_indexes)].tolist()
        else:
            indexed_items = []
    return indexed_items


def scannet_create_subsample(indexed_items, sample_frac, split):
    sample_file = scannet_sample_files[str(sample_frac)]
    if sample_file != "scannet_100" and split == "train":
        sample_file = osp.join(cachedir, "sampling_splits", sample_file, f"{split}.pkl")
        sampled_indexes = select_sampled_items(
            indexed_items=indexed_items, sample_file=sample_file
        )
        if len(sampled_indexes) > 0:
            indexed_items = np.array(indexed_items)[np.array(sampled_indexes)].tolist()
        else:
            indexed_items = []
    return indexed_items


def select_sampled_items(indexed_items, sample_file):
    with open(sample_file, "rb") as f:
        sample_data = pkl.load(f)
    sampled_indexes = []
    for ix, index_item in enumerate(indexed_items):
        house_id = index_item["house_id"]
        sample_img_ids = sample_data[house_id]
        img_id = index_item["image_names"]
        if type(img_id) == str:
            if img_id.replace(".jpg", "") in sample_img_ids:
                sampled_indexes.append(ix)
        else:
            if img_id in sample_img_ids:
                sampled_indexes.append(ix)

    return sampled_indexes


def scannet_mesh_sample(house_ids, sample_frac, split):
    sample_dir = scannet_sample_files[str(sample_frac)]
    if sample_dir == "scannet_100":
        return {}
    sample_dir = osp.join(cachedir, "sampling_splits", sample_dir, "mesh")
    sample_house2mesh_data = {}
    logger.info("Loading sample frac face ids for houses")
    for house_id in tqdm(house_ids):
        house_file = osp.join(sample_dir, f"{house_id}.pkl")
        with open(house_file, "rb") as f:
            house_data = pkl.load(f)
        sample_house2mesh_data[house_id] = {"faces_ids": house_data["faces_ids"]}
    return sample_house2mesh_data


def tdf_mesh_sample(house_ids, sample_frac, split):
    sample_dir = tdf_sample_files[str(sample_frac)]
    if sample_dir == "threedf_100":
        return {}
    sample_dir = osp.join(cachedir, "sampling_splits", sample_dir, "mesh")
    sample_house2mesh_data = {}
    logger.info("Loading sample frac face ids for houses")
    for house_id in tqdm(house_ids):
        house_file = osp.join(sample_dir, f"{house_id}.pkl")
        with open(house_file, "rb") as f:
            house_data = pkl.load(f)
        sample_house2mesh_data[house_id] = {
            "faces_ids": house_data["faces_ids"],
            "faces_ids_strict": house_data["faces_ids_strict"],
        }
    return sample_house2mesh_data


def matterport_mesh_sample(house_ids, sample_frac, split):
    sample_dir = matterport_sample_files[str(sample_frac)]
    if sample_dir == "matterport_100":
        return {}
    sample_dir = osp.join(cachedir, "sampling_splits", sample_dir, "mesh")
    sample_house2mesh_data = {}

    for house_id in house_ids:
        house_file = osp.join(sample_dir, f"{house_id}.pkl")
        with open(house_file, "rb") as f:
            house_data = pkl.load(f)
        sample_house2mesh_data[house_id] = house_data
    return sample_house2mesh_data


def taskonomy_create_subsample(indexed_items, sample_frac, split):

    sample_file = taskonomy_sample_files[str(sample_frac)]

    if sample_file != "taskonomy_100" and "train" == split:
        sample_file = osp.join(cachedir, "sampling_splits", sample_file, f"{split}.pkl")
        sampled_indexes = select_sampled_items(
            indexed_items=indexed_items, sample_file=sample_file
        )
        indexed_items = np.array(indexed_items)[np.array(sampled_indexes)].tolist()
    return indexed_items


def taskonomy_mesh_sample(house_ids, sample_frac, split):
    sample_dir = taskonomy_sample_files[str(sample_frac)]
    if sample_dir == "taskonomy_100":
        return {}
    sample_dir = osp.join(cachedir, "sampling_splits", sample_dir, "mesh")
    sample_house2mesh_data = {}

    for house_id in house_ids:
        house_file = osp.join(sample_dir, f"{house_id}.pkl")
        with open(house_file, "rb") as f:
            house_data = pkl.load(f)
        sample_house2mesh_data[house_id] = house_data
    return sample_house2mesh_data

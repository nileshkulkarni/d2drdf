"""

# Preprocess matterport data store visible faces.


"""
import csv
import os
import os.path as osp
import pdb

import imageio
import numpy as np
import ray
import scipy.io as sio
import torch
import trimesh
from loguru import logger

from rgbd_drdf.utils import geometry_utils as geometry
from rgbd_drdf.utils import matterport_parse

filter_keys = ["_i1_"]
regex = r"(\w+)_(i)(\d)_(\d).jpg"
"""
ext : [4x4]
K: [3x3]
vertices: [N x 3]

"""

from absl import app, flags

flags.DEFINE_integer("start_index", 0, "start index for house")

flags.DEFINE_integer("end_index", -1, "start index for house")

from trimesh.ray.ray_pyembree import RayMeshIntersector

MAX_ZMAX = 8
MIN_ZMIN = 0

curr_path = osp.dirname(osp.abspath(__file__))
cachedir = osp.join(curr_path, "..", "cachedir")

# from ..data import matterport

# class_ignore_list = matterport.class_ignore_list



def find_visible_verts(ext, mesh, points):
    tri_mesh = trimesh.Trimesh(
        vertices=mesh["vertices"], faces=mesh["faces"], process=False
    )
    rayintesector = RayMeshIntersector(tri_mesh, scale_to_box=False)
    RT = np.linalg.inv(ext)
    T = RT[0:3, 3]
    ray_origins = (
        points * 0
        + T[
            None,
        ]
    )
    ray_directions = points - ray_origins
    triangle_index = rayintesector.intersects_first(ray_origins, ray_directions)
    temp = np.where(triangle_index > 0)[0]
    triangle_index = triangle_index[temp]
    triangle_index = np.unique(triangle_index)
    new_faces = tri_mesh.faces[triangle_index]
    return new_faces, triangle_index


def find_visible_faces(
    ext,
    mesh,
):
    faces = mesh["faces"]
    verts = mesh["vertices"]
    vert_ids = faces.reshape(-1)
    vert_points = verts[faces.reshape(-1)].reshape(-1, 3, 3)
    points = vert_points.mean(1)
    tri_mesh = trimesh.Trimesh(
        vertices=mesh["vertices"], faces=mesh["faces"], process=False
    )
    points2, _ = trimesh.sample.sample_surface(tri_mesh, len(faces))
    # pdb.set_trace()
    all_points = np.concatenate([points, points2, vert_points.reshape(-1, 3)])
    new_faces, valid_ids = find_visible_verts(ext, mesh, points)
    return new_faces, valid_ids


def find_valid_verts(ext, K, vertices, img_size):
    is_numpy = False
    if type(ext) == np.ndarray:
        is_numpy = True
        ext = torch.FloatTensor(ext)
        K = torch.FloatTensor(K)
        vertices = torch.FloatTensor(vertices)

    vertices = vertices.transpose(1, 0)  # 3 x  N

    vertices = vertices[None].cuda()
    ext = ext[None].cuda()
    K = K[None].cuda()

    vertsCam = geometry.transform_points(vertices, ext)
    img_points = geometry.prespective_transform(vertsCam, K)

    # vertsCam = vertsCam[:, 0:3, :]
    # depth = -1 * vertsCam[:, None, 2]
    # vertsCam = vertsCam / vertsCam[:, None, 2]
    # img_points = torch.bmm(K, vertsCam)
    x = img_points[:, 0]
    y = img_points[:, 1]
    depth = vertsCam[:, 2, :] / K[:, 2, 2]
    validity = (
        (x >= 0)
        * (x < img_size[0])
        * (y >= 0)
        * (y < img_size[1])
        * (depth >= 0)
        * (depth < MAX_ZMAX)
    )
    return validity


def find_valid_faces(ext, K, vertices, faces, img_size):
    valid_f1 = find_valid_verts(ext, K, vertices[faces[:, 0]], img_size)
    valid_f2 = find_valid_verts(ext, K, vertices[faces[:, 1]], img_size)
    valid_f3 = find_valid_verts(ext, K, vertices[faces[:, 2]], img_size)
    valid_faces = torch.logical_or(valid_f1, torch.logical_or(valid_f2, valid_f3))
    valid_fids = torch.where(valid_faces)[1]
    valid_fids = valid_fids.cpu().numpy()
    return valid_fids


def get_depths(ext, K, vertices, img_size):
    is_numpy = False
    if type(ext) == np.ndarray:
        is_numpy = True
        ext = torch.FloatTensor(ext)
        K = torch.FloatTensor(K)
        vertices = torch.FloatTensor(vertices)

    vertices = vertices.transpose(1, 0)  # 3 x  N
    vertices = vertices[None].cuda()
    ext = ext[None].cuda()
    K = K[None].cuda()

    vertsCam = geometry.transform_points(vertices, ext)
    img_points = geometry.prespective_transform(vertsCam, K)

    # vertsCam = vertsCam[:, 0:3, :]
    # depth = -1 * vertsCam[:, None, 2]
    # vertsCam = vertsCam / vertsCam[:, None, 2]
    # img_points = torch.bmm(K, vertsCam)
    x = img_points[:, 0]
    y = img_points[:, 1]
    depth = vertsCam[:, 2, :] / K[:, 2, 2]
    valid_depths = depth <= 0
    return depth[valid_depths]


class MatterportPreprocess:
    def __init__(self, opts, matterport_cachedir, start_index, end_index):
        self.opts = opts
        self.matterport_dir = osp.join("./data_dir/", "Matterport3d/scans")
        self.start_index = start_index
        self.end_index = end_index
        self.matterport_cachedir = matterport_cachedir

        self.catIds = []
        # for key, value in self.categoryId2nyuClass.items():
        #     ignore = False
        #     for ignore_word in class_ignore_list:
        #         if ignore_word in value:
        #             # logger.info(value)
        #             ignore = True
        #             break
        #         if value == '':
        #             ignore = True
        #     if ignore:
        #         # logger.info(value)
        #         continue
        #     else:
        #         # logger.info(value)
        #         self.catIds.append(int(key))

    def get_dump_save_path(self, meta_data):
        save_dir = osp.join(self.matterport_cachedir, meta_data["house_id"])
        mat_filename = meta_data["image_names"].replace(".jpg", ".mat")
        save_path = osp.join(save_dir, mat_filename)
        return save_path

   

    def dump_visible_faces_per_index(self, meta_data):
        img_name = meta_data["image_names"]
        house_id = meta_data["house_id"]

        cam2world = meta_data["cam2world"]
        K = meta_data["intrinsics"]
        img = self.load_img(house_id=house_id, img_name=img_name)

        img_size = (img.shape[2], img.shape[1])
        K[0, 0] = -1 * K[0, 0]
        K[2, 2] = -1 * K[2, 2]
        mesh = self.load_mesh(house_id)
        vertices = mesh["vertices"]
        faces = mesh["faces"]
        ext = np.linalg.inv(cam2world)
        valid_fids = find_valid_faces(ext, K, vertices, faces, img_size)

        mesh["faces"] = mesh["faces"][valid_fids]

        data = {}
        data["fids"] = valid_fids
    

        mesh_low_res = self.load_mesh(house_id, low_res=True)
        valid_fids = find_valid_faces(
            ext, K, mesh_low_res["vertices"], mesh_low_res["faces"], img_size
        )
        mesh_low_res["faces"] = mesh_low_res["faces"][valid_fids]
     
        save_path = self.get_dump_save_path(meta_data)
        if not osp.exists(osp.dirname(save_path)):
            os.makedirs(osp.dirname(save_path), exist_ok=True)
        sio.savemat(save_path, data)

    
        return

    def load_img(self, img_name, house_id):
        # img_name = self.indexed_items[index]['image_names']
        # house_id = self.indexed_items[index]['house_id']
        img_dir = osp.join(
            self.matterport_dir,
            "undistorted_color_images",
            house_id,
            "undistorted_color_images",
        )
        img_path = osp.join(img_dir, img_name)
        img = imageio.imread(img_path)
        # logger.info(img_path)
        img = (img[:, :, 0:3]).astype(np.float) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.array(img)
        return img



    def load_mesh(self, house_id, low_res=False):
        # house_id = self.indexed_items[index]['house_id']
        if low_res:
            mesh_dir = osp.join(
                self.matterport_dir,
                "poisson_meshes",
                house_id,
                "poisson_meshes",
            )
        else:
            mesh_dir = osp.join(
                self.matterport_dir,
                "house_segmentations",
                house_id,
                "house_segmentations",
            )

        mesh_path = osp.join(mesh_dir, f"{house_id}.ply")

        with open(mesh_path, "rb") as f:
            mesh = trimesh.exchange.ply.load_ply(f)
        # pdb.set_trace()
        return mesh

    def preload_conf_files(self, house_ids):
        indexed_items = []
        matterport_dir = self.matterport_dir
        for house_id in house_ids:
            conf_file_dir = osp.join(
                matterport_dir,
                "undistorted_camera_parameters",
                house_id,
                "undistorted_camera_parameters",
            )
            conf_file_path = osp.join(conf_file_dir, f"{house_id}.conf")

            with open(conf_file_path) as f:
                conf_data = matterport_parse.parse_conf_file(f)

            for ix, _ in enumerate(conf_data["image_names"]):
                ix_struct = {}
                ix_struct["house_id"] = house_id
                for key in conf_data.keys():
                    ix_struct[key] = conf_data[key][ix]

                indexed_items.append(ix_struct)
        return indexed_items

    def filter_indices(self, indexed_items):
        filtered_items = []
        for index_item in indexed_items:
            for filter_key in filter_keys:
                if filter_key in index_item["image_names"]:
                    filtered_items.append(index_item)
        return filtered_items

    def read_matterport_data(
        self,
    ):
        # List matterport
        opts = self.opts
        house_ids = os.listdir(
            osp.join(self.matterport_dir, "undistorted_camera_parameters")
        )
        house_ids = ["17DRP5sb8fy"]

        logger.info(f"start_index : {self.start_index} end_index {self.end_index}")
        # house_ids = [
        #     'r1Q1Z4BcV1o', 'kEZ7cmS4wCh', 'yqstnuAEVhm', 'r47D5H71a5s',
        #     'JeFG25nYj2p', 'gTV8FGcVJC9', 'EDJbREhghzL', 'VLzqgDo317F',
        #     'PuKPg4mmafe', 'UwV83HsGsw3'
        # ]
        house_ids = house_ids[self.start_index : self.end_index]

        logger.info(house_ids)
        self.indexed_items = self.preload_conf_files(house_ids)
        self.indexed_items = self.filter_indices(self.indexed_items)
        return

    def compute_mesh_depth(self, index):
        img_name = self.indexed_items[index]["image_names"]
        house_id = self.indexed_items[index]["house_id"]

        cam2world = self.indexed_items[index]["cam2world"]
        K = self.indexed_items[index]["intrinsics"]
        img = self.load_img(house_id=house_id, img_name=img_name)

        img_size = (img.shape[2], img.shape[1])
        K[0, 0] = -1 * K[0, 0]
        K[2, 2] = -1 * K[2, 2]
        mesh = self.load_mesh(house_id)
        vertices = mesh["vertices"]
        faces = mesh["faces"]
        ext = np.linalg.inv(cam2world)

        visible_depths = get_depths(ext, K, vertices, img_size)
        visible_depths = visible_depths.data.cpu().numpy()
        self.depths.append(visible_depths)
        return

    def compute_clip_depth(
        self,
    ):
        self.read_matterport_data()
        num_imgs = len(self.indexed_items)
        self.depths = []
        self.depths_ix = {}
        for ix in range(num_imgs):
            if ix % 1000 == 0:
                logger.info(f"{ix}/{num_imgs}")
            try:
                vs_dephts = self.compute_mesh_depth(ix)
                self.depths_ix[ix] = vs_dephts
            except FileNotFoundError as e:
                logger.info(f"Skipping file {ix}")
        torch.save(self.depths_ix, osp.join(cachedir, "matterport", "depth_stats.pth"))
        self.depths = np.concatenate(self.depths)
        self.depths = np.abs(self.depths)
        percentiles = [85 + 2 * i for i in range(8)]
        for per in percentiles:
            logger.info(f"Percentile {per} {np.percentile(self.depths, per)}")

    def preprocess(self):
        self.read_matterport_data()
        num_imgs = len(self.indexed_items)
        for ix in range(0, num_imgs):
            logger.info(ix)
            if ix % 1000 == 0:
                logger.info(f"{ix}/{num_imgs}")
            self.dump_visible_faces_per_index(self.indexed_items[ix])


def preprocess_matterport(opts, matterport_cachedir, start_ind, end_ind):
    matterport_processor = MatterportPreprocess(
        opts, matterport_cachedir, start_ind, end_ind
    )
    matterport_processor.preprocess()
    # test(matterport_processor, index=18)


def test(matterport_processor, index=18):
    matterport_processor.read_matterport_data()
    house_id = matterport_processor.indexed_items[index]["house_id"]
    mesh = matterport_processor.load_mesh(house_id)
    matterport_processor.dump_visible_faces_per_index(index)
    save_path = matterport_processor.get_dump_save_path(index)
    data = sio.loadmat(save_path)
    visible_fids = data["fids"][0]
    # pdb.set_trace()
    test_mesh = trimesh.Trimesh(mesh["vertices"], mesh["faces"][visible_fids])
    trimesh.exchange.export.export_mesh(test_mesh, "temp2.obj")
    logger.info("done")
    return


FLAGS = flags.FLAGS


def main(_):
    opts = FLAGS
    matterport_cachedir = "./rgbd_drdf/cachedir/mp3d_data_zmx8"
    matterport_processor = MatterportPreprocess(
        opts, matterport_cachedir, opts.start_index, opts.end_index
    )
    matterport_processor.preprocess()


if __name__ == "__main__":
    import sys

    app.run(main)

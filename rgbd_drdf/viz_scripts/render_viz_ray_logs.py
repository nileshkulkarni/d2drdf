import glob
import os
import os.path as osp
import pickle as pkl

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from fvcore.common.config import CfgNode

from rgbd_drdf.renderer import (
    pyt3d_renderer as pyt3d_pointcloud_wrapper,
)
from rgbd_drdf.utils import geometry_utils, mesh_utils, plt_vis
from rgbd_drdf.utils.intersection_finder_utils import intersection_finder_drdf
from rgbd_drdf.utils.tensor_utils import copy2cpu as c2c
from rgbd_drdf.utils.tensor_utils import tensor_to_cuda as t2c

_C = CfgNode()

_C.RAW_LOG_DIR = osp.join(
    osp.dirname(osp.abspath(__file__)), "../", "cachedir", "eval/", "raw"
)

_C.RAY_LOG_DIR = osp.join(
    osp.dirname(osp.abspath(__file__)), "../", "cachedir", "pcl_ray_logs"
)

_C.DIR_NAME = (
    "master_4_rgbd_drdf_all_houses_tanh_jitter_bs_pt_depth_no_occ_jitter_no_grad_v100"
)
_C.SPLIT = "train"
NUM_RAYS = 20
overwrite = True


def get_cfg_defaults():
    return _C.clone()


from rgbd_drdf.utils import ray_utils


def unison_shuffled_copies(a, b, rng_state=None):
    assert len(a) == len(b)
    if rng_state:
        p = rng_state.permutation(len(a))
    else:
        p = np.random.permutation(len(a))
    return a[p], b[p], p


def draw_pixel_on_image(image, ray_px, ray_py):
    img = cv2.circle(image, (ray_px * 2, ray_py * 2), 5, (0, 255, 0), -1)
    return img


class RenderRayAndViz:
    def __init__(self, opts):
        render_dir = opts.RAY_LOG_DIR
        name = opts.DIR_NAME
        split = opts.SPLIT

        ## read the npz paths ---

        ## use the pytorch3d wrapper to render sutff

        ## create html ---

        raw_log_dir = osp.join(opts.RAW_LOG_DIR, opts.DIR_NAME, opts.SPLIT)
        ray_log_dir = osp.join(opts.RAY_LOG_DIR, opts.DIR_NAME, opts.SPLIT)
        os.makedirs(raw_log_dir, exist_ok=True)
        if True:
            for gx, npz_file in enumerate(glob.glob(f"{raw_log_dir}/*.pkl")):
                basename = osp.basename(npz_file)
                uuid = basename.replace(".pkl", "")
                ray_log_dir_gx = osp.join(ray_log_dir, uuid)
                if not overwrite and osp.exists(ray_log_dir_gx):
                    ray_log_dir_gx = osp.join(ray_log_dir, uuid + "_v1")
                os.makedirs(ray_log_dir_gx, exist_ok=True)
                self.render_random_rays(npz_file, ray_log_dir_gx)
                if gx > 10:
                    break
        html_name = osp.join(opts.RAY_LOG_DIR, opts.DIR_NAME, f"pred_{opts.SPLIT}.html")
        create_viz_html(render_dir=ray_log_dir, html_name=html_name)

    @staticmethod
    def render_random_rays(pkl_path, render_dir):
        with open(pkl_path, "rb") as f:
            pred_data = pkl.load(f)
        basename = osp.basename(pkl_path)
        rng_state = np.random.RandomState([ord(c) for c in basename])
        RT = pred_data["RT"]
        Kndc = pred_data["Kndc"]
        visibility = pred_data["visibility"]
        pcl = pred_data["pcl"]
        image = pred_data["image"]
        gt_mesh = pred_data["gt_mesh"]
        coords = pred_data["query_coords"][..., 0:3]
        ray_dir = pred_data["ray_dir"]
        pred_distances = pred_data["distances"]
        raw_intersections = pred_data["raw_intersections"]
        raw_intersection_inds = pred_data["ray_intersection_coords"]

        gt_ray_distance_finder = ray_utils.GroundTruthRayDistance(pred_data["gt_mesh"])

        int_count = (raw_intersections >= 0.0).sum(axis=-1)

        ray_inds_px, ray_inds_py = np.where(int_count)

        ray_inds_px, ray_inds_py, perm = unison_shuffled_copies(
            ray_inds_px, ray_inds_py, rng_state=rng_state
        )
        ray_int_count = int_count[(ray_inds_px, ray_inds_py)]

        # ray_inds_px, ray_inds_py, ray_int_count = ray_inds_px[19:20], ray_inds_py[19:20], ray_int_count[19:20]
        camera_center = np.linalg.inv(RT)[0:3, 3]

        for ix, (ray_px, ray_py, intx) in enumerate(
            zip(ray_inds_px, ray_inds_py, ray_int_count)
        ):

            points_ix = coords[ray_px, ray_py]
            temp = points_ix[-1] - points_ix[0]
            ray_dir_ix = temp / np.linalg.norm(temp)
            ray_dir_ix = ray_dir_ix[None, :] + points_ix * 0
            # ray_dir_ix = ray_dir[ray_px, ray_py]
            if False:
                mesh_utils.save_point_cloud(
                    points_ix, osp.join(render_dir, f"ray_{ray_px}px_{ray_py}py.ply")
                )
            # points_ix_local = geometry_utils.transform_points(points_ix.transpose(), RT).transpose()
            # points_ndc = geometry_utils.covert_world_points_to_pixel(points_ix.transpose(1,0), RT, Kndc=Kndc, use_cuda=False)
            gt_ray_distance = gt_ray_distance_finder.gt_ray_distance_function(
                points_ix, ray_dir_ix
            )
            pred_ray_distance = pred_distances[ray_px, ray_py]
            sdistance = np.linalg.norm(
                points_ix
                - camera_center[
                    None,
                ],
                axis=1,
            )
            if False:
                raw_intersection_ix = raw_intersections[ray_px, ray_py]
                non_zero_inds = np.where(raw_intersection_ix > 0)[0]
                if len(non_zero_inds) > 0:
                    raw_intersection_ix = raw_intersection_ix[non_zero_inds]
                else:
                    raw_intersection_ix = []

            if True:
                raw_intersection_ix = raw_intersection_inds[ray_px, ray_py]
                non_zero_inds = np.where(raw_intersection_ix > 0)[0]
                if len(non_zero_inds) > 0:
                    raw_intersection_ix = raw_intersection_ix[non_zero_inds].astype(int)

                    raw_intersection_ix = np.array(
                        [sdistance[k] for k in raw_intersection_ix]
                    )
                else:
                    raw_intersection_ix = []

            rendered_plot = RenderRayAndViz.plot_intersection_graph(
                sdistance,
                pred_distance=pred_ray_distance,
                gt_distance=gt_ray_distance,
                intersect_loc=raw_intersection_ix,
            )

            img_dot = draw_pixel_on_image(image * 255, ray_px, ray_py)
            img_dot = img_dot.astype(np.uint8)
            save_plot_path = osp.join(render_dir, f"plot_{ray_px}px_{ray_py}py.png")
            imageio.imsave(save_plot_path, rendered_plot)
            save_img_path = osp.join(render_dir, f"img_{ray_px}px_{ray_py}py.png")
            imageio.imsave(save_img_path, img_dot)
            if ix > NUM_RAYS:
                break

    @staticmethod
    def plot_intersection_graph(sdistance, pred_distance, gt_distance, intersect_loc):

        fig_size = (3, 6)
        fig, ax = plt_vis.subplots(plt, (1, 1), sz_y_sz_x=fig_size)

        ax.plot(sdistance, gt_distance, c="r", label="gt")
        ax.plot(sdistance, pred_distance, c="b", label="pred")
        if len(intersect_loc) > 0:
            ax.scatter(intersect_loc, intersect_loc * 0, marker="*")
        img = plt_vis.plt_formal_to_image(axis=True, legend=True)
        return img


def create_viz_html(render_dir, html_name):

    image_uuid_lst = os.listdir(render_dir)
    num_pages = len(image_uuid_lst)

    for px in range(num_pages):
        image_uuid = image_uuid_lst[px]
        image_uuid_dir = osp.join(render_dir, image_uuid)
        plot_files = list(
            osp.basename(k) for k in glob.glob(f"{image_uuid_dir}/img_*.png")
        )
        plot_files_suffix = list(
            k.replace(".png", "").replace("img_", "") for k in plot_files
        )

        html_name_px = html_name.split(".html")[0] + f"_p{px}" + ".html"
        fp = open(html_name_px, "w")
        fp.write("<html> <body>\n")

        page_header = ""
        page_header += "<table><tr>\n"
        for px2 in range(num_pages):
            html_name_px2 = html_name.split(".html")[0] + f"_p{px2}" + ".html"
            html_name_px2 = osp.basename(html_name_px2)
            if px == px2:
                page_header += f"<td>{px2}</td>"
            else:
                page_header += f"<td><a href=./{html_name_px2}>{px2}</a></td>"

        page_header += "</tr></table>\n"
        fp.write(page_header)

        html_base_dir = osp.dirname(html_name_px)
        render_px_dir = osp.join(render_dir, image_uuid)
        table_str = "<table>"
        relpath = osp.relpath(osp.join(render_dir, render_px_dir), html_base_dir)

        for suffix in plot_files_suffix:
            image_path = osp.join(relpath, f"img_{suffix}.png")
            tr_str = "<tr>"
            tr_str += f'<td> <img  height="256" src="{image_path}"/> </td> '
            image_path = osp.join(relpath, f"plot_{suffix}.png")

            tr_str += f'<td> <img  height="256" src="{image_path}"/> </td> '
            tr_str += "</tr>\n"

            table_str += tr_str

        table_str += "</table>"
        fp.write(table_str)
        fp.write(page_header)
        fp.write("</body></html>\n")
        fp.close()
    return


if __name__ == "__main__":
    from rgbd_drdf.utils import parse_args

    cfg = get_cfg_defaults()
    cmd_args = parse_args.parse_args()
    if cmd_args.cfg_file is not None:
        cfg.merge_from_file(cmd_args.cfg_file)
    if cmd_args.set_cfgs is not None:
        cfg.merge_from_list(cmd_args.set_cfgs)
    np.random.seed(0)
    renderer = RenderRayAndViz(cfg)

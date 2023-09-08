## Visualizes the rendered PCL logs
import glob
import os
import os.path as osp
import pickle as pkl
import subprocess

import imageio
import numpy as np
import torch
from fvcore.common.config import CfgNode
from loguru import logger
from trimesh.visual.color import ColorVisuals

from rgbd_drdf.renderer import (
    pyt3d_renderer
)
from rgbd_drdf.utils import geometry_utils, mesh_utils
from rgbd_drdf.utils.image import save_animated_webp, save_image_webp
from rgbd_drdf.utils.intersection_finder_utils import intersection_finder_drdf
from rgbd_drdf.utils.tensor_utils import copy2cpu as c2c
from rgbd_drdf.utils.tensor_utils import tensor_to_cuda as t2c

_C = CfgNode()

_C.PCL_LOG_DIR = osp.join(
    osp.dirname(osp.abspath(__file__)), "../", "cachedir", "eval2/raw/"
)
_C.PCL_RENDER_DIR = osp.join(
    osp.dirname(osp.abspath(__file__)), "../", "cachedir", "pcl_render_dir"
)

GT_EVAL_PATH_MATTERPORT = "rgbd_drdf/cachedir/eval2/raw/gt_eval_data"

_C.GIF_VIDEO = False
_C.TEST_EPOCH_NUMBER = -1

_C.HIGH_RES = False
_C.GIF_OUT = False


_C.DIR_NAME = ""
_C.SPLIT = "val"
_C.DATASET = "matterport"

import random

from PIL import Image



def get_cfg_defaults():
    return _C.clone()


ffmpeg_path = "/sw/pkgs/coe/o/ffmpeg/4.4.2/bin/ffmpeg"


def get_point_image_colors(points, RT, Kndc, image):
    points_ndc = geometry_utils.covert_world_points_to_pixel(
        points.transpose(1, 0), RT, Kndc=Kndc, use_cuda=True
    )

    image_th = t2c(image).permute(2, 0, 1)[None]
    xy_points = points_ndc[:2, :].permute(1, 0)
    # xy_points = xy_points[:,(1,0)] ## this is a temporary hack.
    xy_points = xy_points[None, None]
    point_colors = torch.nn.functional.grid_sample(
        image_th,
        xy_points,
    )
    point_colors = c2c(point_colors.squeeze()).transpose(1, 0)
    return point_colors


def create_viz_html(render_dir, html_name):
    rendered_dirs = [dir_name for dir_name in os.listdir(render_dir)]  # [0:10]
    items_per_page = 5
    num_pages = int(np.ceil(len(rendered_dirs) / items_per_page))
    image_names = [
        "image.png",
    ]

    for vx in range(10):
        image_names += [f"pred_view_{vx}.png"]

    for px in range(num_pages):
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
        render_subset_dirs = rendered_dirs[
            px * items_per_page : (px + 1) * items_per_page
        ]
        table_str = "<table>"
        for temp_dir in render_subset_dirs:
            relpath = osp.relpath(osp.join(render_dir, temp_dir), html_base_dir)
            tr_str = "<tr>"
            tr_str += "<td> uuid </td>"
            for image_key in image_names:
                tr_str += f"<td> {image_key} </td>"

            tr_str += "</tr>\n"
            tr_str += "<tr>"
            tr_str += f"<td> {temp_dir} </td>"

            for image_key in image_names:
                image_path = f"./{relpath}/{image_key}"
                tr_str += f'<td><img height="256" src="{image_path}" /> </td>'
            tr_str += "</tr>\n"
            table_str += tr_str
        table_str += "</table>\n"
        fp.write(table_str)
        fp.write(page_header)
        fp.write("</body></html>\n")
        fp.close()
    return


def create_video_viz_html(render_dir, html_name):
    rendered_dirs = [
        dir_name
        for dir_name in os.listdir(render_dir)
        if osp.isdir(osp.join(render_dir, dir_name))
    ]  # [0:10]
    items_per_page = 5
    num_pages = int(np.ceil(len(rendered_dirs) / items_per_page))
    image_names = [
        "image.png",
    ]

    for px in range(num_pages):
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
        render_subset_dirs = rendered_dirs[
            px * items_per_page : (px + 1) * items_per_page
        ]
        table_str = "<table>"
        for temp_dir in render_subset_dirs:
            relpath = osp.relpath(osp.join(render_dir, temp_dir), html_base_dir)
            tr_str = "<tr>"
            tr_str += "<td> uuid </td>"
            for image_key in ["video", "gif"]:
                tr_str += f"<td> {image_key} </td>"

            tr_str += "</tr>\n"
            tr_str += "<tr>"
            tr_str += f"<td> {temp_dir} </td>"
            uuid = temp_dir
            tr_str += f"<td>"
            tr_str += f'<video  width="540" controls autoplay muted loop> <source src="./val/{uuid}.mp4" type="video/mp4" /> </video> </td>'
            tr_str += f'<td><img height="256" src="./val/{uuid}.gif" /></td>'
            tr_str += "</tr>\n"
            table_str += tr_str
        table_str += "</table>\n"
        fp.write(table_str)
        fp.write(page_header)
        fp.write("</body></html>\n")
        fp.close()
    return


class RenderAndViz:
    def __init__(self, opts):
        render_dir = opts.PCL_RENDER_DIR
        name = opts.DIR_NAME
        split = opts.SPLIT

        ## read the npz paths ---

        ## use the pytorch3d wrapper to render sutff

        ## create html ---
        if "medium" in split:
            split = split.split("/")[1]

        self.split = split
        if opts.HIGH_RES:
            opts.PCL_LOG_DIR = opts.PCL_LOG_DIR.replace("eval", "high_res_eval")
            opts.PCL_RENDER_DIR = opts.PCL_RENDER_DIR.replace(
                "pcl_render_dir", "high_res_pcl_render_dir"
            )

        if opts.GIF_OUT:
            # opts.PCL_RENDER_DIR = opts.PCL_RENDER_DIR.replace(
            #     "pcl_render_dir_supp", "pcl_render_dir_supp_gif"
            # )
            opts.PCL_RENDER_DIR = opts.PCL_RENDER_DIR + "_gif"

        if opts.TEST_EPOCH_NUMBER >= 0:
            epoch_number = opts.TEST_EPOCH_NUMBER
            pcl_log_dir = osp.join(
                opts.PCL_LOG_DIR, opts.DIR_NAME, split + f"_{epoch_number}"
            )
            render_log_dir = osp.join(
                opts.PCL_RENDER_DIR, opts.DIR_NAME, split + f"_{epoch_number}"
            )
        else:
            pcl_log_dir = osp.join(opts.PCL_LOG_DIR, opts.DIR_NAME, split)
            render_log_dir = osp.join(opts.PCL_RENDER_DIR, opts.DIR_NAME, split)
        os.makedirs(render_log_dir, exist_ok=True)

        files = [k for k in glob.glob(f"{pcl_log_dir}/*.pkl")]
        files.sort()
        random.seed(0)
        random.shuffle(files)
        files = files[0:500]

        if False:
            selected_ids = [
                "82sE5b5pLXE_1f1625604f114171895fba7bdf84ce03_i1_0",
                # 'sKLMLpTHeUy_df0ddfb1212245bbbf95a6c94ed19b6c_i1_1',
                # '8WUmhLawc2A_d90dea9462374ee08a44f35e4505926e_i1_5'
            ]
            selected_ids = [
                "82sE5b5pLXE_62e1c55d62054d149d3d23431ffe4289_i1_4",
                "ULsKaCPVFJR_41e03824893747c4a334895c8edd6b05_i1_4",
                "82sE5b5pLXE_fe74bed2bbc04c95a32236cd127280e8_i1_5",
            ]
            if False:
                # supp_lst = 'matterport_high_res_uuid_file.lst'
                supp_lst = "paper_supp_figures/matterport_supp.lst"
                if opts.DATASET == "taskonomy":
                    supp_lst = "paper_supp_figures/taskonomy_supp.lst"

                with open(supp_lst) as f:
                    selected_ids = [line.rstrip("\n") for line in f]
                # selected_ids = ["ULsKaCPVFJR_41e03824893747c4a334895c8edd6b05_i1_4"]
            files = []
            for sel_id in selected_ids:
                # files.append(osp.join(pcl_log_dir, sel_id +".jpg"+".pkl"))
                files.append(osp.join(pcl_log_dir, sel_id + ".pkl"))

        if True:

            for nx, npz_file in enumerate(files):
                basename = osp.basename(npz_file)
                uuid = basename.replace(".pkl", "").replace(".jpg", "")
                render_dir = osp.join(render_log_dir, uuid)
                os.makedirs(render_dir, exist_ok=True)
                logger.info(f"{nx} Rendering {uuid}")

                if opts.GIF_OUT:
                    self.render_npz_gif(
                        npz_file,
                        render_dir,
                        dataset=opts.DATASET,
                        high_res=opts.HIGH_RES,
                    )
                else:
                    self.render_npz(
                        npz_file,
                        render_dir,
                        dataset=opts.DATASET,
                        high_res=opts.HIGH_RES,
                    )
                # if nx > 40:
                #     print("Only generating 100 visuals")
                #     break

        if opts.GIF_OUT:
            html_name = osp.join(
                opts.PCL_RENDER_DIR, opts.DIR_NAME, f"pred_video_{split}.html"
            )
            create_video_viz_html(render_dir=render_log_dir, html_name=html_name)
        else:
            html_name = osp.join(
                opts.PCL_RENDER_DIR, opts.DIR_NAME, f"pred_{split}.html"
            )

            create_viz_html(render_dir=render_log_dir, html_name=html_name)

    def render_npz(
        self,
        npz_path,
        render_dir,
        dataset,
        high_res=False,
    ):

        if not osp.exists(npz_path):
            logger.info(f"File does not exist  {npz_path}")
        try:
            with open(npz_path, "rb") as f:
                pred_data = pkl.load(f)
        except:
            logger.info("error reading pkl")
            return
        if "pixel_nerf" in npz_path or "finetune_val" in npz_path:
            base_name = osp.basename(npz_path)

            gt_data_path = osp.join(GT_EVAL_PATH_MATTERPORT, self.split, base_name)
            with open(gt_data_path, "rb") as f:
                gt_data = pkl.load(f)
            if high_res:
                pred_data["image_hr"] = gt_data["image_hr"]
                pred_data["image"] = gt_data["image"]
            else:
                pred_data["image"] = gt_data["image"]
            pred_data["RT"] = gt_data["RT"]
            pred_data["Kndc"] = gt_data["Kndc"]

        if False:
            pred_data = np.load(npz_path)

        RT = pred_data["RT"]
        Kndc = pred_data["Kndc"]
        visibility = pred_data["visibility"]
        pcl = pred_data["pcl"]
        image = pred_data["image"]

        point_colors = get_point_image_colors(pcl, RT, Kndc=Kndc, image=image)
        pcl_renderer_wrapper = pyt3d_renderer.Pyt3DWrapperPointCloud(
            image_size=(1080, 1280), device="cuda:0", dataset=dataset
        )

        rendered_images = pyt3d_renderer.render_pcl_w_normals_w_visiblity(
            pcl,
            visibility=visibility,
            image_point_colors=point_colors,
            RT=RT,
            py3d_renderer=pcl_renderer_wrapper,
        )
        if high_res:
            image = pred_data["image_hr"]

        imageio.imsave(osp.join(render_dir, "image.png"), image)
        for rx, rendered_img in enumerate(rendered_images):
            imageio.imsave(
                osp.join(render_dir, f"pred_view_{rx}.png"),
                (rendered_img * 255).astype(np.uint8),
            )

    def render_npz_gif(
        self,
        npz_path,
        render_dir,
        dataset,
        high_res=False,
    ):
        try:
            with open(npz_path, "rb") as f:
                pred_data = pkl.load(f)
        except:
            logger.info("error reading pkl")
            return

        video_outpath = render_dir + ".mp4"
        gif_outpath = render_dir + ".gif"

        if "image" in pred_data.keys():
            image = pred_data["image"]
            if "image_hr" in pred_data.keys():
                image = pred_data["image_hr"]
            imageio.imsave(osp.join(render_dir, "image.png"), 
                (image * 255).astype(np.uint8)
            )
        if osp.exists(video_outpath):
            return

        if "pixel_nerf" in npz_path or "finetune_val" in npz_path:
            base_name = osp.basename(npz_path)

            gt_data_path = osp.join(GT_EVAL_PATH_MATTERPORT, self.split, base_name)
            if osp.exists(gt_data_path):
                with open(gt_data_path, "rb") as f:
                    gt_data = pkl.load(f)
                if high_res:
                    pred_data["image_hr"] = gt_data["image_hr"]
                    pred_data["image"] = gt_data["image"]
                else:
                    pred_data["image"] = gt_data["image"]
                pred_data["RT"] = gt_data["RT"]
                pred_data["Kndc"] = gt_data["Kndc"]

        if False:
            pred_data = np.load(npz_path)

        RT = pred_data["RT"]
        Kndc = pred_data["Kndc"]
        visibility = pred_data["visibility"]
        pcl = pred_data["pcl"]
        image = pred_data["image"]

        point_colors = get_point_image_colors(pcl, RT, Kndc=Kndc, image=image)

        if dataset == "matterport":
            pcl_renderer_wrapper = pyt3d_renderer.Pyt3DWrapperPointCloud(
                image_size=(1080, 1280), device="cuda:0", dataset=dataset
            )
            camera_params = pcl_renderer_wrapper.get_matterport_camera_params()
        else:
            assert False, f"camera params for dataset not known. {dataset}"
        if True:
            rendered_images = (
                pyt3d_renderer.render_pcl_w_normals_w_visiblity_gif(
                    pcl,
                    visibility=visibility,
                    image_point_colors=point_colors,
                    RT=RT,
                    py3d_renderer=pcl_renderer_wrapper,
                    camera_params=camera_params,
                )
            )

            if "image_hr" in pred_data.keys():
                image = pred_data["image_hr"]
            imageio.imsave(osp.join(render_dir, "image.png"), (image * 255).astype(np.uint8))
            rendered_images = [(k * 255).astype(np.uint8) for k in rendered_images]
            for rx, rendered_img in enumerate(rendered_images):
                imageio.imsave(
                    osp.join(render_dir, f"frame_{rx:04d}.png"),
                    rendered_img,
                )
                # save_image_webp(osp.join(render_dir, "frame_{:04d}.webp".format(rx)),
                #     rendered_img,
                # )
        img_path = f"{render_dir}/frame_%04d.png"

        create_video_from_frames(img_path, video_outpath, fps=45)
        create_gif_from_frames(img_path, gif_outpath, fps=45)


# def create_gif(rendered_images, outputpath, fps):
#     save_animated_webp(outputpath, rendered_images, fps=fps)


def create_gif_from_frames(
    img_path,
    out_path,
    fps,
):
    command = [
        "ffmpeg ",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        img_path,
        "-filter_complex",
        '"scale=540:-1:flags=lanczos[x]; [x] split [a][b];[a] palettegen [p];[b][p] paletteuse" ',
        "-loop 0",
        out_path,
    ]
    print(" ".join(command))

    os.system(" ".join(command))
    return


def create_video_from_frames(
    img_path,
    out_path,
    fps,
):
    command = [
        "ffmpeg ",
        "-y ",
        "-r ",
        str(fps),
        " -i ",
        img_path,
        "-vcodec ",
        "libx264 ",
        "-crf ",
        "25 ",
        "-vf ",
        "scale=540:-1 " "-pix_fmt",
        "yuv420p",
        out_path,
    ]
    print(" ".join(command))
    # subprocess.run(command)
    os.system(" ".join(command))
    return


if __name__ == "__main__":
    from rgbd_drdf.utils import parse_args

    cfg = get_cfg_defaults()
    cmd_args = parse_args.parse_args()
    if cmd_args.cfg_file is not None:
        cfg.merge_from_file(cmd_args.cfg_file)
    if cmd_args.set_cfgs is not None:
        cfg.merge_from_list(cmd_args.set_cfgs)

    renderer = RenderAndViz(cfg)

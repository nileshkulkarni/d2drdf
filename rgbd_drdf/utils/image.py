import cv2
import numpy as np
import torch
from PIL import Image

from . import tensor_utils


def resize_img(img, scale_factor, nearest=False):
    new_size = (np.round(np.array(img.shape[:2]) * scale_factor)).astype(int)
    if nearest:
        new_img = cv2.resize(
            img, (new_size[1], new_size[0]), interpolation=cv2.INTER_NEAREST
        )
    else:
        new_img = cv2.resize(img, (new_size[1], new_size[0]))
        # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [
        new_size[0] / float(img.shape[0]),
        new_size[1] / float(img.shape[1]),
    ]
    return new_img, actual_factor


def peturb_bbox(bbox, pf=0, jf=0):
    """
    Jitters and pads the input bbox.
    Args:
        bbox: Zero-indexed tight bbox.
        pf: padding fraction.
        jf: jittering fraction.
    Returns:
        pet_bbox: Jittered and padded box. Might have -ve or out-of-image coordinates
    """
    pet_bbox = [coord for coord in bbox]
    bwidth = bbox[2] - bbox[0] + 1
    bheight = bbox[3] - bbox[1] + 1

    pet_bbox[0] -= (pf * bwidth) + (1 - 2 * np.random.random()) * jf * bwidth
    pet_bbox[1] -= (pf * bheight) + (1 - 2 * np.random.random()) * jf * bheight
    pet_bbox[2] += (pf * bwidth) + (1 - 2 * np.random.random()) * jf * bwidth
    pet_bbox[3] += (pf * bheight) + (1 - 2 * np.random.random()) * jf * bheight

    return pet_bbox


def save_image_webp(save_path, image):
    im = Image.fromarray(image)
    im.save(save_path, format="webp")



def save_animated_webp(save_path, images, fps):
    invfps = int(1000 / 30)
    breakpoint()
    pil_imgs = [Image.fromarray(img) for img in images]
    pil_imgs[0].save(
        save_path, save_all=True, duration=invfps, append_images=pil_imgs[1:], loop=0
    )
    webp.save_images(pil_imgs, "anim.webp", fps=10, lossless=False, quality=70)

    return


def square_bbox(bbox):
    """
    Converts a bbox to have a square shape by increasing size along non-max dimension.
    """
    sq_bbox = [int(round(coord)) for coord in bbox]
    bwidth = sq_bbox[2] - sq_bbox[0] + 1
    bheight = sq_bbox[3] - sq_bbox[1] + 1
    maxdim = float(max(bwidth, bheight))

    dw_b_2 = int(round((maxdim - bwidth) / 2.0))
    dh_b_2 = int(round((maxdim - bheight) / 2.0))

    sq_bbox[0] -= dw_b_2
    sq_bbox[1] -= dh_b_2
    sq_bbox[2] = sq_bbox[0] + maxdim - 1
    sq_bbox[3] = sq_bbox[1] + maxdim - 1

    return sq_bbox


def crop(img, bbox, bgval=0):
    """
    Crops a region from the image corresponding to the bbox.
    If some regions specified go outside the image boundaries, the pixel values are set to bgval.
    Args:
        img: image to crop
        bbox: bounding box to crop
        bgval: default background for regions outside image
    """
    bbox = [int(round(c)) for c in bbox]
    bwidth = bbox[2] - bbox[0] + 1
    bheight = bbox[3] - bbox[1] + 1

    im_shape = np.shape(img)
    im_h, im_w = im_shape[0], im_shape[1]

    nc = 1 if len(im_shape) < 3 else im_shape[2]

    img_out = np.ones((bheight, bwidth, nc)) * bgval
    x_min_src = max(0, bbox[0])
    x_max_src = min(im_w, bbox[2] + 1)
    y_min_src = max(0, bbox[1])
    y_max_src = min(im_h, bbox[3] + 1)

    x_min_trg = x_min_src - bbox[0]
    x_max_trg = x_max_src - x_min_src + x_min_trg
    y_min_trg = y_min_src - bbox[1]
    y_max_trg = y_max_src - y_min_src + y_min_trg

    img_out[y_min_trg:y_max_trg, x_min_trg:x_max_trg, :] = img[
        y_min_src:y_max_src, x_min_src:x_max_src, :
    ]
    return img_out


import cv2


def interpolate_depth_numpy(depth, xy_coords):
    # sampled_depth = np.empty(xy_coords.shape[1], dtype=np.float32)

    img_h, img_w = depth.shape[1], depth.shape[2]
    xy_coords = xy_coords * 1.0
    assert xy_coords.shape[1] == img_w * img_h, "this only works for this setting."
    xy_coords[..., 0] = img_w * (xy_coords[..., 0] + 1) / 2
    xy_coords[..., 1] = img_h * (xy_coords[..., 1] + 1) / 2
    invalid = (
        (xy_coords[..., 0] < 0)
        + (xy_coords[..., 0] > img_w - 1)
        + (xy_coords[..., 1] < 0)
        + (xy_coords[..., 1] > img_h - 1)
    )
    invalid = (invalid > 0).reshape(-1)
    xy_coords[..., 0] = np.clip(xy_coords[..., 0], 0, img_w - 1)
    xy_coords[..., 1] = np.clip(xy_coords[..., 1], 0, img_w - 1)
    xy_coords = xy_coords.reshape(img_h, img_w, 2)
    sampled_depth = cv2.remap(
        depth[0], xy_coords[..., 0], xy_coords[..., 1], cv2.INTER_NEAREST
    )
    sampled_depth = sampled_depth.reshape(
        -1,
    )
    sampled_depth = sampled_depth * (1 - invalid) + invalid * 0
    # sampled_depth = bilinear_interpolate_numpy(depth[0].astype(np.float32), xy_coords[0].astype(np.float32))
    return sampled_depth[None, None]


@torch.jit.script
def interpolate_depth_torch(depth, xy_coords):
    sampled_depth = torch.nn.functional.grid_sample(
        depth, xy_coords, mode="nearest", align_corners=False
    )
    return sampled_depth


def interpolate_depth(depth, xy_coords):
    is_numpy = False
    if type(depth) == np.ndarray:
        is_numpy = True
        depth = torch.from_numpy(depth)
        xy_coords = torch.from_numpy(xy_coords)

    is_batched = True
    if len(depth.shape) == 3:
        depth = depth[None]
        xy_coords = xy_coords[None]
        is_batched = False

    sampled_depth = interpolate_depth_torch(depth, xy_coords)

    if not is_batched:
        sampled_depth = sampled_depth[0]

    if is_numpy:
        sampled_depth = tensor_utils.tensor_to_numpy(sampled_depth)

    return sampled_depth


import matplotlib.pyplot as plt


def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    depth_colored = 255 * plt.cm.plasma(depth_relative)[:, :, :3]  # H, W, C
    depth_colored = depth_colored.astype(np.uint8)
    return depth_colored


def draw_points_on_img(
    img, pixel_locs, draw_index=False, color=(0, 0, 255), alpha=1.0, size=3
):
    img_size = np.array([img.shape[1], img.shape[0]])
    pixel_locs = (pixel_locs / 2 + 0.5) * img_size[
        None,
    ]
    img = img * 1
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    overlay_img = img.copy()
    for px, pixel_loc in enumerate(pixel_locs):
        overlay_img = cv2.circle(
            overlay_img, (int(pixel_loc[0]), int(pixel_loc[1])), size, color, -1
        )
        if draw_index:
            overlay_img = cv2.putText(
                overlay_img,
                f"{px}",
                ((int(pixel_loc[0]), int(pixel_loc[1]))),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),
                4,
            )
    image_new = cv2.addWeighted(overlay_img, alpha, img, 1 - alpha, 0)
    return image_new


def get_point_validity(xy_ndc):
    valid_x = np.abs(xy_ndc[:, 0]) < 1
    valid_y = np.abs(xy_ndc[:, 1]) < 1
    valid = valid_x * valid_y
    return valid


def get_depth_validity(depth):
    return np.abs(depth) > 1e-4

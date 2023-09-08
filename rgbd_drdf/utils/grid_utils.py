import pdb

import numpy as np
import torch
from numba import jit


def create_pixel_aligned_grid(resX, resY, resZ, b_min, b_max, transform=None):
    """
    Create a pixel aligned grid.

    """

    length = b_max - b_min
    xy_coords = np.mgrid[
        :resX,
        :resY,
    ]
    xy_coords = xy_coords.reshape(2, -1)
    xy_matrix = np.eye(3)
    coords_matrix = np.eye(4)

    coords_matrix[0, 0] = length[0] / resX
    coords_matrix[1, 1] = length[1] / resY
    coords_matrix[2, 2] = length[2] / resZ

    res = np.array([resX, resY, resZ])

    xy_matrix[0, 0] = length[0] / resX
    xy_matrix[1, 1] = length[1] / resY
    xy_matrix[0:2, 2] = b_min[
        0:2,
    ]
    xy_coords = np.matmul(xy_matrix[:2, :2], xy_coords) + xy_matrix[:2, 2:3]
    depths = np.mgrid[:resZ] * length[2] / resZ + b_min[2]
    depths = depths[None, None, :].repeat(resX, axis=0).repeat(resY, axis=1)
    xy_coords = xy_coords.reshape(2, resX, resY)
    coords = []
    for dx in range(resZ):
        coords.append(
            np.concatenate(
                [xy_coords * depths[None, :, :, dx], depths[None, :, :, dx]], axis=0
            )
        )
    coords = np.stack(coords, axis=-1)

    if transform is not None:
        coords = np.matmul(transform[:3, :3], coords) + transform[:3, 3:4]
        coords_matrix = np.matmul(transform, coords_matrix)
    coords = coords.reshape(3, resX, resY, resZ)
    coords = torch.FloatTensor(coords)
    return coords


def sample_img_grid_ndc(img_size, jitter_grid=False):
    img_h, img_w = img_size[1], img_size[0]
    x = np.linspace(0, img_w - 1, num=img_w).astype(np.float32)
    y = np.linspace(0, img_h - 1, num=img_h).astype(np.float32)
    ys, xs = np.meshgrid(y, x)
    # xs, ys = meshgrid(y, x)

    if jitter_grid:
        # max_x_jitter = 0.5 / img_w
        # max_y_jitter = 0.5 / img_h
        max_x_jitter = 1
        max_y_jitter = 1

        jitter_x = np.random.uniform(-max_x_jitter, max_x_jitter, size=xs.shape)
        jitter_y = np.random.uniform(-max_y_jitter, max_y_jitter, size=ys.shape)
        xs += jitter_x
        ys += jitter_y
    if img_h != img_w:
        "rect ndc grid"
        img_s = min(img_h, img_w)
        coordinates = np.stack([xs / img_s, ys / img_s], axis=0)
        s = img_w / img_h
        coordinates[0] = coordinates[0] * 2 - s
        coordinates[1] = coordinates[1] * 2 - 1
        ## assumes heigh is < width
        coordinates[0] = np.clip(coordinates[0], -s, s)
        coordinates[1] = np.clip(coordinates[1], -1, 1)
    else:
        coordinates = np.stack([xs / (img_w - 1), ys / (img_h - 1)], axis=0)
        coordinates = coordinates * 2 - 1
        coordinates = np.clip(coordinates, -1, 1)
    return coordinates


@jit(nopython=True)
def meshgrid(
    x,
    y,
):
    xx = np.empty(shape=(x.size, y.size), dtype=x.dtype)
    yy = np.empty(shape=(x.size, y.size), dtype=y.dtype)

    for j in range(y.size):
        for k in range(x.size):
            xx[j, k] = k  # change to x[k] if indexing xy
            yy[j, k] = j  # change to y[j] if indexing xy
    return xx, yy

import pdb

import numpy as np
import torch

from . import tensor_utils


def convert_to_ndc(points, K, projection_fn, zmin=0.1, zmax=10.0):
    """
    Convert points to normalized device coordinates.
    args:   points: (B, 3, N)
            K: (B, 3, 3)
            projection_fn: function that takes in points and returns projected points
    returns:
            points_ndc: (B, 2, N)
            normalized_depth: (B, 1, N)
    """

    xyz = projection_fn(
        points,
        K,
    )
    m = 2.0 / (zmax - zmin)
    b = -2.0 * zmin / (zmax - zmin) - 1
    xyz[:, 2, :].mul_(m).add_(b)
    return xyz


@torch.jit.script
def transform_points_torch(points, RT):
    pointsCam = torch.bmm(RT, points)
    return pointsCam


def transform_points(points, RT):
    batched = True
    if len(points.shape) == 2:
        points = points[None]
        RT = RT[None]
        batched = False

    if type(points) == np.ndarray:
        points = np.concatenate([points, points[:, 0:1, :] * 0 + 1], axis=1)
        pointsCam = np.matmul(RT, points)
    else:
        points = torch.cat([points, points[:, 0:1, :] * 0 + 1], dim=1)
        # pointsCam = torch.bmm(RT, points)
        pointsCam = transform_points_torch(points, RT)
    pointsCam = pointsCam[:, 0:3, :]
    if not batched:
        pointsCam = pointsCam[0]

    return pointsCam


@torch.jit.script
def prespective_transform_torch(points, K):
    img_points = torch.bmm(K, points)
    z = points[:, 2:3, :]
    z_sign = z.sign()
    eps = 1e-4
    z_sign[z == 0] = 1
    z = z_sign * z.abs().clamp(min=eps)
    xy = img_points[:, :2, :] / z
    xyz = torch.cat([xy, z], 1)
    return xyz


def prespective_transform_np(points, K):
    img_points = np.matmul(K, points)
    z = points[:, 2:3, :]
    z_sign = np.sign(z)
    eps = 1e-4
    z_sign[z == 0] = 1
    zabs = np.abs(z)
    np.clip(zabs, a_min=eps, a_max=None, out=zabs)
    z = z_sign * zabs
    xy = img_points[:, :2, :] / z
    xyz = np.concatenate([xy, z], axis=1)
    return xyz


def prespective_transform(points, K):
    # points = points/points[:, 2, None, :]
    batched = True
    is_numpy = False
    if len(points.shape) == 2:
        points = points[None]
        K = K[None]
        batched = False

    if type(points) == np.ndarray:
        is_numpy = True
        xyz = prespective_transform_np(points, K)
        # K = torch.FloatTensor(K * 1)
        # points = torch.FloatTensor(points * 1)
    else:
        xyz = prespective_transform_torch(points, K)

    if not batched:
        xyz = xyz[0]

    return xyz


def apply_log_transform(tsdf):
    sgn = torch.sign(tsdf)
    out = torch.log(torch.abs(tsdf) + 1)
    out = sgn * out
    return out


def covert_world_points_to_pixel(coords, RT, Kndc, use_cuda=False):
    batch = True
    if len(coords.shape) == 2:
        batch = False
        coords = coords[None]
        RT = RT[None]
        Kndc = Kndc[None]
    np_array = False
    if type(coords) == np.ndarray:
        np_array = True

    if np_array and (not use_cuda):
        points_cam = transform_points(coords, RT)
        xyz = prespective_transform(points_cam, Kndc)
    else:
        coords = tensor_utils.tensor_to_cuda(coords, use_cuda)
        RT = tensor_utils.tensor_to_cuda(RT, use_cuda)
        Kndc = tensor_utils.tensor_to_cuda(Kndc, use_cuda)

        points_cam = transform_points(coords, RT)
        xyz = prespective_transform(points_cam, Kndc)

    if not batch:
        xyz = xyz[0]

    return xyz


def convert_pixel_to_world_points(coords, RT, Kndc):

    batch = True

    if len(coords.shape) == 2:
        batch = False
        coords = coords[None]
        RT = RT[None]
        Kndc = Kndc[None]
    np_array = False
    if type(coords) == np.ndarray:
        np_array = True

    if np_array:
        invK = np.linalg.inv(Kndc)
        invRT = np.linalg.inv(RT)
        coords = np.matmul(invK, coords)
        coords = transform_points(coords, invRT)
    else:
        invK = torch.inverse(Kndc)
        invRT = torch.inverse(RT)
        coords = torch.bmm(
            invK,
            coords,
        )
        coords = transform_points(coords, invRT)
    if not batch:
        coords = coords[0]

    return coords


def convert_depth_pcl(
    depth,
    RT,
    kNDC,
    use_cuda=False,
    return_xyz=False,
    jitter_grid=False,
    hw_scalar=1.0,
    interp_mode="nearest",
):
    from . import grid_utils

    img_h, img_w = depth.shape[0], depth.shape[1]
    img_h = int(img_h * hw_scalar)
    img_w = int(img_w * hw_scalar)
    img_grid = grid_utils.sample_img_grid_ndc(
        [img_w, img_h], jitter_grid=jitter_grid
    )  ## (2, img_w, img_h)
    if img_h != img_w:
        s = img_w / img_h
        img_grid_normalized = img_grid * 1
        img_grid_normalized[0] = img_grid[0] / s
        ## grid sample expects the image coordinates with -1 to 1
    else:
        img_grid_normalized = img_grid

    img_grid_th = tensor_utils.tensor_to_cuda(img_grid_normalized, use_cuda)
    depth_th = tensor_utils.tensor_to_cuda(depth, use_cuda)
    depth_th = torch.nn.functional.grid_sample(
        depth_th[None, None],
        img_grid_th.permute(1, 2, 0)[
            None,
        ],
        mode=interp_mode,
        align_corners=False,
    )[0, 0]
    depth = depth_th.data.numpy()
    ndc_pts = np.concatenate(
        [
            img_grid,
            1
            + 0
            * depth[
                None,
            ],
        ],
        axis=0,
    )
    points = (
        ndc_pts
        * depth[
            None,
        ]
    )
    valid_points = depth > 0.05
    valid_points = valid_points.reshape(-1)
    points = points.reshape(3, -1)

    if use_cuda:
        points = tensor_utils.tensor_to_cuda(
            points, use_cuda, tensor_type=torch.FloatTensor
        )
        valid_points = tensor_utils.tensor_to_cuda(
            valid_points, use_cuda, tensor_type=torch.ByteTensor
        )
        RT = tensor_utils.tensor_to_cuda(RT, use_cuda, tensor_type=torch.FloatTensor)
        kNDC = tensor_utils.tensor_to_cuda(
            kNDC, use_cuda, tensor_type=torch.FloatTensor
        )
    points_world = convert_pixel_to_world_points(points, RT, kNDC)
    points_world = points_world.transpose(1, 0)
    if return_xyz:
        ndc_pts = ndc_pts.transpose(1, 2, 0)
        return points_world, valid_points, ndc_pts
    else:
        return points_world, valid_points


def convert_ndc_to_pixels(ndc, img_size):
    """
    ndc: ... x 2
    """
    coords = ndc[..., 0:2] * 1
    coords[..., 0] = (coords[..., 0] * 0.5 + 0.5) * img_size[0]
    coords[..., 1] = (coords[..., 1] * 0.5 + 0.5) * img_size[1]
    return coords

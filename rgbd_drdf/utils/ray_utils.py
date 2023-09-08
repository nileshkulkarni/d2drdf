import pdb

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from numpy.core.fromnumeric import clip

from ..utils import geometry_utils
from ..utils import plt_vis as plt_vis_utils


def signed_zero_clip(data, a_max=None, a_min=None):
    data_sign = np.sign(data)
    data_sign[data_sign == 0] = 1.0
    data_clip = data_sign * np.clip(np.abs(data), a_max=a_max, a_min=a_min)
    return data_clip


"""
Basic function to compute distance along the ray given a mesh, points, ray_dir
"""
def basic_distance_along_ray(
    mesh: trimesh.Trimesh,
    points: np.array,
    ray_dirs: np.array,
    mesh_intersector=None,
    embree: bool = True,
    clip_distance: float = None,
):

    from trimesh.ray.ray_pyembree import RayMeshIntersector

    if mesh_intersector is None:
        mesh_intersector = RayMeshIntersector(mesh)

    def compute_distance_along_ray(points, rays, clip_distance=None):
        locations, index_ray, index_tri = mesh_intersector.intersects_location(
            points, rays, multiple_hits=False
        )
        intersect_locations = points * 0
        intersect_locations[index_ray] = locations
        valid_intersects = points[:, 0] * 0
        valid_intersects[index_ray] = 1
        tri_ids = (points[:, 0] * 0 - 1).astype(int)
        tri_ids[index_ray] = index_tri
        distance = np.linalg.norm(points - intersect_locations, axis=1)
        distance = distance * valid_intersects + (1 - valid_intersects) * 0

        if clip_distance is not None:
            assert type(clip_distance) == float, "clip distance should be of type float"
            invalid_int = (1 - valid_intersects).astype(bool)
            intersect_locations[invalid_int, :] = (
                points[invalid_int, :] + rays[invalid_int, :] * clip_distance
            )
            valid_intersects[invalid_int] += 1.0
            distance[invalid_int] = clip_distance

        return distance, intersect_locations, valid_intersects, tri_ids

    dist, int_loc, valid_int, tri_ids = compute_distance_along_ray(
        points, ray_dirs, clip_distance=clip_distance
    )
    return dist, int_loc, valid_int, tri_ids


def get_scene_distances(
    mesh: trimesh.Trimesh,
    points: np.array,
):

    closest_point, distance, triangle_id = trimesh.proximity.closest_point(mesh, points)
    return distance, closest_point


def get_special_ray_distances(
    mesh: trimesh.Trimesh,
    points: np.array,
    rays: np.array,
    unsigned_ray_dist: bool = False,
    signed: bool = False,
    mesh_intersector=None,
    **kwargs,
):

    from trimesh.ray.ray_pyembree import RayMeshIntersector

    if mesh_intersector is None:
        assert mesh is not None, "mesh input is necessary"
        mesh_intersector = RayMeshIntersector(mesh)

    dist, int_loc, valid_int, tri_ids = basic_distance_along_ray(
        mesh, points, rays, mesh_intersector, **kwargs
    )

    if unsigned_ray_dist:
        dist_bid, int_loc_bid, valid_int_bid, tri_ids_bid = basic_distance_along_ray(
            mesh, points, -1 * rays, mesh_intersector, **kwargs
        )
        min_selector = (dist + (1 - valid_int) * 10) > (
            dist_bid + (1 - valid_int_bid) * 10
        )
        min_selector = min_selector.astype(np.float32)
        int_loc = (
            int_loc * (1 - min_selector[:, None]) + min_selector[:, None] * int_loc_bid
        )
        valid_int = np.logical_or(valid_int, valid_int_bid)
        tri_ids = tri_ids * (1 - min_selector) + tri_ids * min_selector
        dist = dist * (1 - min_selector) + min_selector * dist_bid
    elif signed:
        dist_bid, int_loc_bid, valid_int_bid, tri_ids_bid = basic_distance_along_ray(
            mesh, points, -1 * rays, mesh_intersector, **kwargs
        )
        min_selector = (dist + (1 - valid_int) * 10) > (
            dist_bid + (1 - valid_int_bid) * 10
        )
        min_selector = min_selector.astype(np.float32)
        int_loc = (
            int_loc * (1 - min_selector[:, None]) + min_selector[:, None] * int_loc_bid
        )
        valid_int = np.logical_or(valid_int, valid_int_bid)
        tri_ids = tri_ids * (1 - min_selector) + tri_ids * min_selector
        dist = dist * (1 - min_selector) + min_selector * (-1) * dist_bid

    return dist, int_loc, valid_int, tri_ids


def get_camera_ray_dir(points: np.array, RT: np.array, return_hitting_depth=False):
    """[summary]

    Args:
        points (np.array): N x 3 matrix
        RT (np.array): 4 x 4  matrix

    Returns:
        [type]: [description]
    """

    invRT = np.linalg.inv(RT)
    pts_cam = geometry_utils.transform_points(points.transpose(), RT).transpose()
    hitting_depth = pts_cam[:, 2]
    hitting_depth_clip = signed_zero_clip(hitting_depth, a_min=1e-4, a_max=None)
    local_ray_dir = pts_cam / hitting_depth_clip[:, None]
    local_ray_dir = local_ray_dir / (
        1e-5 + np.linalg.norm(local_ray_dir, axis=1)[:, None]
    )

    start_point = pts_cam - (1 + hitting_depth[:, None]) * local_ray_dir
    world_start_points = geometry_utils.transform_points(
        start_point.transpose(), invRT
    ).transpose()
    ray_dir = 1 * (points - world_start_points)
    ray_dir = ray_dir / (1e-8 + np.linalg.norm(ray_dir, axis=1)[:, None])
    # ray_dir = ray_dir / np.linalg.norm(ray_dir, axis=1)[:, None]
    if return_hitting_depth:
        return ray_dir, hitting_depth
    else:
        return ray_dir


def get_axis_ray_dir(points, axis="X"):
    ray_dir = points * 0

    if axis == "X":
        ray_dir[..., 0] = 1
    elif axis == "Y":
        ray_dir[..., 1] = 1
    elif axis == "Z":
        ray_dir[..., 2] = 1
    else:
        assert False, "incorrect ray dir only supports X, Y , Z"

    return ray_dir


"""
points : 3 x N
RT : 4 x 4
"""
def convert_pts_to_rays2(points, RT):
    pts_cam = geometry_utils.transform_points(points, RT)
    hitting_depth = pts_cam[2, :]
    hitting_depth_clip = signed_zero_clip(hitting_depth, a_min=1e-4, a_max=None)
    local_ray_dir = pts_cam / hitting_depth_clip[None, :]
    return local_ray_dir, hitting_depth


def convert_rays_to_world2(pts, invRT):
    pts = geometry_utils.transform_points(pts, invRT)
    return pts


def compute_all_interesections_point(point, mesh_intersector, RT):
    ## point ## (3)
    ## Works on  single 3D point.
    ## this function needs to work recursively to find all intersections.\
    int_locs = []
    int_dists = []
    point = point[None]
    RT = RT.numpy()
    while True:
        ray_dir = get_camera_ray_dir(
            point,
            RT,
        )
        temp_point = point + 0.05 * ray_dir

        dist, int_loc, valid_int, tri_ids = get_special_ray_distances(
            mesh=None,
            points=temp_point,
            rays=ray_dir,
            mesh_intersector=mesh_intersector,
        )
        point = int_loc
        if (len(valid_int) > 0) and valid_int[0] > 0:
            int_locs.append(int_loc)
            int_dists.append(dist)
        else:
            break
    if len(int_dists) > 0:
        int_dists = np.cumsum(np.concatenate(int_dists))
    else:
        int_dists = np.array(int_dists)

    return int_dists


def compute_all_intersection(points, mesh, RT, max_ints=5):
    mesh_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(
        trimesh.Trimesh(vertices=mesh["verts"], faces=mesh["faces"])
    )
    all_int_dists = []
    for px in range(points.shape[1]):
        zero_ints = np.zeros(max_ints)
        point = points[:, px]
        int_dists = compute_all_interesections_point(point, mesh_intersector, RT)
        nints = min(max_ints, len(int_dists))
        zero_ints[0 : min(max_ints, len(int_dists))] = int_dists[0:nints]
        all_int_dists.append(zero_ints)
    all_int_dists = np.stack(all_int_dists)
    return all_int_dists


def compute_view_frustrum(RT: np.array, Kndc: np.array, z_max):
    """[summary]

    Args:
        points (np.array): [N x 3 array of points]
        rays (np.array): [N x 3 array of ray dirs] --> world to cam
        RT (np.array): [4 x 4 ext matrix]
        Kndc (np.array): [3 x 3 int matrix]
    """
    ## complete this function
    im_h = 2
    im_w = 2

    view_frust_pts = np.array(
        [
            (np.array([0, -1, -1, 1, 1]) - Kndc[0, 2])
            * np.array([0, z_max, z_max, z_max, z_max])
            / Kndc[0, 0],
            (np.array([0, -1, 1, -1, 1]) - Kndc[1, 2])
            * np.array([0, z_max, z_max, z_max, z_max])
            / Kndc[1, 1],
            np.array([0, z_max, z_max, z_max, z_max]),
        ]
    )

    view_frust_pts = geometry_utils.transform_points(view_frust_pts, np.linalg.inv(RT))
    return view_frust_pts


def signed_divide(tensor1, tensor2):
    tensor2_sign = np.sign(tensor2)
    eps = 1e-5
    tensor2_sign[tensor2 == 0] = 1
    tensor2 = tensor2_sign * np.clip(np.abs(tensor2), a_min=eps, a_max=None)
    return tensor1 / tensor2


def plot_distance_func(z, dist_func, marker=".", color="r", **kwargs_plot):
    plt.scatter(z, dist_func, marker=marker, color=color)
    img = plt_vis_utils.plt_formal_to_image(**kwargs_plot)
    return img


class GroundTruthRayDistance:
    def __init__(self, mesh):
        from trimesh.ray.ray_pyembree import RayMeshIntersector

        self.mesh_intersector = RayMeshIntersector(mesh)
        self.mesh = mesh
        return

    def gt_ray_distance_function(self, points, ray_dir):
        distance, int_loc, valid_int, tri_ids = get_special_ray_distances(
            mesh=self.mesh,
            points=points,
            rays=ray_dir,
            signed=True,
            mesh_intersector=self.mesh_intersector,
        )
        return distance

import copy
import pdb

import numpy as np
import torch
import trimesh
from loguru import logger

from . import (
    geometry_utils,
    grid_utils,
    intersection_finder_utils,
    mesh_utils,
    ray_utils,
)
from .tensor_utils import Struct
from .tensor_utils import tensor_to_cuda as t2c


def gen_depth_from_net(
    opts,
    net,
    device,
    meta_data,
    intersection_finder,
    resolution=128,
    catch_exceptions=True,
):

    depths = None
    if "depth" in meta_data:
        depths = meta_data["depth"][None]
    image_tensor = meta_data["img"].to(device=device)
    Kndc = meta_data["Kndc"].to(device=device)
    net.filter_images(
        images=image_tensor[
            None,
        ],
        depths=depths,
    )
    if "RT" in meta_data.keys():
        RT = meta_data["RT"].to(device=device)
    else:
        RT = None
    if "transforms" in meta_data.keys():
        transforms = meta_data["transforms"]
    else:
        transforms = None

    if True:
        coords, ray_dir = create_pixel_query_grid(
            resolution=resolution, z_max=meta_data["z_max"], RT=RT, Kndc=Kndc
        )
        coords = torch.cat([coords, ray_dir * 0], dim=0)

    distances, depths = reconstruction_vol(
        net=net,
        cuda=device,
        Kndc=Kndc,
        RT=RT,
        coords=coords,
    )

    coords = coords.transpose(1, 0)
    coords = coords.reshape(resolution, resolution, resolution, 6).cpu().numpy()

    distances = distances.reshape(resolution, resolution, resolution)
    depths = depths.reshape(resolution, resolution, resolution)
    distances = distances.cpu().numpy()
    depths = depths.cpu().numpy()

    try:
        _, results = intersection_finder_utils.double_batch_compute_intersections(
            distances,
            depths,
            intersection_finder,
            num_workers=opts.RAY.NUM_WORKERS,
            add_zeros=True,
        )
    except Exception as e:
        if catch_exceptions:
            print("Exception in -- ignore generation & distance computation")
            return None
        else:
            raise e
    interesection_bumps_pred = []
    intersection_coords = []
    for result in results:
        non_zero = np.where(result[0] > 0)[0]
        if len(non_zero) > 0:
            interesection_bumps_pred.append(result[0][non_zero[0]])
        else:
            interesection_bumps_pred.append(0)
        if len(result[2]) > 0:
            intersection_coords.extend(
                [result[2][0]]
            )  ## this is only keeping the first one.

    interesection_bumps_pred = np.stack(interesection_bumps_pred)
    interesection_bumps_pred = interesection_bumps_pred.reshape(
        resolution, resolution, -1
    )

    intersection_coords = np.array(intersection_coords)
    # yapf: disable
    try:
        if len(intersection_coords) > 0:
            intersection_locs = coords[(intersection_coords[:,0], intersection_coords[:,1], intersection_coords[:,2])]
            point_locs =  intersection_locs[:, :3]
            depth_map = interesection_bumps_pred.transpose(1,0, 2)[:,:,0]
        else:
            depth_map = np.zeros((resolution, resolution))
    except IndexError:
        depth_map = np.zeros((resolution, resolution))
        breakpoint()

    # yapf: enable
    return depth_map


def pad_zeros(data, size=5):
    if len(data) > size:
        data = data[:size]
    elif len(data) > 0:
        pad_size = size - len(data)
        pads = np.zeros(pad_size) - 1
        data = np.concatenate([pads, data])
    else:
        pads = np.zeros(size) - 1
        data = pads

    return data


def gen_pcl_from_mesh(
    opts,
    meta_data,
    intersection_finder,
    return_mesh=False,
    return_pcl=False,
    return_visiblity=False,
):
    device = "cpu"
    if "meshes" in meta_data.keys():
        meshes = meta_data["meshes"]
    else:
        meshes = None

    Kndc = meta_data["Kndc"].to(device=device)
    RT = meta_data["RT"].to(device=device)
    transforms = None
    if isinstance(opts.MODEL.RESOLUTION, list):
        resolution_x, resolution_y, resolution_z = (
            opts.MODEL.RESOLUTION[0],
            opts.MODEL.RESOLUTION[1],
            opts.MODEL.RESOLUTION[2],
        )
    else:
        resolution_x, resolution_y, resolution_z = (
            opts.MODEL.RESOLUTION,
            opts.MODEL.RESOLUTION,
            opts.MODEL.RESOLUTION,
        )
        resolution = resolution_x

    gt_intersector = ray_utils.GroundTruthRayDistance(meshes)
    coords, ray_dir = create_pixel_query_grid(
        resolution=resolution, z_max=meta_data["z_max"], RT=RT, Kndc=Kndc
    )
    coords = coords.transpose(1, 0).numpy()
    ray_dir = ray_dir.transpose(1, 0).numpy()
    gt_distance = gt_intersector.gt_ray_distance_function(
        points=coords, ray_dir=ray_dir
    )
    xyz = geometry_utils.covert_world_points_to_pixel(coords.transpose(), RT, Kndc)
    depths = xyz[2].numpy()
    gt_distance = gt_distance.reshape(resolution, resolution, resolution)
    depths = depths.reshape(resolution, resolution, resolution)
    coords = coords.reshape(resolution, resolution, resolution, 3)

    _, results = intersection_finder_utils.double_batch_compute_intersections(
        gt_distance,
        depths,
        intersection_finder,
        add_zeros=True,
        direction=True,
        num_workers=opts.RAY.NUM_WORKERS,
    )

    interesection_bumps_pred = []
    intersection_coords = []
    raw_intersection_coords = []
    visibility = []

    for result in results:
        interesection_bumps_pred.append(result[0])
        num_ints = len(result[1])
        # if num_ints > 1:
        #     breakpoint()
        raw_intersection_coords.append(pad_zeros(result[1]))
        if len(result[2]) > 0:
            intersection_coords.extend(result[2])
            visibility.extend(result[3])

    raw_intersection_coords = np.stack(raw_intersection_coords)
    interesection_bumps_pred = np.stack(interesection_bumps_pred)
    interesection_bumps_pred = interesection_bumps_pred.reshape(
        resolution, resolution, -1
    )

    raw_intersection_coords = raw_intersection_coords.reshape(
        resolution, resolution, -1
    )
    intersection_coords = np.array(intersection_coords)
    visibility = np.array(visibility)
    # yapf: disable

    intersection_locs = coords[(intersection_coords[:,0], intersection_coords[:,1], intersection_coords[:,2])]
    point_locs = intersection_locs[:, :3]
    mesh = convert_points_to_mesh(point_locs, mesh_file=None, radius=0.05)

    return_outs = {}

    if return_mesh:
        return_outs['mesh'] = mesh

    if return_pcl:
        return_outs['pcl'] = point_locs

    if return_visiblity:
        return_outs['visibility'] = visibility

    return_outs['raw_intersections'] = interesection_bumps_pred
    return_outs['ray_intersection_coords'] = raw_intersection_coords
    return_outs['query_coords'] = coords
    return_outs['ray_dir'] = ray_dir
    return_outs['distances'] = gt_distance
    # yapf: enable
    return Struct(**return_outs)


def gen_mesh_from_net(
    opts,
    net,
    device,
    meta_data,
    intersection_finder,
    return_mesh=False,
    return_pcl=False,
    return_visiblity=False,
):

    image_tensor = meta_data["img"].to(device=device)
    Kndc = meta_data["Kndc"].to(device=device)
    save_path_pref = meta_data["save_path"]
    if "meshes" in meta_data.keys():
        meshes = meta_data["meshes"]
    else:
        meshes = None
    net.filter_images(
        images=image_tensor[
            None,
        ]
    )
    if "RT" in meta_data.keys():
        RT = meta_data["RT"].to(device=device)
    else:
        RT = None
    if "transforms" in meta_data.keys():
        transforms = meta_data["transforms"]
    else:
        transforms = None

    save_path = save_path_pref
    if isinstance(opts.MODEL.RESOLUTION, list):
        resolution_x, resolution_y, resolution_z = (
            opts.MODEL.RESOLUTION[0],
            opts.MODEL.RESOLUTION[1],
            opts.MODEL.RESOLUTION[2],
        )
    else:
        resolution_x, resolution_y, resolution_z = (
            opts.MODEL.RESOLUTION,
            opts.MODEL.RESOLUTION,
            opts.MODEL.RESOLUTION,
        )
        resolution = resolution_x

    if True:
        coords, ray_dir = create_pixel_query_grid(
            resolution=resolution, z_max=meta_data["z_max"], RT=RT, Kndc=Kndc
        )
        coords = torch.cat([coords, ray_dir * 0], dim=0)
    distances, depths = reconstruction_vol(
        net=net,
        cuda=device,
        Kndc=Kndc,
        RT=RT,
        coords=coords,
    )

    resolution = opts.MODEL.RESOLUTION
    coords = coords.transpose(1, 0)
    coords = coords.reshape(resolution, resolution, resolution, 6).cpu().numpy()
    ray_dir = ray_dir.reshape(resolution, resolution, resolution, 3).cpu().numpy()
    distances = distances.reshape(resolution, resolution, resolution)
    depths = depths.reshape(resolution, resolution, resolution)
    distances = distances.cpu().numpy()
    depths = depths.cpu().numpy()

    if False:
        temp = geometry_utils.transform_points(
            t2c(coords[0, 0, :, 0:3].transpose(1, 0)), RT
        ).permute(1, 0)
        coords = t2c(coords).to(RT.device)
        points_ndc = geometry_utils.covert_world_points_to_pixel(
            coords[0, 0, :, 0:3].transpose(1, 0), RT, Kndc=Kndc, use_cuda=True
        )

    _, results = intersection_finder_utils.double_batch_compute_intersections(
        distances,
        depths,
        intersection_finder,
        add_zeros=True,
        direction=True,
        num_workers=opts.RAY.NUM_WORKERS,
    )
    interesection_bumps_pred = []
    intersection_coords = []
    raw_intersection_coords = []
    visibility = []

    for result in results:
        interesection_bumps_pred.append(result[0])
        num_ints = len(result[1])
        # if num_ints > 1:
        #     breakpoint()
        raw_intersection_coords.append(pad_zeros(result[1]))
        if len(result[2]) > 0:
            intersection_coords.extend(result[2])
            visibility.extend(result[3])

    raw_intersection_coords = np.stack(raw_intersection_coords)
    interesection_bumps_pred = np.stack(interesection_bumps_pred)
    interesection_bumps_pred = interesection_bumps_pred.reshape(
        resolution, resolution, -1
    )

    raw_intersection_coords = raw_intersection_coords.reshape(
        resolution, resolution, -1
    )
    intersection_coords = np.array(intersection_coords)
    visibility = np.array(visibility)
    # yapf: disable

    if len(intersection_coords) == 0:
        logger.info('No intersections detected.')
        return None
    intersection_locs = coords[(intersection_coords[:,0], intersection_coords[:,1], intersection_coords[:,2])]
    point_locs = intersection_locs[:, :3]
    mesh = convert_points_to_mesh(point_locs, save_path_pref, radius=0.05)

    return_outs = {}

    if return_mesh:
        return_outs['mesh'] = mesh

    if return_pcl:
        return_outs['pcl'] = point_locs

    if return_visiblity:
        return_outs['visibility'] = visibility

    return_outs['raw_intersections'] = interesection_bumps_pred
    return_outs['ray_intersection_coords'] = raw_intersection_coords
    return_outs['query_coords'] = coords
    return_outs['ray_dir'] = ray_dir
    return_outs['distances'] = distances
    # yapf: enable
    return Struct(**return_outs)


def convert_points_to_mesh(points, mesh_file, radius=0.1):

    if mesh_file is not None:
        mesh = trimesh.points.PointCloud(points)
        trimesh.exchange.export.export_mesh(mesh, mesh_file)
    # pdb.set_trace()
    icosphere = trimesh.creation.icosahedron()
    icosphere.vertices *= radius
    faces = icosphere.faces
    vertices = icosphere.vertices
    faces_offset = np.arange(0, len(points), dtype=np.int32)
    faces_offset = len(vertices) * faces_offset[:, None] * np.ones((1, len(faces)))

    new_vertices = (
        vertices[
            None,
        ]
        + points[:, None, :]
    )
    new_vertices = new_vertices.reshape(-1, 3)
    new_faces = (
        faces_offset[:, :, None]
        + faces[
            None,
        ]
    )
    new_faces = new_faces.reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    # trimesh.exchange.export.export_mesh(mesh, mesh_file)
    # for p in points:
    #     sp = copy.deepcopy(icosphere)
    #     sp.vertices += p[None, :]
    #     mesh = mesh + sp
    # breakpoint()
    # pdb.set_trace()
    return mesh


def create_pixel_query_grid(resolution, z_max, RT, Kndc):
    b_min = np.array([-1.0, -1.0, 0.1])
    b_max = np.array([1.0, 1.0, z_max])
    coords = grid_utils.create_pixel_aligned_grid(
        resolution, resolution, resolution, b_min, b_max, transform=None
    )
    coords = coords.reshape(3, -1)
    device = RT.device
    coords = coords.to(device=device)
    coords = geometry_utils.convert_pixel_to_world_points(coords, RT, Kndc)
    invRT = torch.inverse(RT)
    ray_dir = coords - invRT[0:3, 3][:, None]
    ## TODO: Convert ray directions to camera frame.
    ## these directions are not relative to the camera. But in the global frame of reference.
    ## So we need to transform them to the camera frame. Eventually at some point.
    ray_dir = torch.nn.functional.normalize(ray_dir, dim=0)
    return coords, ray_dir


def reconstruction_vol(
    net,
    cuda,
    Kndc,
    RT,
    coords,
):
    """
    Create distance volume from network:
    Args:
        net: network
        cuda: cuda device

    """

    # coords, mat = grid_utils.create_pixel_aligned_grid(
    #     resolution, resolution, resolution, b_min, b_max, transform=transform
    # )
    # pdb.set_trace()
    # coords = geometry_utils.convert_pixel_to_world_points(
    #     coords, RT[0], calibndc[0]
    # )
    """
        points is torch.FloatTensor

    """

    def eval_func(points):
        # points = np.repeat(points, net.num_views, axis=0)
        # samples = points.to(device=cuda).float() ## this is B x 6 x N. Where first 3 are xyz, last 3 are ray dir
        # transform points here.
        coords = points[:, :3, :]
        ray_dir = points[:, 3:, :]
        net.query(
            points=coords,
            ray_dir=ray_dir,
            kNDC=Kndc[
                None,
            ],
            RT=RT[
                None,
            ],
        )
        pred = net.get_preds()
        return pred

    def get_depth(points):
        coords = points[:, :3, :]
        ray_dir = points[:, 3:, :]

        # samples = points.to(device=cuda).float()
        depths, xyz_ndc = net.get_depth_points(
            points=coords,
            ray_dir=ray_dir,
            RT=RT[
                None,
            ],
            kNDC=Kndc[
                None,
            ],
        )
        return depths, xyz_ndc

    net.eval()
    coords = coords.view(6, -1)
    with torch.no_grad():
        distances = eval_batch(coords, eval_func)
        depths, _ = eval_batch(coords, get_depth)
    return distances, depths


def convert_intersections_to_mesh(intersections, mesh_file):
    mesh = trimesh.base.Trimesh(
        vertices=intersections[:, :3], faces=intersections[:, 3:]
    )
    mesh.export(mesh_file)
    return mesh_file


def recursive_collate(outputs):
    if len(outputs) > 0:
        if type(outputs[0]) == tuple:
            collected_outputs = ()
            for i in range(len(outputs[0])):
                collected_outputs += (
                    torch.cat([output[i] for output in outputs], dim=-1),
                )
        else:
            collected_outputs = torch.cat(outputs, dim=-1)

        return collected_outputs
    else:
        return []


def eval_batch(points, eval_func, num_samples=10000):
    num_pts = points.shape[1]
    num_batches = (num_pts // num_samples) + 1
    outputs = []
    for i in range(num_batches):
        pts = points[:, i * num_samples : i * num_samples + num_samples]
        outputs.append(eval_func(pts[None]))

    outputs = recursive_collate(outputs)
    if type(outputs) == tuple:
        return_outs = ()
        for ix in range(len(outputs)):
            return_outs = return_outs + (outputs[ix][0],)
        outputs = return_outs
        # outputs = (output[0] for output in outputs)
    else:
        outputs = outputs[0]
    return outputs

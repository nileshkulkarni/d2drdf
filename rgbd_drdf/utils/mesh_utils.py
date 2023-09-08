import copy
import pdb
import typing
from typing import Dict

import numpy as np
import trimesh


def convert_points_to_mesh(points, color=[255, 0, 0], radius=0.01):

    icosahedron = trimesh.creation.icosahedron()
    icosahedron.vertices *= radius
    mesh = trimesh.Trimesh()

    for point in points:
        point_mesh = copy.deepcopy(icosahedron)
        point_mesh.vertices += point[None, :]
        mesh = mesh + point_mesh

    color = trimesh.visual.color.ColorVisuals(mesh, vertex_colors=color)
    mesh = trimesh.Trimesh(mesh.vertices, mesh.faces, visual=color)
    return mesh


def create_marker2(
    marker_height,
):
    origin_size = marker_height / 10.0
    # meshes = [trimesh.creation.axis(origin_size=origin_size)]
    x = marker_height * np.tan(np.deg2rad(120) / 2.0)
    y = marker_height * np.tan(np.deg2rad(120) / 2.0)
    z = marker_height * 2
    box_mesh = trimesh.creation.box((x, y, z))
    box_mesh.vertices += np.array([0, 0, z / 2])[None]

    origin_vertices = np.array([0, 2, 4, 6])
    box_mesh.vertices[origin_vertices] *= 0.01

    keep_faces = np.array([0, 1, 2, 3, 5, 7, 8, 9, 10, 11])
    box_mesh.faces = box_mesh.faces[keep_faces]
    box_mesh = trimesh.Trimesh(vertices=box_mesh.vertices, faces=box_mesh.faces)
    return box_mesh


def camera_marker(marker_height):
    origin_size = marker_height / 10.0
    meshes = [trimesh.creation.axis(origin_size=origin_size)]
    x = marker_height * np.tan(np.deg2rad(90) / 2.0) * 1.25
    y = marker_height * np.tan(np.deg2rad(90) / 2.0)
    z = marker_height

    points = np.array(
        [(0, 0, 0), (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)], dtype=float
    )

    trimesh.creation.box

    # create line segments for the FOV visualization
    # a segment from the origin to each bound of the FOV
    segments = np.column_stack((np.zeros_like(points), points)).reshape((-1, 3))

    # add a loop for the outside of the FOV then reshape
    # the whole thing into multiple line segments
    segments = np.vstack((segments, points[[1, 2, 2, 3, 3, 4, 4, 1]])).reshape(
        (-1, 2, 3)
    )

    breakpoint()
    from trimesh.path.exchange.load import load_path

    # add a single Path3D object for all line segments
    meshes.append(load_path(segments))

    # breakpoint()
    return meshes


def create_cameras_from_RT(
    RT,
):

    marker = create_marker2(0.2)
    transform = np.linalg.inv(RT)
    vertices = np.array(marker.vertices)

    new_vertices = (
        np.matmul(transform[:3, :3], vertices.transpose()) + transform[:3, 3, None]
    )
    new_vertices = new_vertices.transpose()
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=marker.faces)
    # save_mesh(new_mesh, 'camera.ply')
    return new_mesh


def color_ray_with_signature(points, point_color, radius=0.01):

    icosahedron = trimesh.creation.icosahedron()
    icosahedron.vertices *= radius
    mesh = trimesh.Trimesh()
    vertex_colors_all = []
    for px, point in enumerate(points):
        point_mesh = copy.deepcopy(icosahedron)
        point_mesh.vertices += point[None, :]
        visuals = trimesh.visual.color.ColorVisuals(
            point_mesh, vertex_colors=point_color[px]
        )
        vertex_colors = visuals.vertex_colors
        point_mesh = trimesh.Trimesh(
            np.array(point_mesh.vertices) * 1,
            np.array(point_mesh.faces) * 1,
            vertex_color=vertex_colors,
        )
        vertex_colors_all.append(np.array(vertex_colors))
        # trimesh.exchange.export.export_mesh(point_mesh, 'test.ply')
        # point_mesh.visual.vertex_colors = vertex_colors
        mesh = mesh + point_mesh
    vertex_colors_all = np.concatenate(vertex_colors_all)
    visuals = trimesh.visual.color.ColorVisuals(mesh, vertex_colors=vertex_colors_all)
    mesh = trimesh.Trimesh(
        np.array(mesh.vertices) * 1,
        np.array(mesh.faces),
        vertex_colors=visuals.vertex_colors,
    )

    # trimesh.exchange.export.export_mesh(mesh, 'test.ply')
    return mesh


def save_mesh(mesh, file_name):
    trimesh.exchange.export.export_mesh(mesh, file_name)
    return


def save_point_cloud(points, file_name):
    mesh = trimesh.Trimesh(points)
    trimesh.exchange.export.export_mesh(mesh, file_name)
    return


def get_closest_pts(mesh, points):
    closest, distance, _ = trimesh.proximity.closest_point(mesh, points)
    return distance, closest


def visualize_ray_sig(ray_sig: Dict, camera_pose: Dict):
    cone = trimesh.creation.cone(radius=0.1, height=0.2)
    neg_z = np.eye(4)
    neg_z[2, 2] = -1
    cone.apply_transform(neg_z)
    RT = np.linalg.inv(np.array(camera_pose["RT"]))
    cone = cone.apply_transform(RT)
    # pdb.set_trace()
    return cone

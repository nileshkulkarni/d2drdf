import numpy as np
import scipy.spatial
import trimesh


def distance_p2p(
    points_src,
    points_tgt,
):
    """Computes minimal distances of each point in points_src to points_tgt.
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    """

    kdtree = scipy.spatial.KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)
    return dist


def distance_p2m(points, mesh):
    """Compute minimal distances of each point in points to mesh.
    Args:
        points (numpy array): points array
        mesh (trimesh): mesh
    """
    _, dist, _ = trimesh.proximity.closest_point(mesh, points)
    return dist


def compute_nearest(source_pts, target_pts):
    from ..onet.utils.libkdtree import KDTree

    kdtree = KDTree(source_pts)
    dist, idx = kdtree.query(target_pts)
    return dist, idx


def rowData2latex(
    data,
):
    data = [f"{k:.1f}" for k in np.round(data, 3).tolist()]
    return " & ".join(data)

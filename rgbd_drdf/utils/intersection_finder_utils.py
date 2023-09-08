import itertools
import pdb
import tempfile

import numpy as np
import ray
from loguru import logger

from . import sal_utils


def intersection_finder_drdf(
    distance_func, point_depth, direction=False, level_set=0.0, **kwargs
):

    distance_func = distance_func - level_set
    intersections, intersection_inds = sal_utils.zero_crossings(
        distance_func, point_depth, window=5, direction=direction
    )
    intersections = np.array(intersections)
    return intersections, intersection_inds


def pad_zeros(intersections, num_intersections=5):
    new_intersections = np.zeros(num_intersections)

    allowed_len = min(len(intersections), num_intersections)
    if allowed_len > 0:
        new_intersections[-allowed_len:] = intersections[0:allowed_len]
    return new_intersections


@ray.remote(num_cpus=1)
def intersection_finder_remote_batched(
    distance_func, point_depth, intersection_finder, rays, add_zeros=False, **kwargs
):
    int_lst = []
    distr_lst = []
    coords_lst = []
    visibility_lst = []
    for ray_k in rays:
        intersection, intersection_inds = intersection_finder(
            distance_func=distance_func[ray_k[0], ray_k[1]],
            point_depth=point_depth[ray_k[0], ray_k[1]],
            **kwargs,
        )
        coords = []
        visibility = []
        if len(intersection_inds) > 0:
            for ix, ind in enumerate(intersection_inds):
                coords.append([ray_k[0], ray_k[1], ind])
                if ix == 0:
                    visibility.append(ix)  ## basically intersection id.
                else:
                    visibility.append(ix)

        if add_zeros:
            intersection = pad_zeros(intersection)
        int_lst.append(intersection)
        distr_lst.append(intersection_inds)
        coords_lst.append(coords)
        visibility_lst.append(visibility)

    return int_lst, distr_lst, coords_lst, visibility_lst


def double_batch_compute_intersections(
    distances_pred,
    point_depth,
    intersection_finder,
    num_workers=1,
    add_zeros=True,
    **kwargs,
):

    resX, resY, _ = distances_pred.shape
    xs = [k for k in range(resX)]
    ys = [k for k in range(resY)]
    rays = list(itertools.product(xs, ys))
    # inputs_dist_rays = []
    # inputs_point_depth = []
    # for k in rays:
    #     inputs_dist_rays.append(distances_pred[k[0], k[1]])
    #     inputs_point_depth.append(point_depth[k[0], k[1]])
    bsize = 2048
    num_cpus = num_workers
    temp_dir = tempfile.TemporaryDirectory("ray")
    if num_workers == 1:
        ray.init(num_cpus=num_cpus, _temp_dir=temp_dir.name, local_mode=True)
    else:
        ray.init(num_cpus=num_cpus, _temp_dir=temp_dir.name, local_mode=False)
    logger.info(f"Using {num_cpus} for ray parallelization")

    nbatches = (len(rays) + bsize - 1) // bsize
    if True:
        future_results = []
        for bx in range(nbatches):
            ft_res = intersection_finder_remote_batched.remote(
                distance_func=distances_pred,
                point_depth=point_depth,
                intersection_finder=intersection_finder,
                add_zeros=add_zeros,
                rays=rays[bx * bsize : bsize * (bx + 1)],
                **kwargs,
            )
            future_results.append(ft_res)

        results = ray.get(future_results)
    ray.shutdown()

    results_collated = ([], [], [], [])  ## fixed current only works with two outputs.

    for result in results:
        results_collated[0].extend(result[0])
        results_collated[1].extend(result[1])
        results_collated[2].extend(result[2])
        results_collated[3].extend(result[3])

    # temp = [len(results[0][k]) for k in range(nbatches)]

    final_results = [
        k
        for k in zip(
            results_collated[0],
            results_collated[1],
            results_collated[2],
            results_collated[3],
        )
    ]
    return rays, final_results

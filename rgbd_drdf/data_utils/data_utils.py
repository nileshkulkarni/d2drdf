


 @staticmethod
def find_overlap_vis_neighbours(
    img_name,
    overlap_data,
    vis_data,
    depth_frust_vis_data,
    num_neighbours=5,
    opts=None,
    avaliable_nbrs=None,
):
    if overlap_data:
        order = overlap_data["order"]
    elif vis_data:
        order = vis_data["order"]
    elif depth_frust_vis_data:
        order = order = depth_frust_vis_data["order"]
    assert img_name in order, "img_name should be in the order"
    img_index = order.index(img_name)
    if avaliable_nbrs is not None:
        valid_items = np.array(
            [
                True if k.replace(".jpg", "") in avaliable_nbrs else False
                for k in order
            ]
        )
    else:
        valid_items = np.array([True for _ in order])

    depth_vis_frust = None
    if True:
        good_inds = 0
        if "VIS_OVERLAP" in opts.NBR_SELECTION:
            depth_vis = vis_data["vis_ious"]
            # ious = overlap_data["ious"]
            # good_frustrum_overlap = np.array(ious[img_index] > 0.2) * np.array(
            #     ious[img_index] < 0.98
            # )
            good_vis_overlaps = np.array(depth_vis[img_index] > 0.15) * np.array(
                depth_vis[img_index] < 0.80
            )
            good_inds = good_vis_overlaps
            good_inds += good_vis_overlaps
            metric_ious = depth_vis[img_index]
        if "FRUSTRUM_OVERLAP" in opts.NBR_SELECTION:
            order = overlap_data["order"]
            ious = overlap_data["ious"]
            good_frustrum_overlap = np.array(
                ious[img_index] > opts.FRUSTRUM_OVERLAP[0]
            ) * np.array(ious[img_index] < opts.FRUSTRUM_OVERLAP[1])
            good_inds += good_frustrum_overlap
            metric_ious = ious[img_index]
        if "DEPTH_VIS_FRUST_OVERLAP" in opts.NBR_SELECTION:
            order = depth_frust_vis_data["order"]
            # depth_vis_frust_occ = depth_frust_vis_data["occluded_ious"]
            # depth_vis_frust = depth_frust_vis_data["depth_frust_ious"]
            if "DEPTH_VIS_FRUST_OVERLAP_OCC" in opts.NBR_SELECTION:
                depth_vis_frust_occ = depth_frust_vis_data["occluded_ious"][
                    img_index
                ]
                crietria1 = np.array(
                    depth_vis_frust_occ > opts.DEPTH_VIS_FRUSTRUM_OVERLAP[0]
                )
                depth_vis_frust = depth_frust_vis_data["depth_frust_ious"][
                    img_index
                ]
                metric_ious = depth_vis_frust_occ
                crietria2 = (depth_vis_frust_occ - depth_vis_frust) < -0.1
                good_vis_frustrum_overlap = np.logical_and(crietria1, crietria2)

                if good_vis_frustrum_overlap.sum() < num_neighbours // 4:
                    logger.info(f"Relaxing Criteria for {img_name}")
                    crietria1 = np.array(depth_vis_frust_occ > 0.05)
                    good_vis_frustrum_overlap = np.logical_and(crietria1, crietria2)

                good_vis_frustrum_overlap = np.logical_and(
                    good_vis_frustrum_overlap, valid_items
                )
            else:
                depth_vis_frust = depth_frust_vis_data["depth_frust_ious"][
                    img_index
                ]
                good_vis_frustrum_overlap = np.array(
                    depth_vis_frust > opts.DEPTH_VIS_FRUSTRUM_OVERLAP[0]
                )
                metric_ious = depth_vis_frust
            good_inds += good_vis_frustrum_overlap
        # good_inds = good_vis_overlaps + good_frustrum_overlap + good_vis_frustrum_overlap

        iou_inds = np.where(good_inds)[0]

    if False:
        good_inds = good_vis_overlaps * good_frustrum_overlap
        good_inds = good_vis_overlaps
        iou_inds = np.where(good_inds)[0]
   
    data = {}
    bad_neighbours = True
    if len(iou_inds) >= 1:
        bad_neighbours = False

        if False:
            np.random.shuffle(iou_inds)
            sixty_ind = int(num_neighbours * 0.6)
            forty_ind = int(num_neighbours * 0.4)
            try:
                i1_nbrs = [ind for ind in iou_inds if "_i1_" in order[ind]][
                    :sixty_ind
                ]
                oth_nbrs = [ind for ind in iou_inds if "_i1_" not in order[ind]][
                    :forty_ind
                ]
            except IndexError as e:
                breakpoint()
                print("error")
            # i1_overlaps = np.array([ious[img_index, i] * 100 for i in i1_nbrs])
            # oth_overlaps = np.array([ious[img_index, i] * 100 for i in oth_nbrs])
            nbr_inds = i1_nbrs + oth_nbrs
            if len(nbr_inds) < num_neighbours:
                extra_count = num_neighbours - len(nbr_inds)
                np.random.shuffle(iou_inds)
                extra_inds = [k for k in iou_inds[:extra_count]]
                nbr_inds = nbr_inds + extra_inds
            nbr_img_names = [order[i] for i in nbr_inds]

        if True:
            np.random.shuffle(iou_inds)
            nbr_inds = iou_inds[:num_neighbours]
            overlaps = np.array(list(metric_ious[i] * 100 for i in nbr_inds))
            sort_order = np.argsort(overlaps)[::-1]
            nbr_inds = nbr_inds[sort_order]
            nbr_img_names = [order[i] for i in nbr_inds]
        data["nbr_names"] = nbr_img_names
        if "VIS_OVERLAP" in opts.NBR_SELECTION:
            data["vis"] = [depth_vis[img_index, i] * 100 for i in nbr_inds]
        if "FRUSTRUM_OVERLAP" in opts.NBR_SELECTION:
            data["overlaps"] = [ious[img_index, i] * 100 for i in nbr_inds]
        if "DEPTH_VIS_FRUST_OVERLAP" in opts.NBR_SELECTION:
            if "OCC" in opts.NBR_SELECTION:
                data["depth_frust_vis_occluded"] = [
                    depth_vis_frust_occ[i] * 100 for i in nbr_inds
                ]
            else:
                data["depth_frust_vis"] = [
                    depth_vis_frust[i] * 100 for i in nbr_inds
                ]

    if bad_neighbours:
        return None
    else:
        return data
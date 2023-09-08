import os.path as osp

from fvcore.common.config import CfgNode

_C = CfgNode()


curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, "../", "cachedir")

_C.EVAL_DIR = osp.join(cache_path, "eval2")
# _C.MODEL_NAME = "master_4_rgbd_drdf_all_houses_tanh_jitter_bs_pt_depth_no_occ_jitter_no_grad_v100"
_C.MODEL_NAME = ""
_C.NICK_NAME = ""
_C.EVAL_SPLIT = ""
_C.EVAL_GT_OUTPUTS = ""

_C.SCENE_PR_THRESHOLDS = [0.05, 0.1, 0.2, 0.5]

_C.RAY_PR_THRESHOLDS = [0.05, 0.1, 0.2, 0.5]
_C.OCCLUDED_INTERSECTIONS = False
_C.TEST_EPOCH_NUMBER = -1


def get_cfg_defaults() -> CfgNode:
    return _C.clone()

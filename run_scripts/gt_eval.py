import os.path as osp
import random

import numpy as np
import torch

from rgbd_drdf.config import defaults
from rgbd_drdf.testing.scene_drdf_gt import SceneGT
from rgbd_drdf.utils import parse_args

"""
python train_scripts/gt_eval.py  --cfg rgbd_drdf/config/mp3d_gt_eval_data.yaml
"""
if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    cmd_args = parse_args.parse_args()
    cfg = defaults.get_cfg_defaults()
    if cmd_args.cfg_file is not None:
        cfg.merge_from_file(cmd_args.cfg_file)
    if cmd_args.set_cfgs is not None:
        cfg.merge_from_list(cmd_args.set_cfgs)

    cfg.RESULT_DIR = osp.join(cfg.RESULT_DIR, f"{cfg.NAME}")

    tester = SceneGT(cfg)
    tester.initialize()
    tester.dump_eval_data()

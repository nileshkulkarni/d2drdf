import os.path as osp
import random

import numpy as np
import ray
import torch

from rgbd_drdf.config import defaults, defaults_drdf
from rgbd_drdf.testing.scene_drdf import SceneDRDFTester
from rgbd_drdf.utils import parse_args

if __name__ == "__main__":
    # from ..utils import parse_args
    # from ..config import defaults_rgbd
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    cmd_args = parse_args.parse_args()
    cfg = defaults_drdf.get_cfg_defaults()

    if cmd_args.cfg_file is not None:
        cfg.merge_from_file(cmd_args.cfg_file)
    if cmd_args.set_cfgs is not None:
        cfg.merge_from_list(cmd_args.set_cfgs)

    train_opts_file = osp.join(
        "rgbd_drdf/cachedir/", "checkpoints", cfg.NAME, "opts.log"
    )

    if osp.exists(train_opts_file):
        cfg.merge_from_file(train_opts_file)

    if cmd_args.set_cfgs is not None:
        cfg.merge_from_list(cmd_args.set_cfgs)

    cfg.DATALOADER.INFERENCE_ONLY = True
    cfg = defaults.update_derived_params(cfg)
    # ray.init(local_mode=False, num_cpus=4)
    tester = SceneDRDFTester(cfg)
    tester.initialize()
    tester.test_eval_model()

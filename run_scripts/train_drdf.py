import os.path as osp
import random

import numpy as np
import torch

from rgbd_drdf.config import defaults_drdf
from rgbd_drdf.trainer.scene_drdf import SceneDRDFTrainer
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

    cfg.RESULT_DIR = osp.join(cfg.RESULT_DIR, f"{cfg.NAME}")

    trainer = SceneDRDFTrainer(cfg)
    trainer.initialize()
    trainer.train()

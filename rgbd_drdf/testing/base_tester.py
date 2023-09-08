import os
import os.path as osp
import time

import ipdb
import torch
from loguru import logger
from torch.profiler import ProfilerActivity, profile, record_function

from ..html_utils import base_html as base_html_utils
from ..html_utils import scene_html
from ..nnutils import net_blocks as nb
from ..utils import elastic as elastic_utils
from ..utils import tensorboard_utils
from ..utils.timer import Timer


class BaseTester:
    def __init__(self, opts):
        self.opts = opts
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.save_dir = osp.join(opts.CHECKPOINT_DIR, opts.NAME)
        self.log_dir = osp.join(opts.LOGGING_DIR, opts.NAME)
        tf_dir = osp.join(opts.TENSORBOARD_DIR, opts.NAME)
        self.sc_dict = {}
        self.dataloader = None

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        log_file = os.path.join(self.save_dir, "opts_test.log")
        with open(log_file, "w") as f:
            f.write(opts.dump())

        self.dataloader = None
        self.model = None
        self.optimizer = None
        self.scheduler = None

        tf_dir = osp.join(opts.TENSORBOARD_DIR, opts.NAME)
        self.tensorboard_writer = tensorboard_utils.TensorboardWriter(tf_dir)
        return

    def init_dataset(
        self,
    ):
        raise NotImplementedError

    def init_model(
        self,
    ):
        raise NotImplementedError

    def initialize(
        self,
    ):
        self.init_dataset()  ## define self.dataloader
        self.init_model()  ## define self.model
        return

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_id=None):
        print("Saving to deice ")
        save_filename = f"{network_label}_net_{epoch_label}.pth"
        save_path = os.path.join(self.save_dir, save_filename)
        if isinstance(network, torch.nn.DataParallel):
            torch.save(network.module.state_dict(), save_path)
        elif isinstance(network, torch.nn.parallel.DistributedDataParallel):
            torch.save(network.module.state_dict(), save_path)
        else:
            torch.save(network.state_dict(), save_path)
        # if gpu_id is not None and torch.cuda.is_available():
        #     network.cuda(device=gpu_id)
        return

    def load_network(self, network, network_label, epoch_label, network_dir=None):
        save_filename = f"{network_label}_net_{epoch_label}.pth"
        if network_dir is None:
            network_dir = self.save_dir
        save_path = os.path.join(network_dir, save_filename)
        print(f"Loading model : {save_path}")
        network.load_state_dict(torch.load(save_path, map_location="cpu"))
        return

    def set_input(self, batch):
        raise NotImplementedError

    def log_step(self, total_steps):
        raise NotImplementedError

    def log_visuals(self, total_steps):
        raise NotImplementedError

    def forward(
        self,
    ):
        raise NotImplementedError

    def backward(self, total_loss):
        raise NotImplementedError

    def test(
        self,
    ):
        raise NotImplementedError

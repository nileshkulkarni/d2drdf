import os.path as osp

from fvcore.common.config import CfgNode

_C = CfgNode()
_C.MATTERPORT_PATH = ""
_C.MATTERPORT_TAR_PATH = ""


_C.SCANNET = CfgNode()
_C.SCANNET.VIEWS_PER_SCENE = 50
_C.DATALOADER = CfgNode()
_C.DATALOADER.LOAD_FROM_TAR = False
_C.DATALOADER.FILTER_DEPTH = True
_C.DATALOADER.SAMPLING = CfgNode()
_C.DATALOADER.SAMPLING.NOISE_VARIANCE = 0.1
_C.DATALOADER.MODE = ""
_C.DATALOADER.NO_GT_MESH = False
_C.DATALOADER.PRELOAD_MESHES = True
_C.DATALOADER.STRICT_FACE_SELECTION = False
## "X" -- xaxis , "Y" -- axis, "C" -- camera direction. All directions are relative to camera.
_C.DATALOADER.SAMPLING.RAY_DIR_LST = ["Y", "C"]
## same length as ray_dir, corresponding to every direction
_C.DATALOADER.SAMPLING.N_RAYS_LST = [20, 20]

_C.DATALOADER.SAMPLING.Z_MAX = 8.0
_C.DATALOADER.SPLIT = "train"
_C.DATALOADER.VAL_SPLIT = "val"
_C.DATALOADER.DATASET_TYPE = "matterport"

_C.DATALOADER.SIGNED_RAY_DIST = True
_C.DATALOADER.UNSIGNED_RAY_DIST = False
_C.DATALOADER.CLAMP_MAX_DIST = 1.0
_C.DATALOADER.IMG_SIZE = 256

_C.DATALOADER.SINGLE_INSTANCE = False
_C.DATALOADER.SINGLE_BATCH = False
_C.DATALOADER.SHUFFLE = False
_C.DATALOADER.SHUFFLE_ONCE = False

_C.DATALOADER.NO_CROP = False

_C.DATALOADER.START_HOUSE_IND = 0
_C.DATALOADER.END_HOUSE_IND = 2000
_C.DATALOADER.NUM_HOUSES = 100
_C.DATALOADER.SAMPLE_FRAC = 1.0
_C.DATALOADER.USE_POISSON_MESH = False

_C.DATALOADER.INFERENCE_ONLY = False
_C.DATALOADER.RAY_V2 = False

_C.DATALOADER.NUM_NEIGHBORS = 20

_C.MODEL = CfgNode()

_C.MODEL.USE_POINT_FEATURES = True
_C.MODEL.DECODER = "pixNerf"
_C.MODEL.DIR_ENCODING = True
_C.MODEL.RESOLUTION = 128
_C.MODEL.MLP_ACTIVATION = ""
_C.MODEL.PROJECT_POS_EMB = -1
_C.MODEL.MLP_BATCH_NORM = False



_C.OPTIM = CfgNode()
_C.OPTIM.BETA1 = 0.9
_C.OPTIM.LEARNING_RATE = 0.0003
_C.OPTIM.BETA2 = 0.999
_C.OPTIM.GRAD_CLIPPING = CfgNode()
_C.OPTIM.GRAD_CLIPPING.ENABLED = False
_C.OPTIM.GRAD_CLIPPING.MAX_NORM_VALUE = 1.0

_C.TRAIN = CfgNode()
_C.TRAIN.NUM_EPOCHS = 100
_C.TRAIN.NUM_PRETRAIN_EPOCHS = 0
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.NUM_WORKERS = 8
_C.TRAIN.BN_OFF_EPOCH = 50
_C.TRAIN.VALIDATE = True

_C.RAY = CfgNode()
_C.RAY.NUM_WORKERS = 1

_C.TEST = CfgNode()
_C.TEST.NUM_EPOCHS = 200
_C.TEST.LATEST_CKPT = False
_C.TEST.NUM_ITER = 100
_C.TEST.OVERWRITE = False

_C.TEST.HIGH_RES = False
_C.TEST.UUID_SELECT_FILE = ""

_C.MODEL_TYPE = "rgbd_drdf"  ## drdf ## ldi

_C.TEST.EVAL_DIR = osp.join(
    osp.dirname(osp.abspath(__file__)), "../", "cachedir", "eval2"
)
_C.CHECKPOINT_DIR = osp.join(
    osp.dirname(osp.abspath(__file__)), "../", "cachedir", "checkpoints"
)
_C.LOGGING_DIR = osp.join(osp.dirname(osp.abspath(__file__)), "../", "cachedir", "logs")
_C.TENSORBOARD_DIR = osp.join(
    osp.dirname(osp.abspath(__file__)), "../", "cachedir", "tb_logs"
)
_C.ENABLE_ELASTIC_CHECKPOINTING = True
_C.RESULT_DIR = osp.join(
    osp.dirname(osp.abspath(__file__)), "../", "cachedir", "results"
)
_C.PCL_LOG_DIR = osp.join(
    osp.dirname(osp.abspath(__file__)), "../", "cachedir", "pcl_log"
)
_C.PCL_RENDER_DIR = osp.join(
    osp.dirname(osp.abspath(__file__)), "../", "cachedir", "pcl_renders"
)

_C.RENDERER = CfgNode()
_C.RENDERER.RENDER_DIR = osp.join(
    osp.dirname(osp.abspath(__file__)), "../", "cachedir", "render_dir"
)
_C.RENDERER.RENDER_FUNC = "render_utils.render_mesh_matterport"

_C.NAME = "experiment_name"
_C.ENV_NAME = "main"

_C.LOGGING = CfgNode()
_C.LOGGING.PLOT_SCALARS = True
_C.LOGGING.VISUAL_COUNT = 2
_C.LOGGING.PRINT_FREQ = 100
_C.LOGGING.VALID_EPOCH_FREQ = 10
_C.LOGGING.SAVE_VIS_FREQ = 5000
_C.LOGGING.SAVE_CHECKPOINT_FREQ = 1000
_C.LOGGING.SAVE_EPOCH_FREQ = 20
_C.LOGGING.SAVE_VIS = True


_C.LOGGING.WEB_VIS_SERVER = "http://fouheylab.eecs.umich.edu"
_C.LOGGING.WEB_VIS_PORT = 8097

def get_cfg_defaults() -> CfgNode:
    return _C.clone()


curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, "../", "cachedir")


def update_derived_params(cfg):
    # base_data_dir = cfg.BASE_DATA_DIR
    cfg.RESULT_DIR = osp.join(cache_path, "results")
    cfg.CHECKPOINT_DIR = osp.join(cache_path, "checkpoints")
    cfg.LOGGING_DIR = osp.join(cache_path, "logs")
    cfg.RESULTS_DIR = osp.join(cache_path, "results")
    cfg.RENDER_DIR = osp.join(cache_path, "render_dir")

    cfg.TENSORBOARD_DIR = osp.join(cache_path, "tb_logs")
    return cfg

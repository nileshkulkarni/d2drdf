from fvcore.common.config import CfgNode

from . import defaults

_C = defaults.get_cfg_defaults()

# _C.DATALOADER.CLAMP_MIN_DIST = -1.0
# _C.DATALOADER.CLAMP_MAX_DIST = 1.0

_C.DATALOADER.MODE = ""

_C.MODEL.CLIPPED_DISTANCE_VALUE = 1.0
_C.MODEL.ALLOW_CLIPPING = True
_C.MODEL.CLIP_ACTIVATION = ""
_C.MODEL.DIR_ENCODING = False
_C.MODEL.APPLY_LOG_TRANSFORM = True


def get_cfg_defaults() -> CfgNode:
    return _C.clone()


# +VIS_OVERLAP +FRUSTRUM_OVERLAP

from .unet import Unet
from .metrics import iou, soft_iou, dice_score, DiceLoss
import logging
logger = logging.getLogger('backboned_unet')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('[%(name)s: %(filename)s: %(funcName)s]- %(levelname)s - %(message)s'))
logger.addHandler(handler)
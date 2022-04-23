from .unet import Unet
from .metrics import iou,dice_score
import logging
logger = logging.getLogger('backboned_unet')
logger.setLevel(logging.WARNING)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('[%(name)s: %(filename)s: %(funcName)s]- %(levelname)s - %(message)s'))
logger.addHandler(handler)

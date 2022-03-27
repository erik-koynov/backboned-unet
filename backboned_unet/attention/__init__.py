from .attention import MultiplicativeImageAttention,AdditiveAttention,GridAttention
from .base import AttentionModule
import logging
logger = logging.getLogger('backboned_unet_attention')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('[%(name)s: %(filename)s: %(funcName)s]- %(levelname)s - %(message)s'))
logger.addHandler(handler)
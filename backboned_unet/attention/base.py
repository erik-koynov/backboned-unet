from ..base_classes import StrictMeta
from abc import abstractmethod
import torch
import logging
from .utils import transpose_and_reshape
logger = logging.getLogger('backboned_unet_attention')

class AttentionModuleAbstract(metaclass=StrictMeta):
    @abstractmethod
    def compute_attention_map(self, key, query) -> torch.Tensor:
        """
        Compute the attention map between the key and the query
        """

    @abstractmethod
    def forward(self, key, query, value, return_attention_mask):
        """
        Compute the attentive representation fo the value
        """

class AttentionModule(torch.nn.Module, AttentionModuleAbstract):
    def __init__(self,
                 key_channels:int,
                 query_channels:int,
                 position_encoding: type = None,
                 position_encoding_dim=10,
                 n_encoding_positions=64*64,
                 **kwargs):
        super().__init__()
        if position_encoding is None:
            self.position_encoding = lambda x: x
            self.key_channels = key_channels
            self.query_channels = query_channels
        else:
            self.position_encoding = position_encoding(d_hid=position_encoding_dim,
                                                       n_position=n_encoding_positions)
            self.key_channels = key_channels+position_encoding_dim
            self.query_channels = query_channels+position_encoding_dim


    def transpose_and_reshape(self, tensor):
        return transpose_and_reshape(tensor)

    def compute_attentive_representation(self, key, query, value):
        """
        Compute the attentive representation of the value in the context of the key and query correlation given
        by the attention map
        :param key: image of dimensions (B, Ck, Hk, Wk)
        :param query: image of dimensions (B, Cq, Hq, Wq)
        :param value: image of dimensions (B, Cv, Hk, Wk)
        :return:
        """
        assert value.shape[0] == key.shape[0]
        assert value.shape[2] == key.shape[2]
        assert value.shape[3] == key.shape[3]

        B, C, H, W = query.shape
        value_ = self.transpose_and_reshape(value)

        attn_mask = self.compute_attention_map(key, query)

        logger.info(f"value_ shape: {value_.shape}")
        result = torch.matmul(attn_mask.permute(0, 2, 1), value_)
        # print(result.)
        result = result.permute(0, 2, 1)  # BATCH x CHANNEL x H x W

        logger.info(f"Attentive representation shape: {result.shape}")
        result = result.view(B, result.shape[1], H, W)
        logger.info(f"Attentive representation final shape: {result.shape}")

        return result, attn_mask
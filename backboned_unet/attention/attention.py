import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import AttentionModule
import logging
logger = logging.getLogger('backboned_unet_attention')

class GridAttention(AttentionModule, nn.Module):
    def __init__(self, key_channels: int, query_channels: int, out_channels: int):
        super().__init__()
        logger.info(f"Initializing {self.__class__}: \n"
              f"key_channels: {key_channels}, query_channels: {query_channels}, out_channels: {out_channels}")
        self.key_transformation = nn.Conv2d(in_channels=key_channels, out_channels=out_channels,
                                            kernel_size=1)

        self.query_transformation = nn.Conv2d(in_channels=query_channels, out_channels=out_channels,
                                              kernel_size=1)

        self.dimensionality_reducer = nn.Conv2d(in_channels=out_channels,
                                                out_channels=1, kernel_size=1)

        self.non_linearity_pre = nn.ReLU()
        self.non_linearity_post = nn.Sigmoid()

    def compute_attention_map(self, key, query):
        logger.info(f"key before transformation: {key.shape}") # ([B, Ck, Hk, Wk])
        key_ = self.key_transformation(key)
        logger.info(f"key after transformation: {key_.shape}") # ([B, Ct, Hk, Wk])
        query_ = self.query_transformation(query)
        logger.info(f"query after transformation: {query_.shape}") # ([B, Ct, Hq, Wq])
        query_ = F.upsample(query_, size=key_.shape[2:], mode='bilinear')
        logger.info(f"query after upsampling: {query_.shape}") # ([B, Ct, Hk, Wk])
        preactivation = self.non_linearity_pre(query_ + key_)
        logger.info(f"preactivation: {preactivation.shape}") # ([B, Ct, Hk, Wk])
        preactivation = self.dimensionality_reducer(preactivation)
        logger.info(f"preactivation after dimensionality reduction: {preactivation.shape}") # ([B, 1, Hk, Wk])
        preactivation = preactivation.squeeze(1)
        logger.info(f"preactivation after squeezing C dim: {preactivation.shape}") # ([B, Hk, Wk])
        return self.non_linearity_post(preactivation)

    def forward(self, key, query, value, return_attention_mask=False):
        """
        Output has the H W of the value & key!
        :param key:
        :param query:
        :param value:
        :param return_attention_mask:
        :return:
        """
        assert value.shape[0] == key.shape[0]
        assert value.shape[2] == key.shape[2]
        assert value.shape[3] == key.shape[3]
        attention_map = self.compute_attention_map(key, query)
        result = attention_map.unsqueeze(1) * value
        if return_attention_mask:
            return result, attention_map
        return result

class AdditiveAttention(AttentionModule, nn.Module):
    def __init__(self, key_channels: int, query_channels: int, out_channels: int):
        super().__init__()
        self.channel_transformation_key = nn.Conv2d(key_channels, out_channels, kernel_size=1)
        self.channel_transformation_value = nn.Conv2d(query_channels, out_channels, kernel_size=1)
        self.non_linearity_pre = nn.Tanh()
        self.channel_flattening = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.non_linearity_post = nn.Sigmoid()

    def compute_attention_map(self, key, query):
        key_ = self.channel_transformation_key(key)
        logger.info(f"key after channel transform: {key_.shape}")
        query_ = self.channel_transformation_value(query)
        logger.info(f"query after channel transform: {query_.shape}")
        key_ = self.transpose_and_reshape(key_)
        logger.info(f"key after transpose and reshape: {key_.shape}")
        query_ = self.transpose_and_reshape(query_)
        logger.info(f"query after transpose and reshape: {query_.shape}")
        query_ = query_.unsqueeze(1)
        logger.info(f"query unsqueeze: {query_.shape}")
        key_ = key_.unsqueeze(-2)
        logger.info(f"key unsqueeze: {key_.shape}")
        preactivation = key_ + query_  # B x V x K x C
        logger.info(f"preactivation: {preactivation.shape}")
        preactivation = self.non_linearity_pre(preactivation)
        preactivation = preactivation.permute(0, 3, 1, 2)
        logger.info(f"preactivation after transpose: {preactivation.shape}")
        preactivation = self.channel_flattening(preactivation)
        logger.info(f"preactivation after channel flattening: {preactivation.shape}")
        preactivation = preactivation.squeeze(1)
        logger.info(f"preactivation after squeezing C dim: {preactivation.shape}")
        return self.non_linearity_post(preactivation)

    def forward(self, key, query, value, return_attention_mask=False):
        """
        The final output will have the H and W dimensions of the query!
        """
        result, attn_mask = self.compute_attentive_representation(key, query, value)
        if return_attention_mask:
            return result, attn_mask
        return result


class MultiplicativeImageAttention(AttentionModule, nn.Module):
    def __init__(self, key_channels: int, query_channels: int, out_channels: int):
        AttentionModule.__init__(self)
        nn.Module.__init__(self)
        self.channel_transformation_key = nn.Conv2d(key_channels, out_channels, kernel_size=1)
        self.channel_transformation_value = nn.Conv2d(query_channels, out_channels, kernel_size=1)

        self.non_linearity = nn.Sigmoid()

    def forward(self, key, query, value, return_attention_mask=False):
        """
        The final output will have the H and W dimensions of the query!
        """
        result, attn_mask = self.compute_attentive_representation(key, query, value)
        if return_attention_mask:
            return result, attn_mask
        return result

    def compute_attention_map(self, key, query):
        """
        Input Shapes: B,C,H,W
        Output shape: B,Hk*Wk,Hq*Wq
        """
        # logger.info("input: ", key)
        key_ = self.channel_transformation_key(key)
        # logger.info('key_: ', key_)
        logger.info(f"key shape after transform: {key_.shape}" )
        query_ = self.channel_transformation_value(query)
        logger.info(f"query shape after transform: {query_.shape}")

        key_ = self.transpose_and_reshape(key_)
        logger.info(f"key shape after reshape: {key_.shape}" )

        query_ = self.transpose_and_reshape(query_)
        logger.info(f"query shape after reshape: {query_.shape}" )
        # logger.info("key: ", key_)
        # logger.info("value: ", value_)
        attn_mask = torch.matmul(key_, query_.permute(0, 2, 1))
        logger.info(f"attn mask shape: {attn_mask.shape}")
        attn_mask = self.non_linearity(attn_mask)
        return attn_mask

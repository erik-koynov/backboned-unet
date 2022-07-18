import torch.nn as nn
import numpy as np
import torch
from .utils import transpose_and_reshape, inverse_transpose_and_reshape
import logging
logger = logging.getLogger("backboned_unet_attention")
from math import ceil, floor
from abc import ABC, abstractmethod

from typing import Union, Tuple

class EncodingMethod:
    CAT = 'concat'
    ADD = 'add'

class PositionalEncodingAbstract(ABC):
    @abstractmethod
    def _get_sinusoid_encoding_table(self, n_position, d_hid)-> torch.Tensor:
        """compute the sinusoid encoding table: 1 x n_position x d_hid"""

class PositionalEncoding(PositionalEncodingAbstract, nn.Module):

    def __init__(self,
                 d_hid,
                 n_position: Union[int, Tuple[int]] =200,
                 method=EncodingMethod.CAT,
                 ):
        super().__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))
        self.method = method

    def forward(self, x):
        """

        :param x: image : B x C x H x W
        :return:
        """
        original_shape = x.shape
        x = transpose_and_reshape(x) # B x (h*w) x C

        # repeat to make the position embedding match the batch dim of the input
        posional_encoding = self.pos_table[:, :x.shape[1]].clone().detach().repeat(x.shape[0], 1, 1)

        logger.info(f"shape before adding positional encoding: {x.shape}")
        logger.info(f"shape of positional encoding: {posional_encoding.shape}")

        if self.method == EncodingMethod.CAT:
            x = torch.cat([x, posional_encoding], dim=-1) # b x (h*w) x (C+h_dim)
            x = inverse_transpose_and_reshape(x, (original_shape[0], x.shape[-1], *original_shape[2:]))
        else:
            x = x + posional_encoding
            x = inverse_transpose_and_reshape(x, original_shape)

        logger.info(f"shape after adding positional encoding: {x.shape}")

        return x


class PositionalEncoding1d(PositionalEncoding):

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])

        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0) # 1 x positions x h_dim



class PositionalEncoding2d(PositionalEncoding):

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_frequencies_vec(d_hid: int)->np.ndarray:
            """

            :param d_hid: the frequency component (to be multiplied by the time component) and
                                     later this will parameterize a sinusoid function
            :return: the frequency component parameters of the sinusoid
            """
            return np.array([1./np.power(10000, 4 * hid_j / d_hid) for hid_j in range(d_hid)])
        y_pos = n_position[0]
        x_pos = n_position[1]
        # create index arrays of shape y_pos x x_pos
        y_indices, x_indices = np.meshgrid(np.arange(y_pos), np.arange(x_pos), indexing='ij')

        print("shape y_indices: ", y_indices.shape)
        print("shape x_indices: ", x_indices.shape)

        y_dim = ceil(d_hid/2.)
        x_dim = floor(d_hid/2.)

        print("freq vec: ", get_frequencies_vec(y_dim))

        y_sinusoid_table = (y_indices[..., None]*get_frequencies_vec(y_dim)).reshape(-1, y_dim)
        x_sinusoid_table = (x_indices[..., None]*get_frequencies_vec(x_dim)).reshape(-1, x_dim)

        print("y_sinusoid: ",y_sinusoid_table.shape)
        print("x_sinusoid: ",y_sinusoid_table.shape)

        x_sinusoid_table[:, 0::2] = np.sin(x_sinusoid_table[:, 0::2])  # dim 2i
        x_sinusoid_table[:, 1::2] = np.sin(x_sinusoid_table[:, 1::2])  # dim 2i +1
        y_sinusoid_table[:, 0::2] = np.sin(y_sinusoid_table[:, 0::2])  # dim 2i
        y_sinusoid_table[:, 1::2] = np.sin(y_sinusoid_table[:, 1::2])  # dim 2i +1
        final_sinusoidal_table = np.hstack([x_sinusoid_table, y_sinusoid_table])
        return torch.FloatTensor(final_sinusoidal_table).unsqueeze(0) # 1 x positions x h_dim

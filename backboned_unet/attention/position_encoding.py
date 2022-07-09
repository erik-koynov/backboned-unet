import torch.nn as nn
import numpy as np
import torch
from .utils import transpose_and_reshape, inverse_transpose_and_reshape
import logging
logger = logging.getLogger("backboned_unet_attention")

class EncodingMethod:
    CAT = 'concat'
    ADD = 'add'

class PositionalEncoding1d(nn.Module):

    def __init__(self,
                 d_hid,
                 n_position=200,
                 method=EncodingMethod.CAT,
                 ):
        super().__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))
        self.method = method

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])

        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0) # 1 x positions x h_dim

    def forward(self, x):
        """

        :param x: image : B x C x H x W
        :return:
        """
        original_shape = x.shape
        x = transpose_and_reshape(x) # B x (h*w) x C
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


class PositionEncoding2d(nn.Module):
    #https://openjournals.uwaterloo.ca/index.php/vsl/article/download/3533/4579

    def __init__(self, d_hid, n_positions_x=64, n_positions_y=64):
        super().__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_positions_x, n_positions_y, d_hid))

    def _get_sinusoid_encoding_table(self, n_positions_x, n_positions_y, d_hid):
        ''' Sinusoid position encoding table '''

        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])

        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()
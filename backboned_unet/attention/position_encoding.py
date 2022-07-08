import torch.nn as nn
import numpy as np
import torch
from .utils import transpose_and_reshape, inverse_transpose_and_reshape

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
        posional_encoding = self.pos_table[:, :x.shape[1]].clone().detach()

        if self.method == EncodingMethod.CAT:
            x = torch.cat([x, posional_encoding], dim=-1) # b x (h*w) x (C+h_dim)
            x = inverse_transpose_and_reshape(x, (original_shape[0], x.shape[-1], *original_shape[2:]))
        else:
            x = x + posional_encoding
            x = inverse_transpose_and_reshape(x, original_shape)

        return x



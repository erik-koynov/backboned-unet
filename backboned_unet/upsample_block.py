import torch.nn as nn
import torch.nn.functional as F
import torch
import logging
from .attention import GridAttention
logger = logging.getLogger("backboned_unet")


concatenate = lambda *args: torch.cat(args, dim=1)

class UpsampleBlock(nn.Module):

    def __init__(self, ch_in, ch_out=None, attention: type = None, skip_in=0, use_bn=True, parametric=False):
        super(UpsampleBlock, self).__init__()
        logger.info(f"Initializing Upsample block with: attention: {attention.__name__ if attention is not None else None}")
        ch_out = ch_in / 2 if ch_out is None else ch_out
        self.input_channels = ch_in

        if attention is not None and skip_in != 0:
            if attention==GridAttention:
                self.attention_function = attention(key_channels=ch_out, query_channels=skip_in, out_channels=16)
                skip_in = ch_out
            else:
                self.attention_function = attention(key_channels=skip_in, query_channels=ch_out, out_channels=16)
        else:
            self.attention_function = None

        self.parametric = parametric


        # first convolution: either transposed conv, or conv following the skip connection
        if parametric:
            # versions: kernel=4 padding=1, kernel=2 padding=0
            self.upsample = nn.ConvTranspose2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(4, 4),
                                         stride=2, padding=1, output_padding=0, bias=(not use_bn))
            self.batch_norm1 = nn.BatchNorm2d(ch_out) if use_bn else None
        else:
            self.upsample = lambda x: F.interpolate(x, size=None, scale_factor=2, mode='bilinear',
                                                             align_corners=None)
            ch_in = ch_in + skip_in
            self.conv1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(3, 3),
                                   stride=1, padding=1, bias=(not use_bn))
            self.batch_norm1 = nn.BatchNorm2d(ch_out) if use_bn else None

        self.relu = nn.ReLU(inplace=True)

        # second convolution
        conv2_in = ch_out if not parametric else ch_out + skip_in

        self.conv2 = nn.Conv2d(in_channels=conv2_in, out_channels=ch_out, kernel_size=(3, 3),
                               stride=1, padding=1, bias=(not use_bn))
        self.batch_norm2 = nn.BatchNorm2d(ch_out) if use_bn else None

    def forward(self, x, skip_connection=None):
        logger.info(f"x.shape: {x.shape},"
                    f" skip connection: {skip_connection.shape if skip_connection is not None else None}")
        x = self.upsample(x)

        if self.parametric:
            x = self.batch_norm1(x) if self.batch_norm1 is not None else x
            x = self.relu(x)



        if skip_connection is not None:

            if self.attention_function is not None:
                if isinstance(self.attention_function, GridAttention):
                    skip_connection = self.attention_function(x, skip_connection, x)
                else:
                    skip_connection = self.attention_function(skip_connection, x, skip_connection)
            print(x.shape, skip_connection.shape)
            x = concatenate(x, skip_connection)

        if not self.parametric:
            x = self.conv1(x)
            x = self.batch_norm1(x) if self.batch_norm1 is not None else x
            x = self.relu(x)

        x = self.conv2(x)
        x = self.batch_norm2(x) if self.batch_norm2 is not None else x
        x = self.relu(x)

        return x

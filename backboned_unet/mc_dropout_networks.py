from torchvision.models.resnet import Bottleneck, BasicBlock
from typing import Optional, Callable
import torch.nn as nn
from torch import Tensor
import logging
logger = logging.getLogger("backboned_unet")
class BasicBlockMcDropout(BasicBlock):
    dropout_proba = 0.1
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__(inplanes,
                                 planes,
                                 stride,
                                 downsample,
                                 groups,
                                 base_width,
                                 dilation,
                                 norm_layer)
        self.dropout = nn.Dropout(BasicBlockMcDropout.dropout_proba)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.dropout(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BackboneMcDropout(Bottleneck):
    dropout_proba = 0.1
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 ):
        super(BackboneMcDropout, self).__init__(inplanes,
                                                planes,
                                                stride,
                                                downsample,
                                                groups,
                                                base_width,
                                                dilation,
                                                norm_layer)
        self.dropout = nn.Dropout(BackboneMcDropout.dropout_proba)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        #print("input before dropout: ", x)
        out = self.dropout(x)
        #print("input after dropout: ", out)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.dropout(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

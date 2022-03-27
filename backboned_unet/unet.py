import torch
import torch.nn as nn
from .utils import get_backbone
from .upsample_block import  UpsampleBlock
import logging
from .attention import AdditiveAttention, MultiplicativeImageAttention, GridAttention
logger = logging.getLogger("backboned_unet")

class Unet(nn.Module):

    """ U-Net (https://arxiv.org/pdf/1505.04597.pdf) implementation with pre-trained torchvision backbones."""

    def __init__(self,
                 backbone_name='resnet50',
                 attention_module: type = None,
                 pretrained=True,
                 encoder_freeze=False,
                 classes=1,
                 decoder_filters=(256, 128, 64, 32, 16),
                 parametric_upsampling=True,
                 feature_layer_names='default',
                 decoder_use_batchnorm=True,
                 concat_with_input=False,
                 input_shape: tuple = (1,3,224,224)):
        """
        attention_module: the class of the attention. The actual object is going to be initialized when
                          the upsampling layer is initialized
        concat_with_input: whether to use the raw input to compute attention if the final upsampling is
                          done without skip connection from the backbone
        """
        super(Unet, self).__init__()

        self.concat_with_input = concat_with_input
        self.backbone_name = backbone_name

        self.backbone, self.feature_layer_names, self.bb_out_name = get_backbone(backbone_name, pretrained=pretrained)


        shortcut_chs, bb_out_chs = self.infer_skip_channels(input_shape)

        logger.info(f"inferred feature channels: {shortcut_chs}")

        if feature_layer_names != 'default':
            self.feature_layer_names = feature_layer_names

        # build decoder part
        self.upsample_blocks = nn.ModuleList()
        decoder_filters = decoder_filters[:len(self.feature_layer_names)]  # avoiding having more blocks than skip connections
        decoder_filters_in = [bb_out_chs] + list(decoder_filters[:-1])

        # use the more computationally expensive modules only at the last layer
        if attention_module in [AdditiveAttention, MultiplicativeImageAttention]:
            attention_module = [GridAttention]*(len(decoder_filters)-1) + [attention_module]
        else:
            attention_module = [attention_module]*(len(decoder_filters))

        num_blocks = len(self.feature_layer_names)
        for i, [filters_in, filters_out] in enumerate(zip(decoder_filters_in, decoder_filters)):
            logger.info('upsample_blocks[{}] in: {}   out: {}'.format(i, filters_in, filters_out))
            # looping backwards
            #TODO: full attention only at the last layer
            logger.info(f'Skip connections input channels: {shortcut_chs[num_blocks-i-1]}')
            self.upsample_blocks.append(UpsampleBlock(filters_in, filters_out,
                                                      attention=attention_module[i],
                                                      skip_in=shortcut_chs[num_blocks-i-1],
                                                      parametric=parametric_upsampling,
                                                      use_bn=decoder_use_batchnorm))

        self.final_conv = nn.Conv2d(decoder_filters[-1], classes, kernel_size=(1, 1))

        if encoder_freeze:
            self.freeze_encoder()

        self.replaced_conv1 = False  # for accommodating  inputs with different number of channels later

    def freeze_encoder(self):

        """ Freezing encoder parameters, the newly initialized decoder parameters are remaining trainable. """

        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, *input):

        """ Forward propagation in U-Net. """

        x, features = self.forward_backbone(*input)

        for skip_name, upsample_block in zip(self.feature_layer_names[::-1], self.upsample_blocks):
            skip_features = features[skip_name]
            x = upsample_block(x, skip_features)

        x = self.final_conv(x)
        return x

    def forward_backbone(self, x):

        """ Forward propagation in backbone encoder network.  """
        if None in self.feature_layer_names:
            if self.concat_with_input:
                logger.info("Last upsampling will be concatenated with the raw input!")
                features = {None: x}
            else:
                logger.info("Last upsampling done without concatenation with encoder features")
                features = {None: None}
        else:
            features = {}

        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.feature_layer_names:
                features[name] = x
            if name == self.bb_out_name:
                break

        return x, features

    def infer_skip_channels(self, shape: tuple = (1, 3, 224, 224)):

        """ Getting the number of channels at skip connections and at the output of the encoder. """

        x = torch.zeros(shape)
        has_fullres_features = self.backbone_name.startswith('vgg') or self.backbone_name == 'unet_encoder'
        if has_fullres_features: # only VGG has features at full resolution
            channels = []
        elif self.concat_with_input:
            channels =[shape[1]]
        else:
            channels = [0]

        # forward run in backbone to count channels (dirty solution but works for *any* Module)
        logger.info(f"Backbone: {list(self.backbone.named_children())}")
        for name, child in self.backbone.named_children():
            x = child(x)

            if name in self.feature_layer_names:
                logger.info(f"name:{name}, shape: {x.shape}")
                channels.append(x.shape[1])
            if name == self.bb_out_name:
                logger.info(f"Output shape: {x.shape[1]}")
                out_channels = x.shape[1]
                break
        return channels, out_channels

    def get_pretrained_parameters(self):
        for name, param in self.backbone.named_parameters():
            if not (self.replaced_conv1 and name == 'conv1.weight'):
                yield param

    def get_random_initialized_parameters(self):
        pretrained_param_names = set()
        for name, param in self.backbone.named_parameters():
            if not (self.replaced_conv1 and name == 'conv1.weight'):
                pretrained_param_names.add('backbone.{}'.format(name))

        for name, param in self.named_parameters():
            if name not in pretrained_param_names:
                yield param




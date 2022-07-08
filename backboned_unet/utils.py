from torchvision import models
from torchvision.models.resnet import _resnet
from .mc_dropout_networks import BasicBlockMcDropout, BackboneMcDropout
import logging
logger = logging.getLogger("backboned_unet")

def get_backbone(name, pretrained=True, dropout = None):

    """ Loading backbone, defining names for skip-connections and encoder output. 
    output: backbone: nn.Module
            skip_layer_names: the names of the layers that are going to be used as for skip connections
            backbone_output_layer: name of the backbone output layer
    """

    # TODO: More backbones
    BasicBlockMcDropout.dropout_proba = dropout
    BackboneMcDropout.dropout_proba = dropout

    # loading backbone model
    if name == 'resnet18':
        if dropout is None or dropout == 0:
            backbone = models.resnet18(pretrained=pretrained)
        else:
            logger.critical(f"using dropout net with {BasicBlockMcDropout.dropout_proba}")
            backbone = _resnet("resnet18", BasicBlockMcDropout, [2, 2, 2, 2], pretrained, True)

    elif name == 'resnet34':
        if dropout is None or dropout == 0:
            backbone = models.resnet34(pretrained=pretrained)
        else:
            logger.critical(f"using dropout net with {BasicBlockMcDropout.dropout_proba}")
            backbone = _resnet('resnet34', BasicBlockMcDropout, [3, 4, 6, 3], pretrained, True)

    elif name == 'resnet50':
        if dropout is None or dropout == 0:
            backbone = models.resnet50(pretrained=pretrained)
        else:
            logger.critical(f"using dropout net with {BackboneMcDropout.dropout_proba}")
            backbone = _resnet("resnet50", BackboneMcDropout, [3, 4, 6, 3], pretrained, True)

    elif name == 'resnet101':
        if dropout is None or dropout == 0:
            backbone = models.resnet101(pretrained=pretrained)
        else:
            logger.critical(f"using dropout net with {BackboneMcDropout.dropout_proba}")
            backbone = _resnet("resnet101", BackboneMcDropout, [3, 4, 23, 3], pretrained, True)

    elif name == 'resnet152':
        if dropout is None or dropout == 0:
            backbone = models.resnet152(pretrained=pretrained)
        else:
            logger.critical(f"using dropout net with {BackboneMcDropout.dropout_proba}")
            backbone = _resnet("resnet152", BackboneMcDropout, [3, 8, 36, 3], pretrained, True)

    elif name == 'vgg16':
        backbone = models.vgg16_bn(pretrained=pretrained).features
    elif name == 'vgg19':
        backbone = models.vgg19_bn(pretrained=pretrained).features
    # elif name == 'inception_v3':
    #     backbone = models.inception_v3(pretrained=pretrained, aux_logits=False)
    elif name == 'densenet121':
        backbone = models.densenet121(pretrained=True).features
    elif name == 'densenet161':
        backbone = models.densenet161(pretrained=True).features
    elif name == 'densenet169':
        backbone = models.densenet169(pretrained=True).features
    elif name == 'densenet201':
        backbone = models.densenet201(pretrained=True).features
    elif name == 'unet_encoder':
        from .unet_backbone import UnetEncoder
        backbone = UnetEncoder(3)
    else:
        raise NotImplemented('{} backbone model is not implemented so far.'.format(name))

    # specifying skip feature and output names
    if name.startswith('resnet'):
        skip_layer_names = [None, 'relu', 'layer1', 'layer2', 'layer3']
        backbone_output_layer = 'layer4'
    elif name == 'vgg16':
        # TODO: consider using a 'bridge' for VGG models, there is just a MaxPool between last skip and backbone output
        skip_layer_names = ['5', '12', '22', '32', '42']
        backbone_output_layer = '43'
    elif name == 'vgg19':
        skip_layer_names = ['5', '12', '25', '38', '51']
        backbone_output_layer = '52'
    # elif name == 'inception_v3':
    #     skip_layer_names = [None, 'Mixed_5d', 'Mixed_6e']
    #     backbone_output_layer = 'Mixed_7c'
    elif name.startswith('densenet'):
        skip_layer_names = [None, 'relu0', 'denseblock1', 'denseblock2', 'denseblock3']
        backbone_output_layer = 'denseblock4'
    elif name == 'unet_encoder':
        skip_layer_names = ['module1', 'module2', 'module3', 'module4']
        backbone_output_layer = 'module5'
    else:
        raise NotImplemented('{} backbone model is not implemented so far.'.format(name))
    # foreward pass is
    return backbone, skip_layer_names, backbone_output_layer
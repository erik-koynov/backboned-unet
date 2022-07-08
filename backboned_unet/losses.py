from .metrics import dice_score
import torch.nn as nn
import torch
import logging
logger = logging.getLogger('backboned_unet')

class SoftFocalLoss(nn.Module):
    """
    Compute the focal loss  for binary classification allowing for soft labels.
    :param alpha -> class weight (currently not implemented)
    :param gamma -> the down weighting factor (if set to 0 the loss is equal to BCELoss)
    """
    def __init__(self, alpha: float = None,
                 gamma: float = 2.0,
                 ignore_index: int = None,
                 reduction: str = 'mean') -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-6
        self.ignore_index = ignore_index

    def forward(self, probas, targets):

        positive_class = (targets * torch.log(probas + self.eps)) * ((1 - probas) ** self.gamma)
        negative_class = (1 - targets) * torch.log(1 - probas + self.eps) * ((probas) ** self.gamma)
        loss = positive_class + negative_class
        if self.ignore_index is not None:
            loss = loss*(targets != self.ignore_index).type(torch.float64)

        if self.reduction == 'none':
            return -loss
        elif self.reduction == 'mean':
            return -torch.mean(loss)
        elif self.reduction == 'sum':
            return -torch.sum(loss)
        else:
            raise NotImplementedError("Invalid reduction mode: {}"
                                      .format(self.reduction))

class DiceLoss(nn.Module):

    """ Dice score implemented as a nn.Module. """

    def __init__(self, classes, loss_mode='negative_log', ignore_index=255, activation=None):
        super().__init__()
        self.classes = classes
        self.ignore_index = ignore_index
        self.loss_mode = loss_mode
        self.activation = activation

    def forward(self, input, target):
        if self.activation is not None:
            input = self.activation(input)
        score = dice_score(input, target, self.classes, self.ignore_index)
        if self.loss_mode == 'negative_log':
            eps = 1e-12
            return (-(score+eps).log()).mean()
        elif self.loss_mode == 'one_minus':
            return (1 - score).mean()
        else:
            raise ValueError('Loss mode unknown. Please use \'negative_log\' or \'one_minus\'!')

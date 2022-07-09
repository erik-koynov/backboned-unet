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

    def forward(self,
                probas: torch.Tensor,
                targets: torch.Tensor,
                dynamic_class_weights: torch.Tensor = None):
        """

        :param probas: B x 1 x H x W
        :param targets: B x 1 H x W
        :param dynamic_class_weights: B x 2 ( in the following sequence negative , positive class)
        :return:
        """

        if dynamic_class_weights is None:
            dynamic_class_weights = torch.tensor([[1., 1.]])

        dynamic_class_weights = dynamic_class_weights[..., None, None] # follow the broadcasting rules

        positive_class = (targets * torch.log(probas + self.eps)) * ((1 - probas) ** self.gamma)
        negative_class = (1 - targets) * torch.log(1 - probas + self.eps) * ((probas) ** self.gamma)

        logger.info(f"positive loss: {positive_class}")
        logger.info(f"negative loss: {negative_class}")

        loss = dynamic_class_weights[:, 1:2]*positive_class + dynamic_class_weights[:, 0:1]*negative_class

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


def bincount(tensor: torch.LongTensor, n_classes: int):
    """
    calculate the count for each class in an integer array
    tensor: B x 1 x H x W
    out: B x n_classes
    """
    bin_counts = []
    for i in tensor:
        bin_counts.append(torch.bincount(i.ravel(), minlength=n_classes))
    return torch.vstack(bin_counts)

def inverse_class_weight(targets: torch.LongTensor, n_classes: int, maximum = 10., minimum = 1.):
    """
    Inverse class weight, with the possibility of setting thresholds for maximum and minimum weight.
    If the maximum threshold is set (by default 10) the weight will not exceed 10. By setting a
    minimum threshold (by default 1) further reduction of the loss is prevented.
    :param targets:
    :param n_classes:
    :param minimum:
    :return: B x n_classes
    """
    n_dp = targets.shape[-1]*targets.shape[-2]
    logger.info(f"n datapoints: {n_dp}")
    bin_counts = bincount(targets, n_classes)
    logger.info(f"counts: {bin_counts}")
    logger.info(f"before sqrt: {(n_dp / (n_classes * (bin_counts + 1e-8)))}")
    out = torch.minimum(torch.sqrt((n_dp / (n_classes * (bin_counts + 1e-8)))), torch.tensor(maximum))
    out[bin_counts == 0] = 1.
    out = torch.maximum(out, torch.tensor(minimum))
    return out

import torch


def _process_output_probabilities(probabilities: torch.Tensor, threshold: float = None)->torch.Tensor:
    """
    Get class predictions from probabilities.
    :param probabilities: the softmax/sigmoid activated probabilities
    :param threshold: the confidence threshold (if not above thres -> background)
    :return: the predicted classes as a BxHxW integer tensor
    """
    # getting the predictions
    if threshold is None:
        # apply channel-wise argmax
        if probabilities.shape[1]>1:
            pred_tensor = probabilities.argmax(1) #BxHxW
        else:
            # if the output is modeled as bernoulli rv with single output channel : 0.5 thres as substitution for the argmax
            pred_tensor = (probabilities>0.5).type(torch.int64).squeeze(1)

    else:
        # counting predictions above threshold
        pred_tensor = (probabilities>threshold).type(torch.int64).argmax(1) #BxHxW
    return pred_tensor

def _onehot_encode(predictions: torch.Tensor,
        labels: torch.Tensor,
        n_classes: int):
    """
    One-hot-encode the predictions and the labels.
    :param predictions: BxHxW int64 tensor containing the class predictions
    :param labels: BxHxW int64 tensor containing the correct class labels
    :param n_classes: number of classes present
    :return:
    """
    pred_tensor = torch.nn.functional.one_hot(predictions, num_classes=n_classes).permute(0, 3, 1, 2)
    onehot_gt_tensor = torch.nn.functional.one_hot(labels, num_classes=n_classes).permute(0, 3, 1, 2)
    return pred_tensor, onehot_gt_tensor

def iou(predictions: torch.Tensor,
        labels: torch.Tensor,
        threshold=None,
        average=False,
        n_classes=2):

    """ Calculating Intersection over Union score for semantic segmentation.
    predictions: softmax/sigmoid (if binary task) activated class predictions: Tensor: BxClxHxW
    labels: segmentation map, where each pixel is the int encoding of the respective class: Tensor: BxHxW
    """

    pred_tensor = _process_output_probabilities(probabilities = predictions, threshold = threshold)
    pred_tensor, onehot_gt_tensor  = _onehot_encode(pred_tensor, labels, n_classes)

    intersection = (pred_tensor & onehot_gt_tensor).sum([2,3]).float()
    union = (pred_tensor | onehot_gt_tensor).sum([2,3]).float()

    iou = intersection / (union + 1e-12)

    iou = iou.cpu().numpy()
    if average:
        # retain scores for each element of the batch
        iou = iou.mean(axis = 1)

    return iou


def dice_score(predictions: torch.Tensor,
        labels: torch.Tensor,
        threshold=None,
        average=False,
        n_classes=2):

    """ Functional dice score calculation.
    predictions: softmax/sigmoid (if binary task) activated class predictions: Tensor: BxClxHxW
    labels: segmentation map, where each pixel is the int encoding of the respective class: Tensor: BxHxW
    """

    pred_tensor = _process_output_probabilities(probabilities=predictions, threshold=threshold)
    pred_tensor, onehot_gt_tensor = _onehot_encode(pred_tensor, labels, n_classes)

    intersection = (pred_tensor & onehot_gt_tensor).sum([2, 3]).float()
    sum_of_magnitudes = (onehot_gt_tensor.sum([2, 3])+pred_tensor.sum([2, 3])).float()

    dice = 2*intersection/sum_of_magnitudes

    dice = dice.cpu().numpy()
    if average:
        # retain scores for each element of the batch
        dice = dice.mean(axis=1)

    return dice



if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predictions = torch.softmax(torch.empty(7, 21, 224, 224).normal_(), dim=1)
    conv = torch.nn.Conv2d(21, 21, 1, 1, 0)
    labels = (torch.empty(7, 224, 224).normal_(10, 10) % 21)
    criterion = DiceLoss(21, activation=torch.nn.Softmax(dim=1))
    loss = criterion(conv(predictions), labels)

    loss.backward()
    print(loss)

    print('done.')

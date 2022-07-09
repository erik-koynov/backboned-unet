from torch import Tensor

def transpose_and_reshape(tensor: Tensor)->Tensor:
    """
    From the image create a sequence of pixels B x (h*w) x C
    :param tensor: an image with dimensions B x C x H x W
    :return:B x (h*w) x C
    """
    tensor_ = tensor.permute(0, 2, 3, 1)
    tensor_ = tensor_.view(tensor_.size()[0], tensor_.size()[1] * tensor_.size()[2], tensor_.size()[3])
    return tensor_


def inverse_transpose_and_reshape(tensor: Tensor, original_shape: tuple) -> Tensor:
    """
    :param tensor: B x (hxw) x C
    :param original_shape: the original shape of the tensor before transpose_and_reshape
    :return:
    """
    tensor_ = tensor.permute(0, 2, 1).reshape(*original_shape)
    return tensor_

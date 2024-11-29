from typing import Tuple

from .tensor import Tensor
import random


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # Create tiled view
    new_height = height // kh
    new_width = width // kw
    tile_size = kh * kw
    output = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    output = output.permute(0, 1, 2, 4, 3, 5)
    output = output.contiguous().view(batch, channel, new_height, new_width, tile_size)
    return output, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D.

    Args:
    ----
        input: Tensor of size batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    output, new_height, new_width = tile(input, kernel)
    return (
        output.mean(4)
        .contiguous()
        .view(output.shape[0], output.shape[1], new_height, new_width)
    )


def max(input: Tensor, dim: int) -> Tensor:
    """Compute the max along a dimension.

    Args:
    ----
        input: Tensor
        dim: dimension to compute max

    Returns:
    -------
        Tensor with dimension dim reduced to 1

    """
    return input.max(dim)


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D.

    Args:
    ----
        input: Tensor of size batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    output, new_height, new_width = tile(input, kernel)
    return (
        max(output, 4)
        .contiguous()
        .view(output.shape[0], output.shape[1], new_height, new_width)
    )


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor.

    Args:
    ----
        input: Tensor
        dim: dimension to compute softmax

    Returns:
    -------
        Tensor with softmax applied to dimension dim

    """
    # Compute e^x for input and normalize
    out = input.exp()
    return out / (out.sum(dim))


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor.

    Args:
    ----
        input: Tensor
        dim: dimension to compute softmax

    Returns:
    -------
        Tensor with log softmax applied to dimension dim

    """
    max_tensor = max(input, dim)
    shifted_input = input - max_tensor
    sum_exp = shifted_input.exp().sum(dim)
    log_sum_exp = sum_exp.log() + max_tensor
    return input - log_sum_exp


# minitorch.maxpool2d
def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise, include an argument to turn off.

    Args:
    ----
        input: Tensor
        p: probability of dropout
        ignore: if True, do not apply dropout

    Returns:
    -------
        Tensor with dropout applied

    """
    if ignore:
        return input
    mask = input.zeros()
    for i in mask._tensor.indices():
        if random.random() > p:
            mask._tensor.set(i, 1.0)
    output = mask * input
    return output

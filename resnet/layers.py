from collections import OrderedDict
from typing import overload
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import einops
from jaxtyping import Float, Int


class Sequential(nn.Module):
  _modules: dict[str, nn.Module]

  def __init__(self, *args: nn.Module):
    super().__init__()
    if len(args) == 1 and isinstance(args[0], OrderedDict):
      for key, mod in args[0].items():
        self._modules[key] = mod
    else:
      for index, mod in enumerate(args):
        self._modules[str(index)] = mod

  def __getitem__(self, index: int) -> nn.Module:
    index %= len(self._modules)  # deal with negative indices
    return self._modules[str(index)]

  def __setitem__(self, index: int, module: nn.Module) -> None:
    index %= len(self._modules)  # deal with negative indices
    self._modules[str(index)] = module

  def forward(self, x: Tensor) -> Tensor:
    """Chain each module together, with the output from one feeding into the next one."""
    for mod in self._modules.values():
      x = mod(x)
    return x
  
class Conv2d(nn.Module):
  def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
    """
    Same as torch.nn.Conv2d with bias=False.

    Name your weight field `self.weight` for compatibility with the PyTorch version.

    We assume kernel is square, with height = width = `kernel_size`.
    """
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

    # YOUR CODE HERE - define & initialize `self.weight`
    kernel_height = kernel_width = kernel_size
    uki = 1. / np.sqrt(in_channels * kernel_height * kernel_width)
    weight = (2 * t.rand(out_channels, in_channels, kernel_height, kernel_width) - 1) * uki

    self.weight = nn.Parameter(weight)

  def forward(self, x: Tensor) -> Tensor:
    """Apply the functional conv2d, which you can import."""
    return t.nn.functional.conv2d(x, self.weight, stride=self.stride, padding=self.padding)

  def extra_repr(self) -> str:
    keys = ["in_channels", "out_channels", "kernel_size", "stride", "padding"]
    return ", ".join([f"{key}={getattr(self, key)}" for key in keys])
  

class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 1):
      super().__init__()
      self.kernel_size = kernel_size
      self.stride = stride
      self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
      """Call the functional version of maxpool2d."""
      return F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
      """Add additional information to the string representation of this class."""
      return ", ".join([f"{key}={getattr(self, key)}" for key in ["kernel_size", "stride", "padding"]])
    
  
class BatchNorm2d(nn.Module):
  # The type hints below aren't functional, they're just for documentation
  running_mean: Float[Tensor, "num_features"]
  running_var: Float[Tensor, "num_features"]
  num_batches_tracked: Int[Tensor, ""]  # This is how we denote a scalar tensor

  def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
    """
    Like nn.BatchNorm2d with track_running_stats=True and affine=True.

    Name the learnable affine parameters `weight` and `bias` in that order.
    """
    super().__init__()
    self.num_features = num_features
    self.eps = eps
    self.momentum = momentum

    self.weight = nn.Parameter(t.ones(num_features))
    self.bias = nn.Parameter(t.zeros(num_features))

    self.register_buffer("running_mean", t.zeros(num_features))
    self.register_buffer("running_var", t.ones(num_features))
    self.register_buffer("num_batches_tracked", t.tensor(0))

  def forward(self, x: Tensor) -> Tensor:
    """
    Normalize each channel.

    Compute the variance using `torch.var(x, unbiased=False)`
    Hint: you may also find it helpful to use the argument `keepdim`.

    x: shape (batch, channels, height, width)
    Return: shape (batch, channels, height, width)
    """
    if self.training:
      mean = t.mean(x, dim=(0, 2, 3))
      var = t.var(x, dim=(0, 2, 3), unbiased=False) 

      self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
      self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
      self.num_batches_tracked += 1
    
    else:
      mean = self.running_mean
      var = self.running_var

    weight = einops.rearrange(self.weight, "a -> a 1 1")
    bias = einops.rearrange(self.bias, "a -> a 1 1")   
    mean = einops.rearrange(mean, "a -> a 1 1")
    var = einops.rearrange(var, "a -> a 1 1")   

    return ( (x - mean) / t.sqrt(var + self.eps) ) * weight + bias
        

  def extra_repr(self) -> str:
    raise NotImplementedError()
  


class AveragePool(nn.Module):
  def forward(self, x: Tensor) -> Tensor:
    """
    x: shape (batch, channels, height, width)
    Return: shape (batch, channels)
    """
    return t.mean(x, dim=(2, 3))
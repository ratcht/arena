import numpy as np
import torch as t
import torch.nn as nn
from torch import Tensor

class ReLU(nn.Module):
  def forward(self, x: Tensor) -> Tensor:
    return t.maximum(x, t.tensor(0.0))

class Linear(nn.Module):
  def __init__(self, in_features: int, out_features: int, bias=True):
    """
    A simple linear (technically, affine) transformation.

    The fields should be named `weight` and `bias` for compatibility with PyTorch.
    If `bias` is False, set `self.bias` to None.
    """
    super().__init__()

    # init params
    self.in_features = in_features
    self.out_features = out_features

    bound = 1 / np.sqrt(in_features)

    weight = bound * (2 * t.rand(out_features, in_features) - 1) # [0, 1] -> [0, 2] -> [-1, 1] -> [-bound, bound]
    self.weight = nn.Parameter(weight)

    if not bias:
      self.bias = None
      return
    
    bias = bound * (2 * t.rand(out_features) - 1) # [0, 1] -> [0, 2] -> [-1, 1] -> [-bound, bound]
    self.bias = nn.Parameter(bias)

  def forward(self, x: Tensor) -> Tensor:
    """
    x: shape (*, in_features)
    Return: shape (*, out_features)
    """


    y = x @ self.weight.T
    if self.bias != None:
      y += self.bias

    return y

  def extra_repr(self) -> str:
      return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
  

class Flatten(nn.Module):
  def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
    super().__init__()
    self.start_dim = start_dim
    self.end_dim = end_dim

  def forward(self, input: Tensor) -> Tensor:
    """
    Flatten out dimensions from start_dim to end_dim, inclusive of both.
    """
    shape = input.shape

    # Get start & end dims, handling negative indexing for end dim
    start_dim = self.start_dim
    end_dim = self.end_dim if self.end_dim >= 0 else len(shape) + self.end_dim

    # Get the shapes to the left / right of flattened dims, as well as the size of the flattened middle
    shape_left = shape[:start_dim]
    shape_right = shape[end_dim + 1 :]
    shape_middle = t.prod(t.tensor(shape[start_dim : end_dim + 1])).item()

    return t.reshape(input, shape_left + (shape_middle,) + shape_right)

  def extra_repr(self) -> str:
    return ", ".join([f"{key}={getattr(self, key)}" for key in ["start_dim", "end_dim"]])
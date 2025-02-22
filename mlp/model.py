from layers import Flatten, Linear, ReLU
import torch.nn as nn
from torch import Tensor


class SimpleMLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.in_features = 28 * 28
    self.out_features = 10

    self.flatten0 = Flatten()
    self.linear1 = Linear(self.in_features, 100)
    self.relu0 = ReLU()
    self.linear2 = Linear(100, self.out_features)

  def forward(self, x: Tensor) -> Tensor:
    x_1 = self.flatten0(x)
    x_2 = self.linear1(x_1)
    x_3 = self.relu0(x_2)
    x_4 = self.linear2(x_3)

    return x_4
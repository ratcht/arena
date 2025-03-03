import torch as t
import torch.nn as nn
from torch import Tensor
from torchvision import models

from resnet.layers import Sequential, Conv2d, BatchNorm2d, AveragePool, MaxPool2d
from mlp.layers import ReLU, Linear

class ResidualBlock(nn.Module):
  def __init__(self, in_feats: int, out_feats: int, first_stride=1):
    """
    A single residual block with optional downsampling.

    For compatibility with the pretrained model, declare the left side branch first using a `Sequential`.

    If first_stride is > 1, this means the optional (conv + bn) should be present on the right branch. Declare it second using another `Sequential`.
    """
    super().__init__()
    is_shape_preserving = (first_stride == 1) and (in_feats == out_feats)  # determines if right branch is identity

    self.left_branch = Sequential(
      Conv2d(in_feats, out_feats, kernel_size=3, stride=first_stride, padding=1),
      BatchNorm2d(out_feats),
      ReLU(),
      Conv2d(out_feats, out_feats, kernel_size=3, stride=1, padding=1),
      BatchNorm2d(out_feats),
    )

    if is_shape_preserving:
      self.right_branch = nn.Identity()
    else:
      self.right_branch = Sequential(
        Conv2d(in_feats, out_feats, stride=first_stride, kernel_size=1, padding=0),
        BatchNorm2d(out_feats)
      )
    self.relu = ReLU()

  def forward(self, x: Tensor) -> Tensor:
    """
    Compute the forward pass.

    x: shape (batch, in_feats, height, width)

    Return: shape (batch, out_feats, height / stride, width / stride)

    If no downsampling block is present, the addition should just add the left branch's output to the input.
    """
    
    left = self.left_branch(x)
    right = self.right_branch(x)
    out = self.relu(left + right)
    
    return out
  
class BlockGroup(nn.Module):
  def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
    """An n_blocks-long sequence of ResidualBlock where only the first block uses the provided stride."""
    super().__init__()
    first_block = ResidualBlock(in_feats, out_feats, first_stride)
    other_blocks = [ResidualBlock(out_feats, out_feats) for _ in range(1, n_blocks)]
    self.model = Sequential(
      first_block,
      *other_blocks
    )

  def forward(self, x: Tensor) -> Tensor:
    """
    Compute the forward pass.

    x: shape (batch, in_feats, height, width)

    Return: shape (batch, out_feats, height / first_stride, width / first_stride)
    """
    return self.model(x)
  

class ResNet34(nn.Module):
  def __init__(
    self,
    n_blocks_per_group=[3, 4, 6, 3],
    out_features_per_group=[64, 128, 256, 512],
    first_strides_per_group=[1, 2, 2, 2],
    n_classes=1000,
  ):
    super().__init__()
    in_feats0 = 64
    self.n_blocks_per_group = n_blocks_per_group
    self.out_features_per_group = out_features_per_group
    self.first_strides_per_group = first_strides_per_group
    self.n_classes = n_classes

    # YOUR CODE HERE - define all components of resnet34

    self.in_feats_per_group = [in_feats0, *self.out_features_per_group[:-1]]

    self.conv_layers = Sequential(
      Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
      BatchNorm2d(64),
      ReLU(),
      MaxPool2d(3, stride=2),
    )

    self.residual_layers = Sequential(
      *[BlockGroup(n_blocks_per_group[i], self.in_feats_per_group[i], out_features_per_group[i], first_stride=first_strides_per_group[i]) for i in range(0, len(self.n_blocks_per_group))],
    )

    self.out_layers = Sequential(
      AveragePool(),
      Linear(in_features=out_features_per_group[-1], out_features=1000),
    )



  def forward(self, x: Tensor) -> Tensor:
    """
    x: shape (batch, channels, height, width)
    Return: shape (batch, n_classes)
    """
    out_0 = self.conv_layers(x)
    out_1 = self.residual_layers(out_0)
    out = self.out_layers(out_1)
    return out
  
  def copy_weights(self, pretrained_resnet: models.resnet.ResNet):
    """Copy over the weights of `pretrained_resnet` to your resnet."""

    # Get the state dictionaries for each model, check they have the same number of parameters & buffers
    mydict = self.state_dict()
    pretraineddict = pretrained_resnet.state_dict()
    assert len(mydict) == len(pretraineddict), "Mismatching state dictionaries."

    # Define a dictionary mapping the names of your parameters / buffers to their values in the pretrained model
    state_dict_to_load = {
        mykey: pretrainedvalue
        for (mykey, myvalue), (pretrainedkey, pretrainedvalue) in zip(mydict.items(), pretraineddict.items())
    }

    # Load in this dictionary to your model
    self.load_state_dict(state_dict_to_load)

def get_resnet_for_feature_extraction(n_classes: int, device: t.device, freeze: bool = True) -> ResNet34:
  """
  Creates a ResNet34 instance, replaces its final linear layer with a classifier for `n_classes` classes, and freezes
  all weights except the ones in this layer.

  Returns the ResNet model.
  """
  my_resnet = ResNet34()
  
  pretrained_resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1).to(device)
  my_resnet.copy_weights(pretrained_resnet)

  print("Weights copied successfully!")
  
  if freeze:
    # Freeze conv base
    my_resnet.requires_grad_(False)

    my_resnet.out_layers[-1] = Linear(my_resnet.out_features_per_group[-1], n_classes)

  return my_resnet

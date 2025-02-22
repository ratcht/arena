import json
import sys
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from IPython.display import display
from jaxtyping import Float, Int
from PIL import Image
from rich import print as rprint
from rich.table import Table
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm.notebook import tqdm
import os
from pathlib import Path

mlp_dir = Path("mlp")

MNIST_TRANSFORM = transforms.Compose(
  [
    transforms.ToTensor(),
    transforms.Normalize(0.1307, 0.3081),
  ]
)


def get_mnist(trainset_size: int = 10_000, testset_size: int = 1_000) -> tuple[Subset, Subset]:
  """Returns a subset of MNIST training data."""

  # Get original datasets, which are downloaded to "chapter0_fundamentals/exercises/data" for future use
  mnist_trainset = datasets.MNIST(mlp_dir / "data", train=True, download=True, transform=MNIST_TRANSFORM)
  mnist_testset = datasets.MNIST(mlp_dir / "data", train=False, download=True, transform=MNIST_TRANSFORM)

  # # Return a subset of the original datasets
  mnist_trainset = Subset(mnist_trainset, indices=range(trainset_size))
  mnist_testset = Subset(mnist_testset, indices=range(testset_size))

  return mnist_trainset, mnist_testset


mnist_trainset, mnist_testset = get_mnist()
mnist_trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)
mnist_testloader = DataLoader(mnist_testset, batch_size=64, shuffle=False)

# Get the first batch of test data, by starting to iterate over `mnist_testloader`
for img_batch, label_batch in mnist_testloader:
  print(f"{img_batch.shape=}\n{label_batch.shape=}\n")
  break

# Get the first datapoint in the test set, by starting to iterate over `mnist_testset`
for img, label in mnist_testset:
  print(f"{img.shape=}\n{label=}\n")
  break

t.testing.assert_close(img, img_batch[0])
assert label == label_batch[0].item()
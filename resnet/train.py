from torch.utils.data import Subset
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm


IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_TRANSFORM = transforms.Compose(
  [
    transforms.ToTensor(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
  ]
)


def get_cifar(dir) -> tuple[datasets.CIFAR10, datasets.CIFAR10]:
  """Returns CIFAR-10 train and test sets."""
  cifar_trainset = datasets.CIFAR10(dir / "data", train=True, download=True, transform=IMAGENET_TRANSFORM)
  cifar_testset = datasets.CIFAR10(dir / "data", train=False, download=True, transform=IMAGENET_TRANSFORM)
  return cifar_trainset, cifar_testset


@dataclass
class ResNetTrainingArgs:
  batch_size: int = 64
  epochs: int = 5
  learning_rate: float = 1e-3
  n_classes: int = 10


def get_cifar_subset(trainset_size: int = 10_000, testset_size: int = 1_000) -> tuple[Subset, Subset]:
  """Returns a subset of CIFAR-10 train and test sets (slicing the first examples from the datasets)."""
  cifar_trainset, cifar_testset = get_cifar()
  return Subset(cifar_trainset, range(trainset_size)), Subset(cifar_testset, range(testset_size))


def fine_tune(args: ResNetTrainingArgs, model: nn.Module, device: t.device) -> tuple[list[float], list[float], nn.Module]:
  """
  Performs feature extraction on ResNet, returning the model & lists of loss and accuracy.
  """
  # YOUR CODE HERE - write your train function for feature extraction
  optimizer = t.optim.AdamW(model.parameters(), lr=args.learning_rate)

  trainset, testset = get_cifar_subset()
  trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
  testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

  loss_list = []
  accuracy_list = []

  for epoch in range(args.epochs):
    pbar_train = tqdm(trainloader)

    model.train()
    for imgs, labels in pbar_train:
      # Move data to device, perform forward pass
      imgs, labels = imgs.to(device), labels.to(device)
      logits = model(imgs)

      # Calculate loss, perform backward pass
      loss = F.cross_entropy(logits, labels)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      # Update logs & progress bar
      loss_list.append(loss.item())
      pbar_train.set_postfix(epoch=f"{epoch+1}/{args.epochs}", loss=f"{loss:.3f}")


    correct_results = 0
    model.eval()
    for imgs, labels in testloader:
      # Move data to device, perform forward pass
      imgs, labels = imgs.to(device), labels.to(device)

      with t.inference_mode():
        logits: t.Tensor = model(imgs)

        predicted = t.argmax(logits, dim=-1)

        correct_results += (predicted == labels).sum().item()

    # Update logs
    accuracy = correct_results/len(testset)
    accuracy_list.append(accuracy)

  return loss_list, accuracy_list, model
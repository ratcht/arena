
from dataclasses import dataclass
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm



@dataclass
class SimpleMLPTrainingArgs:
  """
  Defining this class implicitly creates an __init__ method, which sets arguments as below, e.g. self.batch_size=64.
  Any of these fields can also be overridden when you create an instance, e.g. SimpleMLPTrainingArgs(batch_size=128).
  """

  batch_size: int = 64
  epochs: int = 3
  learning_rate: float = 1e-3
  device: t.device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
  train_set: Subset = None 
  test_set: Subset = None


def train(model: nn.Module, args: SimpleMLPTrainingArgs) -> tuple[list[float], list[float], nn.Module]:
  """
  Trains the model, using training parameters from the `args` object. Returns the model, and lists of loss & accuracy.
  """
  device = args.device
  model = model.to(device)

  train_loader = DataLoader(args.train_set, batch_size=args.batch_size, shuffle=True)
  test_loader = DataLoader(args.test_set, batch_size=args.batch_size, shuffle=False)

  optimizer = t.optim.Adam(model.parameters(), lr=args.learning_rate)
  loss_list = []
  accuracy_list = []

  for epoch in range(args.epochs):
    pbar_train = tqdm(train_loader)

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

    for imgs, labels in test_loader:
      # Move data to device, perform forward pass
      imgs, labels = imgs.to(device), labels.to(device)

      with t.inference_mode():
        logits: t.Tensor = model(imgs)

      predicted = t.argmax(logits, dim=-1)

      correct_results += (predicted == labels).sum().item()

    # Update logs
    accuracy = correct_results/len(args.test_set)
    accuracy_list.append(accuracy)

  return loss_list, accuracy_list, model
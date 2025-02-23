from pathlib import Path
from torch.utils.data import Subset
from torchvision import datasets, transforms
from pathlib import Path

from model import SimpleMLP
from train import SimpleMLPTrainingArgs, train

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


model = SimpleMLP()
mnist_trainset, mnist_testset = get_mnist()

args = SimpleMLPTrainingArgs(train_set=mnist_trainset, test_set=mnist_testset)

loss_list, accuracy_list, model = train(model, args)

print(f"Accuracy: {accuracy_list}")
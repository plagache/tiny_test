from tinygrad.tensor import Tensor
from tinygrad.nn.state import get_parameters
from tinygrad.nn.optim import Adam

class LinearNet:
  def __init__(self):
    self.l1 = Tensor.kaiming_uniform(784, 128)
    self.l2 = Tensor.kaiming_uniform(128, 10)
  def __call__(self, x:Tensor) -> Tensor:
    return x.flatten(1).dot(self.l1).relu().dot(self.l2)

model = LinearNet()
optim = Adam([model.l1, model.l2], lr=0.001)

print(get_parameters(model))
print([model.l1, model.l2])

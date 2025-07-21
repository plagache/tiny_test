from tinygrad.tensor import Tensor
from lovely_grad import monkey_patch

monkey_patch()

print("=== This is a new Exec ===\n")

x = [[1.3, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2]]
w = [[0, 1, 0, 4], [0, 0, 1, 7], [1, 1, 0, 8]]

a = Tensor(x)
b = Tensor(w)
res = a.dot(b.T)
print(res.numpy())
print(res)

sum = res.sum()
sum.backward()

# pow = res.pow(3)
# print(f"power: {pow}")
#
# print(f"b grad = {b.grad.numpy()}")
# print(f"b = {b}")
# print(f"b.view() = {b.view(-1, 3).numpy()}")
# print(f"b.view() = {b.view(2, -1).numpy()}")
#
# tensor0 = Tensor(0)
# log_0 = tensor0.log()
# print(f"log 0 = {log_0}")
# print(f"log 0 data = {log_0.numpy()}")

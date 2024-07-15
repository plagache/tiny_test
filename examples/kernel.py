from tinygrad.tensor import Tensor

x = [[1,2,3,4],[5,6,7,8],[9,0,1,2]]
y = [[0,1,0,4],[0,0,1,7],[1,1,0,8]]

a = Tensor(x)
b = Tensor(y)
res = a.dot(b.T).numpy()
print(res)

tensor0 = Tensor(0)
log_0 = tensor0.log()
print(f"log 0 = {log_0}")

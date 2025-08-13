import numpy as np
from enum import auto, IntEnum

class Operations(IntEnum):
    ADD = auto()
    MUL = auto()
    SUM = auto()
    NEG = auto()

backward_operations = {
    Operations.ADD: lambda gradient: (gradient, gradient),
    Operations.MUL: lambda gradient, parents: (parents[1] * gradient, parents[0] * gradient),
    Operations.SUM: lambda gradient, parent: (np.full_like(parent, gradient)),
    Operations.NEG: lambda gradient, parents: (None),
}

class Tensor():
    def __init__(self, data):
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data)
        self.grad = None
        # what created this instance
        # if context is None we are at a end of a branch
        self.context = None

    def __add__(self, x):
        return self.ADD(x)

    def ADD(self, x):
        result = Tensor(self.data + x.data)
        result.context = (Operations.ADD, self, x)
        # result.context = (Operations.ADD, self, x)
        return result

    def __mul__(self, x):
        return self.MUL(x)

    def MUL(self, x):
        result = Tensor(self.data * x.data)
        result.context = (Operations.MUL, self, x)
        return result

    def SUM(self):
        result = Tensor(np.sum(self.data))
        result.context = (Operations.SUM, self)
        return result

    def topo_sort(self):
        """
        takes a scalar value
        will explore the different nodes
        """
        # we are doing a DAG: https://en.wikipedia.org/wiki/Directed_acyclic_graph
        # we only need the list of the result, in order
        # we don't had leaf, no context
        # we need to keep track of visited node
        ret = {} # this is our "dict" of nodes | dict have only unique element
        stack = [(self, False)]
        print(f"\n--- First Stack ---\n{stack}")
        # visited = set()
        while stack:
            node, visited = stack.pop()
            if node in ret:
                continue
            if not visited:
                if node.context is not None:
                    ops, *parents = node.context
                    stack.append((node, True))
                    for parent in parents:
                        stack.append((parent, False))
            else:
                ret[node] = None
            print(f"stack: {stack}")
            print(f"return: {ret}")
        return ret

    def backward(self):
        """
        takes a list of nodes as a paramater (the result from the topo_sort)
        apply backward from backward_operations[ops] on each nodes
        """

        # check before doing backward, scalar variable
        self.grad = np.array(1)

        # check type before using them in backward_operations
        # has to be checked at creation actually
        if self.context is not None:
            ops, *parents = self.context
            # print(f"context: {self.context}")
            # print(f"current shape, name:{self.data.shape}, {ops.name}")
            # print(backward_operations[ops], parents)
            for parent in parents:
                print(parent.grad, parent.data)
        return

    def __repr__(self):
        # if self.context is not None:
        #     print(self.context)
        #     ops, *parents = self.context
        #     return f"<Tensor:{self.data.shape}, {self.data}, {self.ops}>"
        # else:
            return f"<Tensor:{self.data.shape}, {self.data}>"

lst = [4, 4, 5, 2]

l_np = np.array(lst, dtype=np.int64)

print(l_np)
print(type(l_np))
print(lst)

# mv_l = memoryview(l)
mv_l_np = memoryview(l_np)
b_l_np = bytearray(l_np)
b_l = bytearray(lst)

print(b_l)
print(b_l_np)
# print(mv_l)
print(mv_l_np)

### Part 2 ###

t_l = Tensor(lst)
t_l_np = Tensor(l_np)

print(t_l is t_l_np)
print(t_l == t_l_np)

print(t_l_np.data, t_l.data)

res_add = t_l + t_l_np
print(res_add.data)
print(type(res_add))
# print(result.context)

add_ops, add_parent1, add_parent2 = res_add.context
print("ops:", add_ops, "parents:", add_parent1, add_parent2)

res_mul = t_l * res_add

print(res_mul.data)
print(type(res_mul))
# print(result.context)

mul_ops, mul_parent1, mul_parent2 = res_mul.context
print("ops:", mul_ops, "parents:", mul_parent1, mul_parent2)

# print(type(result.__add__))

### Part 3 ###

# test difference between IntEnum and Enum
# test auto from enum

# print(Tensor.__dict__)
# print(vars(Tensor))
# print(dir(Tensor))

print(Operations.ADD)
print(Operations.MUL)
# print(list(Operations))
print(Operations['ADD'].value)
print(Operations['MUL'].value)

### Part 4 ###

# add_enum_value, add_lambda = backward_operations[Operations.ADD - 1]
add_lambda = backward_operations[add_ops]
print(add_lambda)

# add_lambda_result = add_lambda((add_parent1.data, add_parent2.data), res_add)
# print("add lambd result:", add_lambda_result)

add_one, add_two = add_lambda(res_add.data)
print("one\n", add_one, "---end one")
print("add_two\n", add_two, "---end add_two")

# print(backward_operations[Operations.MUL - 1])
mul_lambda = backward_operations[mul_ops]
# print(mul_enum_value, mul_lambda)

# print(mul_parent1.data)
# print(type(mul_parent1.data))
# mul_lambda_result = mul_lambda([mul_parent1.data, mul_parent2.data], res_mul.data)
one, two = mul_lambda(res_mul.data, [mul_parent1.data, mul_parent2.data])
# print("mul lambda result:", mul_lambda_result)
print("one---\n", one, "---end one")
print("two\n", two, "---end two")


### Part 5 ###
print("--- Part 5 ---\n")

# Create a simple function that takes has parameters a Tensor and with the function backward wright all the Operations that created it for example F > Sum > Relu > Add > Sum
# for the write of backward we might say this (ops) goes here and the ... rest goes here

bias = Tensor([1, 2, 3, 4])

mul = t_l * t_l_np
add = mul + bias
# print(add.data.shape)
sum = add.SUM()
# print(sum.data, sum.grad)
sum.backward()
elemnt = sum.topo_sort()
print(f"element: {elemnt}")
for elem in reversed(elemnt):
    print(elem)


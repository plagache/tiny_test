import numpy as np
from enum import auto, IntEnum

class Operations(IntEnum):
    ADD = auto()
    MUL = auto()
    SUM = auto()
    NEG = auto()

backward_operations = {
    Operations.ADD: lambda gradient, parent: (gradient, gradient),
    Operations.MUL: lambda gradient, parents: (parents[1] * gradient, parents[0] * gradient),
    Operations.SUM: lambda gradient, parent: (np.full_like(parent, gradient)),
    Operations.NEG: lambda gradient, parents: (None),
}

class Tensor():
    def __init__(self, data, name=None):
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data)
        self.name = name
        self.grad: np.ndarray = None
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
        ret = dict()
        stack = [(self, False)]
        while stack:
            node, visited = stack.pop()
            if node in ret: continue
            if not visited:
                if node.context is not None:
                    stack.append((node, True))
                    ops, *parents = node.context
                    for parent in parents: stack.append((parent, False))
            else:
                ret[node] = None
        return ret


    def backward(self):
        """
        input: a list of nodes as a paramater (the result from the topo_sort)
        apply backward from backward_operations[ops] on each nodes
        """

        operations = []
        # check before doing backward, scalar variable
        self.grad = np.array(1)

        # check type before using them in backward_operations
        # has to be checked at creation actually

        for element in reversed(self.topo_sort()):
            ops, *parents = element.context
            # print(element.__dict__['name'], Operations(ops).name, *parents)
            # print(element, Operations(ops).name)
            operations.append((element, Operations(ops).name))
            backward_operation = backward_operations[ops]
            gradients = backward_operation(element.data, [*parents])
            # print(f"gradients: {gradients}")
            # print(f"type gradients: {type(gradients)}")
            # print(f"\nnew parents")
            for parent, gradient in zip(parents, gradients):
                # print(f"gradient: {gradient}")
                # print(f"parent: {parent}")
                if parent.grad is None:
                    grad = gradient
                else:
                    grad += gradient

        list_ops = []
        for operation in operations:
            tensor, ops = operation
            list_ops.append(f"{ops} : {tensor.data.shape}")

        result = " ---> ".join(list_ops)
        print(f"\n\n{result}")

        return

    def __repr__(self):
        # if self.context is not None:
        #     print(self.context)
        #     ops, *parents = self.context
        #     return f"<{self.data.shape}, {self.data}, {ops}>"
        # else:
            return f"<{self.data.shape}, {self.data}>"

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

add_one, add_two = add_lambda(res_add.data, add_parent1)
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
z = add + bias
sum = z.SUM()
# sum = add.SUM()
# print(sum.data, sum.grad)
sum.backward()

    # def topo_sort(self):
    #     ret = {} # this is our "dict" of nodes | dict have only unique element
    #     stack = [(self, False)] # setup the root node has not visited
    #     # print(f"\n--- First Stack ---\n{stack}")
    #     while stack:
    #         node, visited = stack.pop()
    #         if node in ret:
    #             continue
    #         if not visited:
    #             if node.context is not None:
    #                 stack.append((node, True))
    #                 ops, *parents = node.context
    #                 for parent in parents:
    #                     stack.append((parent, False))
    #         else:
    #             ret[node] = None
    #         print(f"stack: {stack}")
    #         print(f"return: {ret}")
    #     return ret

    # def topo_sort(self):
    #     """
    #     input: a root node
    #     traverse the graph in depth
    #     output: list of all computed node
    #     """
    #     # we have a DAG, https://en.wikipedia.org/wiki/Directed_acyclic_graph
    #     # we are doing a DFS, https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search
    #     # we only need the list of the result, in order
    #     # we don't had leaf, no context
    #     # we need to keep track of visited node | doing that with a boolean
    #     ret = dict()
    #     stack = [(self, False)]
    #     while stack:
    #         node, visited = stack.pop()
    #         # cognitive load reduce:
    #         # soit la node est dans ret (nous sommes dans le rewind on veut donc continuer)
    #         if node in ret:
    #             continue
    #         # soit certaine node n'ont pas encore ete explorer/visite, on veut donc verifier si il y a un context a cette node/parent
    #         # si il y a des parent, on ajoute la node en changeant son Flag a visiter, et on ajoute les parents avec le flag non explorer
    #         if not visited:
    #             if node.context is not None:
    #                 stack.append((node, True))
    #                 ops, *parents = node.context
    #                 for parent in parents:
    #                     stack.append((parent, False))
    #         # soit on est dans le rewind et l'on add les nodes qui ne sont pas dans le return
    #         else:
    #             ret[node] = None
    #
    #     return ret

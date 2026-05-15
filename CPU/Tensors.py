# what is a tensor
# Tensors are multi-dimensional arrays of numbers that generalize scalars, vectors, and matrices
#  to higher dimensions. They are the foundational data structure in machine learning 
# (e.g., TensorFlow) for storing and processing data, representing linear relationships between 
# data points


import torch

# this creates an empty tensor
x = torch.empty(2,2)
print(x)

# create a random tensor
y = torch.rand(2,2)
print(y)

# create a tensor with all zeros
xx = torch.zeros(2,2)
print(xx)

# create a tensor with all ones
yy = torch.ones(2,2)
print(yy)
print(yy.dtype)
print(yy.size)

x = torch.tensor([2.5,0.1])
print(x)


#Basic operations on the tensor

x = torch.rand(2,2)
y = torch.rand(2,2)
print(x*y)
print(x+y)
y.add_(x)
print(y)
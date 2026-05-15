import torch


x = torch.rand(5,3)
print(x)
print(x[1])
print()
print(x[1, :])
print()
print(x[0:2, 1:3])

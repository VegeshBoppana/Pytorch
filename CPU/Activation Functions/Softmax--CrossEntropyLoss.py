import torch
import torch.nn as nn
import numpy as np


## SoftMax  Using Numpy
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis = 0)

x = np.array([2.0,1.0,0.1])
outputs = softmax(x)
print('Softmax numpy:', outputs)

## SoftMax Using Torch

x = torch.tensor([2.0,1.0,0.1])
outputs = torch.softmax(x, dim = 0)
print(outputs)


#Cross Entropy using the Torch
loss = nn.CrossEntropyLoss()
Y = torch.tensor([0])

Y_pred_good = torch.tensor([[2.0,1.0,0.1]])
Y_pred_bad  = torch.tensor([[0.5,2.0,0.3]])

l1 = loss(Y_pred_good,Y)
l2 = loss(Y_pred_bad,Y)
print(l1," ",l2)
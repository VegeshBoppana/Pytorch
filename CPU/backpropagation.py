import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad = True)

y_hat = w * x

loss = (y - y_hat)**2
print(loss)


# backpropagation
loss.backward()
print(w.grad)

# update weights
# next forward and backward pass and couple of iterations
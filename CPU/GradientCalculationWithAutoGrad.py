import torch

x = torch.randn(3, requires_grad = True)
print(x)

y = x+2
print(y)

z = y*y*2
z = z.mean()
print(z)


z.backward() #dz/dx, this gives the gradients and can be used by Gradient descent optimizers
print(x.grad)


print("-------------------------------------------")
xx = torch.tensor(2.0, requires_grad=True)

for step in range(3):
    loss = xx ** 2
    loss.backward()
    print(f"Step {step+1}: {xx.grad}")
    xx.grad.zero_()

##By default, every time you call loss.backward(), PyTorch adds the new gradient on top of the existing one in .grad. 
# It does not overwrite it. 

# So optimizer.zero_grad()   # 1. clear old gradients
# loss.backward()            # 2. compute fresh gradients
# optimizer.step()           # 3. update weights
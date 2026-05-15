import torch

## PyTorch Way 

# f = w * x 
# f = 2 * x
X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad = True)

def forward(x):
    return w * x

def loss(y, y_pred):
    return ((y - y_pred)**2).mean()


learning_rate = 0.01
n_iters = 20

for epoch in range(n_iters):
    y_pred = forward(X)
    l = loss(Y, y_pred)
     
    l.backward()
    
    with torch.no_grad():
        w -= learning_rate * w.grad
    w.grad.zero_()


    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f} , loss = {l:.8f}')

print(f'Prediction for input 5: {forward(5)}')
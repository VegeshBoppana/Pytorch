import torch
import time

device = "cuda"

x = torch.rand(10000, 10000, device=device)
y = torch.rand(10000, 10000, device=device)

torch.cuda.synchronize()

start = time.time()

z = torch.matmul(x, y)

torch.cuda.synchronize()

end = time.time()

print("GPU Time:", end - start)
import torch
import torch.nn as nn
import time

# GPU check
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using Device:", device)
print("GPU:", torch.cuda.get_device_name(0))

# Huge synthetic dataset
batch_size = 2048
input_size = 4096
hidden_size = 8192
output_size = 4096

# Large model
model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size)
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_fn = nn.MSELoss()

print("\nStarting GPU Stress Training...\n")

for epoch in range(50):

    # Random data directly on GPU
    x = torch.randn(batch_size, input_size, device=device)
    y = torch.randn(batch_size, output_size, device=device)

    start = time.time()

    # Forward pass
    outputs = model(x)

    # Loss
    loss = loss_fn(outputs, y)

    # Backprop
    optimizer.zero_grad()
    loss.backward()

    # Update weights
    optimizer.step()

    torch.cuda.synchronize()

    end = time.time()

    print(
        f"Epoch {epoch+1:02d} | "
        f"Loss: {loss.item():.4f} | "
        f"Time: {end-start:.3f}s"
    )

print("\nTraining Complete")
import torch

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("CUDA Version:", torch.version.cuda)

    x = torch.rand(3, 3).cuda()
    print("\nTensor on GPU:")
    print(x)
else:
    print("CUDA not working")
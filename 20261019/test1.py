import torch

print(torch.cuda.get_device_name())
print(torch.cuda.get_device_capability())
print(torch.cuda.is_bf16_supported())
import torch
from src.model import build_generator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = build_generator()
model.load_state_dict(torch.load("best.pth", map_location=device), strict=False)

count = 0
for name, param in model.named_parameters():
    print(name)
    count += 1
print(f"Number of loaded weights: {count}")

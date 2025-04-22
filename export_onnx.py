import torch
import torch.onnx
from src.model import build_generator
import numpy as np
from src.utils import makedirs, fix_model_state_dict

# Load the model
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device, "Loading the Model...")
model_path = './best.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device, "Loading the Model...")
generator = build_generator()
#generator.to(device)
checkpoint = torch.load(model_path, map_location=device)
generator.load_state_dict(fix_model_state_dict(checkpoint['net_G']))
generator.to(device)

# Dummy Input
dummy_input = torch.randn(1, 3, 128, 128*5)


# Export the model to ONNX
try:
    torch.onnx.export(generator,
                  dummy_input,
                  "model.onnx",
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output_b', 'output_mask'],
                  dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                                'output_b': {0: 'batch_size', 2: 'height', 3: 'width'},
                                'output_mask': {0: 'batch_size', 2: 'height', 3: 'width'}})

    print("ONNX export complete")
except Exception as e:
    print(f"Error exporting model to ONNX: {e}")

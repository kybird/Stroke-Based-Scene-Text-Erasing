import torch
import torch.onnx
import numpy as np
from PIL import Image
from src.model import build_generator
from src.utils import makedirs, fix_model_state_dict
import onnxruntime

# 1. 입력 데이터 준비
image_path = "example/images/img_0.jpg"
image = Image.open(image_path).resize((320, 320))
image_array = np.array(image).astype(np.float32) / 255.0
image_array = np.transpose(image_array, (2, 0, 1))
input_data = np.expand_dims(image_array, axis=0)
input_data = torch.from_numpy(input_data)

# 2. PyTorch 모델 실행
model_path = './best.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pytorch_model = build_generator()
checkpoint = torch.load(model_path, map_location=device)
pytorch_model.load_state_dict(fix_model_state_dict(checkpoint['net_G']))
pytorch_model.to(device)
pytorch_model.eval()
input_data = input_data.to(device)
# 2. PyTorch 모델 실행
with torch.no_grad():
    pytorch_outputs = pytorch_model(input_data)
    pytorch_output_b = pytorch_outputs[0].cpu().numpy()
    pytorch_output_mask = pytorch_outputs[1].cpu().numpy()

# 3. ONNX 모델 실행
session = onnxruntime.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name
onnx_outputs = session.run(None, {input_name: input_data.cpu().numpy()})
onnx_output_b = onnx_outputs[0]
onnx_output_mask = onnx_outputs[1]

# 4. 출력 비교
def compare_outputs(name, pytorch_out, onnx_out):
    print(f"\n[{name}] 비교 결과")
    print(f"PyTorch shape: {pytorch_out.shape}")
    print(f"ONNX shape: {onnx_out.shape}")
    diff = np.abs(pytorch_out - onnx_out)
    print(f"Mean absolute difference: {np.mean(diff)}")
    print(f"Max absolute difference: {np.max(diff)}")
    return np.all(diff <= 1e-3)

# 각 출력별 비교
is_valid_b = compare_outputs("output_b", pytorch_output_b, onnx_output_b)
is_valid_mask = compare_outputs("output_mask", pytorch_output_mask, onnx_output_mask)

# 최종 검증
if is_valid_b and is_valid_mask:
    print("\n✓ ONNX 모델이 두 출력 모두 정상 동작")
else:
    print("\n⚠️ 출력간 차이 발견")
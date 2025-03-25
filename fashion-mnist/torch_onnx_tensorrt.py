import torch
import onnx
from torchvision.models import resnet34

# model
ckpt_file = "./resnet34.pth"
float_model = resnet34(num_classes=10)
float_model.load_state_dict(torch.load(ckpt_file, map_location="cpu"))
float_model.eval()

# torch -> onnx
example_inputs = (torch.randn(1, 3, 32, 32), )
onnx_program = torch.onnx.export(float_model, example_inputs, dynamo=True)
onnx_program.save("./resnet34.onnx")

onnx_model = onnx.load("./resnet34.onnx")
onnx.checker.check_model(onnx_model)

import os
import torch
from torch.utils.data import DataLoader
import torch_tensorrt.ts.ptq
from torchvision.models import resnet34
import torch_tensorrt
import torchvision
from torchvision import transforms
from tqdm import tqdm


# dataset
def repeat_channels(image):
    return image.repeat(3, 1, 1)


transform = transforms.Compose(
    [
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(dtype=torch.float32),
        transforms.Normalize(mean=0, std=1),
        transforms.Lambda(repeat_channels),
    ]
)
test_dataset = torchvision.datasets.FashionMNIST(
    "/mnt/z/datasets", train=False, download=True, transform=transform
)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

# model
ckpt_file = "./resnet34.pth"
float_model = resnet34(num_classes=10)
float_model.load_state_dict(torch.load(ckpt_file, map_location="cpu"))
float_model.eval()


# evaluation
def evaluation(model, dataloader, device=torch.device(0)):
    model = model.to(device)
    num_correct = 0
    num_total = 0
    for image, target in tqdm(dataloader, ncols=100, desc="test"):
        image, target = image.to(device), target.to(device)

        with torch.no_grad():
            pred = model(image)

        probs = torch.softmax(pred, axis=1)
        pred_labels = torch.argmax(probs, axis=1)

        num_correct += (pred_labels == target).sum()
        num_total += len(pred_labels)
    acc = num_correct / num_total

    return acc.item()


device = torch.device(0)

model_to_quantize = resnet34(num_classes=10)
model_to_quantize.load_state_dict(torch.load(ckpt_file, map_location="cpu"))
model_to_quantize.eval().to(device)


calibrator = torch_tensorrt.ts.ptq.DataLoaderCalibrator(
    test_dataloader,
    use_cache=False,
    algo_type=torch_tensorrt.ts.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
    device=device,
)

trt_mod = torch_tensorrt.compile(
    model_to_quantize,
    inputs=[torch_tensorrt.Input((1, 3, 32, 32))],
    enabled_precisions={torch.float, torch.half, torch.int8},
    calibrator=calibrator,
    device=device,
)

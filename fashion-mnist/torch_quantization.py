import os
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet34
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
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
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=16)

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


def calibrate(model, dataloader):
    torch.ao.quantization.move_exported_model_to_eval(model)
    with torch.no_grad():
        for image, _ in dataloader:
            model(image)


model_to_quantize = resnet34(num_classes=10)
model_to_quantize.load_state_dict(torch.load(ckpt_file, map_location="cpu"))
model_to_quantize.eval()

inputs = (next(iter(test_dataloader))[0],)

exported_model = torch.export.export_for_training(model_to_quantize, inputs).module()

quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
prepared_model = prepare_pt2e(exported_model, quantizer)
print(prepared_model.graph)
calibrate(prepared_model, test_dataloader)
quantized_model = convert_pt2e(prepared_model)
quantized_model = torch.export.export_for_training(quantized_model, inputs).module()

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p")/1e6)
    os.remove("temp.p")
    
acc = evaluation(float_model, test_dataloader)
quantized_acc = evaluation(quantized_model, test_dataloader)

print_size_of_model(float_model)
print_size_of_model(quantized_model)
print(acc, quantized_acc)

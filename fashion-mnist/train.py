import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.models import resnet34, ResNet34_Weights
from torchvision import transforms
from tqdm import tqdm

# config
device = torch.device(0)
init_lr = 0.001
batch_size = 64
num_workers = 16
num_classes = 10
total_epochs = 10
ckpt_file = './resnet34.pth'

# dataset
def repeat_channels(image):    
    return image.repeat(3, 1, 1)

transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(dtype=torch.float32),
    transforms.Normalize(mean=0, std=1),
    transforms.Lambda(repeat_channels)
])

train_dataset = torchvision.datasets.FashionMNIST(
    "/mnt/z/datasets", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.FashionMNIST(
    "/mnt/z/datasets", train=False, download=True, transform=transform
)

# dataloader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


# model
net = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
net.fc = nn.Linear(512, num_classes)
net = net.to(device)

# loss
lossfn = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.Adam(net.parameters(), lr=init_lr)

for epoch in range(total_epochs):
    # train
    net.train()
    with tqdm(total=len(train_dataloader), ncols=100) as _tqdm:
        _tqdm.set_description(f'epoch: {epoch}/{total_epochs}')
            
        for image, target in train_dataloader:
            image, target = image.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            pred = net(image)
            loss = lossfn(pred, target)
            loss.backward()
            
            optimizer.step()
            
            _tqdm.set_postfix(loss=f'{loss.item():.4f}')
            _tqdm.update(1)
        
    # test
    net.eval()
    
    num_correct = 0
    num_total = 0
    for image, target in tqdm(test_dataloader, ncols=100, desc='test'):
        image, target = image.to(device), target.to(device)
        
        with torch.no_grad():
            pred = net(image)
            
        probs = torch.softmax(pred, axis=1)
        pred_labels = torch.argmax(probs, axis=1)
        
        num_correct += (pred_labels == target).sum()
        num_total += len(pred_labels)
    acc = num_correct / num_total
    
    print(f'Epoch: {epoch}/{total_epochs}, Accuracy: {acc:.4f}')
    
    
# save
torch.save(net.state_dict(), ckpt_file)

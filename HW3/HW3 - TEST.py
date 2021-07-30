import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder

test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5528488 , 0.44283208 ,0.34296584], std=[0.25198472,0.25044954,0.25717643]),
])

batch_size = 32


test_set = DatasetFolder("food-11/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)

test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
print(len(test_set))
print(len(test_loader))



class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input image size: [3, 128, 128]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),

            nn.BatchNorm2d(64),
            nn.SELU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1, 1),

            nn.BatchNorm2d(128),
            nn.SELU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),

            nn.BatchNorm2d(256),
            nn.SELU(),
            nn.MaxPool2d(4, 4, 0),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 256),

            nn.SELU(),
            nn.Linear(256, 256),

            nn.SELU(),
            nn.Linear(256, 11)
        )

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]
        x = self.cnn_layers(x)
        x = x.flatten(1)
        x = self.fc_layers(x)
        return x



torch.cuda.empty_cache()
#超参数的设置
config = {
    'batch_size': 32,
    'early_stop': 300,
    'save_path': './model.ckpt'

}
device = 'cuda'
model = Classifier().to(device)
model.load_state_dict(torch.load(config['save_path']))

#测试
model.eval()
predictions = []
print("test_loader length:" + str(len(test_loader)))
print("test_loader type:" + str(type(test_loader)))
for imgs, labels in test_loader:
    with torch.no_grad():
        logits = model(imgs.to(device))
        print("logits shape:",logits.shape)
    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

with open("predict.csv", "w") as f:
    f.write("Id,Category\n")
    for i, pred in  enumerate(predictions):
         f.write(f"{i},{pred}\n")
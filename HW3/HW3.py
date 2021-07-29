import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset"用于半监督学习
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder


train_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation((0,360)),
    transforms.ToTensor(),
])
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

batch_size = 32

#train_set = DatasetFolder("food-11/training/labeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
train_set1 = DatasetFolder("food-11/training/labeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
train_set2 = DatasetFolder("food-11/training/labeled", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
train_set = ConcatDataset([train_set1, train_set2])
valid_set = DatasetFolder("food-11/validation", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
unlabeled_set = DatasetFolder("food-11/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
test_set = DatasetFolder("food-11/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
print(len(train_set))
print(len(train_loader))

train_list=[]
for x, y in train_set:
       y = torch.tensor(y)
       tr=(x,y)
       train_list.append(tr)
print(len(train_list))

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input image size: [3, 128, 128]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 11)
        )

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]
        x = self.cnn_layers(x)
        x = x.flatten(1)
        x = self.fc_layers(x)
        return x

def get_pseudo_labels(dataset, model, threshold=0.65):

    device = "cuda"
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    model.eval()

    softmax = nn.Softmax(dim=-1)

    new_data=[]

    for img, _ in data_loader:


        with torch.no_grad():
            logits = model(img.to(device))

        probs = softmax(logits)

        probs_argmax = probs.argmax(dim=-1).cpu()
        probs_max = probs.max(dim=-1).values
        probs_max = probs_max.cpu().numpy().tolist()

        feat = []
        for idx,p in enumerate(probs_max):
          if p > threshold:
            feat.append(idx)
            tr=(img[idx],probs_argmax[idx])

            new_data.append(tr)
        #tr = torch.cat((tr,img[feat]))
        #tl = torch.cat((tl,probs_argmax[feat]))

    #new_data = ConcatDataset([tr,tl])
    print("new_data length:",len(new_data))

    model.train()
    return new_data

def train(train_set, unlabeled_set, train_list,valid_loader, model, config):
    learning_rate = 0.0001
    weight_decay = 0.001
    best_acc=0.0
    do_semi = True
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    device = 'cuda'
    num_epoch = config['n_epochs']
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epoch):
        if do_semi:
              pseudo_set = get_pseudo_labels(unlabeled_set, model)
              train_list.extend(pseudo_set)
              #concat_dataset = ConcatDataset([train_list, pseudo_set])
              train_loader = DataLoader(train_list, batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)

        train_loss = []
        train_accs = []
        model.train()
        for imgs, labels  in train_loader:
            logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)#梯度裁剪，防止梯度爆炸
            optimizer.step()
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            train_loss.append(loss.item())
            train_accs.append(acc)

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        print(f"[ Train | {epoch + 1:03d}/{num_epoch:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")


        valid_accs,valid_loss=val(valid_loader, model, device)
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
        print(f"[ Valid | {epoch + 1:03d}/{num_epoch:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), config['save_path'])
            print('saving model with acc {:.3f}'.format(valid_acc))

def val(valid_loader, model, device):
        criterion = nn.CrossEntropyLoss()
        valid_loss = []
        valid_accs = []
        model.eval()
        for imgs, labels in valid_loader:

            with torch.no_grad():
                logits = model(imgs.to(device))
                loss = criterion(logits, labels.to(device))
                acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
                valid_loss.append(loss.item())
                valid_accs.append(acc)
        return  valid_accs,valid_loss

torch.cuda.empty_cache()
#超参数的设置
config = {
    'n_epochs': 80,
    'batch_size': 32,
    'early_stop': 300,
    'save_path': './model.ckpt'

}
device = 'cuda'
model = Classifier().to(device)
model.load_state_dict(torch.load(config['save_path']))
train(train_set, unlabeled_set, train_list,valid_loader, model, config)

#测试
model.eval()
predictions = []
print("test_loader length:" + str(len(test_loader)))
print("test_loader type:" + str(type(test_loader)))
for imgs, labels in test_loader:
    # DatasetFolder returns images and labels for each batch,
    # so we have to create fake labels to make it work normally.

    with torch.no_grad():
        logits = model(imgs.to(device))
        print("logits shape:",logits.shape)
    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())


# 保存预测结果
with open("predict.csv", "w") as f:
    f.write("Id,Category\n")
    for i, pred in  enumerate(predictions):
         f.write(f"{i},{pred}\n")
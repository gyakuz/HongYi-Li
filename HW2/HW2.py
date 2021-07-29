import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import gc
import torch
import torch.nn as nn


data_root='./timit_11/'
train = np.load(data_root + 'train_11.npy')
train_label = np.load(data_root + 'train_label_11.npy')
test = np.load(data_root + 'test_11.npy')
print('Size of training data: {}'.format(train.shape))
print('Size of testing data: {}'.format(test.shape))


class TIMITDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = torch.from_numpy(X).float()
        if y is not None:
            y = y.astype(np.int)
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)

#数据集分割
VAL_RATIO = 0.2
percent = int(train.shape[0] * (1 - VAL_RATIO))
train_x, train_y, val_x, val_y = train[:percent], train_label[:percent], train[percent:], train_label[percent:]
print('Size of training set: {}'.format(train_x.shape))
print('Size of validation set: {}'.format(val_x.shape))
BATCH_SIZE = 64

#数据加载
train_set = TIMITDataset(train_x, train_y)
val_set = TIMITDataset(val_x, val_y)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

#清理不需要的变量释放内存
del train, train_label, train_x, train_y, val_x, val_y
gc.collect()


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer1 = nn.Linear(429, 1024)
        self.layer2 = nn.Linear(1024, 512)
        self.layer3 = nn.Linear(512, 128)
        self.out = nn.Linear(128, 39)
        self.dropout = nn.Dropout(0.15)

        self.relu = nn.SELU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.layer3(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.out(x)

        return x


def train(train_set, val_set, model, config):
    learning_rate = 0.0001
    weight_decay = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    device = 'cuda'
    num_epoch = config['n_epochs']
    best_acc = 0.0
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epoch):

        train_acc = 0.0
        train_loss = 0.0
        model.train()  # set the model to training mode
        for x, y  in train_set:
             x, y = x.to(device), y.to(device)
             optimizer.zero_grad()
             outputs = model(x)
             batch_loss = criterion(outputs, y)
             _, train_pred = torch. max(outputs, 1)  # get the index of the class with the highest probability
             batch_loss.backward()
             optimizer.step()

             train_acc += (train_pred.cpu() == y.cpu()).sum().item()
             train_loss += batch_loss.item()
        if len(val_set) > 0:
            val_acc,val_loss=val(val_set, model, device)
            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc / config['train_len'], train_loss / len(train_set),
            val_acc / config['val_len'], val_loss / len(val_set)
            ))

            if val_acc > best_acc:
                 best_acc = val_acc
                 torch.save(model.state_dict(), config['save_path'])
                 print('saving model with acc {:.3f}'.format(best_acc / config['val_len']))
        else:
             print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc / config['train_len'], train_loss / len(train_set)
            ))


def val(val_set, model, device):
        criterion = nn.CrossEntropyLoss()
        val_acc = 0.0
        val_loss = 0.0
        model.eval()  # set the model to evaluation mode
        for x, y in val_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                outputs = model(x)
                batch_loss = criterion(outputs, y)
                _, val_pred = torch.max(outputs, 1)

                val_acc += (
                            val_pred.cpu() == y.cpu()).sum().item()
                val_loss += batch_loss.item()
        return  val_acc,val_loss


#超参数的设置
config = {
    'n_epochs': 20,
    'batch_size': 64,
    'early_stop': 300,
    'train_len': len(train_set),
    'val_len': len(val_set),
    'save_path': './model.ckpt'

}

device = 'cuda'
model = Classifier().to(device)
train(train_loader, val_loader, model, config)

#评估测试
del model
test_set = TIMITDataset(test, None)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
model = Classifier().to(device)
model.load_state_dict(torch.load(config['save_path']))
predict = []
model.eval()
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs = data
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, test_pred = torch.max(outputs, 1) #得到最大可能性的分类
        for y in test_pred.cpu().numpy():
            predict.append(y)
with open('prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(predict):
        f.write('{},{}\n'.format(i, y))



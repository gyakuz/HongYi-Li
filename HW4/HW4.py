
import os
import json
import torch
import random
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from attention import RelativeMultiHeadSelfAttention
from modules import ConvolutionModule, FeedForwardModule


class myDataset(Dataset):
    def __init__(self, data_dir, segment_len=128):
        self.data_dir = data_dir
        self.segment_len = segment_len

        #speaker to  id.
        mapping_path = Path(data_dir) / "mapping.json"
        mapping = json.load(mapping_path.open())
        self.speaker2id = mapping["speaker2id"]

        # "n_mels": The dimention of mel-spectrogram.
        metadata_path = Path(data_dir) / "metadata.json"
        metadata = json.load(open(metadata_path))["speakers"]

        self.speaker_num = len(metadata.keys())   #key返回关键字集合
        self.data = []
        for speaker in metadata.keys():
            for utterances in metadata[speaker]:
                self.data.append([utterances["feature_path"], self.speaker2id[speaker]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feat_path, speaker = self.data[index]
        #加载mel
        mel = torch.load(os.path.join(self.data_dir, feat_path))

        # 把mel分割成 "segment_len" 个部分
        if len(mel) > self.segment_len:
            start = random.randint(0, len(mel) - self.segment_len)
            # 长度为segment_len.
            mel = torch.FloatTensor(mel[start:start + self.segment_len])
        else:
            mel = torch.FloatTensor(mel)
        # 将speaker id 变成long型为了后面计算损失
        speaker = torch.FloatTensor([speaker]).long()
        return mel, speaker

    def get_speaker_number(self):
        return self.speaker_num

class ConformerBlock(nn.Module):
    def __init__(
        self,
        d_model=1024,
        num_heads=4,
        max_len=512,
        expansion_factor=4,
        kernel_size=31,
        dropout=0.1,
    ):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, expansion_factor, dropout)
        self.attn = RelativeMultiHeadSelfAttention(d_model, num_heads, max_len, dropout)
        self.conv = ConvolutionModule(d_model, kernel_size, dropout)
        self.ff2 = FeedForwardModule(d_model, expansion_factor, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x.shape == (batch_size, seq_len, d_model)
        # half step residual connection
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(x)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        x = self.norm(x)
        return x

def collate_batch(batch):
  mel, speaker = zip(*batch)
  mel = pad_sequence(mel, batch_first=True, padding_value=-20)    # 只能接受长度相同的序列输入。因此如果目前序列长度参差不齐，这时需要使用pad_sequences()。
  # mel: (batch size, length, 40)                                 # 该函数是将序列转化为经过填充以后的一个长度相同的新序列新序列。
  return mel, torch.FloatTensor(speaker).long()


def get_dataloader(data_dir, batch_size, n_workers):
  dataset = myDataset(data_dir)
  speaker_num = dataset.get_speaker_number()
  # 分割训练集和测试集
  trainlen = int(0.9 * len(dataset))
  lengths = [trainlen, len(dataset) - trainlen]
  trainset, validset = random_split(dataset, lengths)

  train_loader = DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=n_workers,
    pin_memory=True,
    collate_fn=collate_batch,
  )
  valid_loader = DataLoader(
    validset,
    batch_size=batch_size,
    num_workers=n_workers,
    drop_last=True,
    pin_memory=True,
    collate_fn=collate_batch,
  )

  return train_loader, valid_loader, speaker_num


class Classifier(nn.Module):
  def __init__(self,
        d_model=1024,
        n_spks=600,
        num_blocks=3,
        dropout=0.1,
        num_heads=8,
        max_len=512,
        expansion_factor=4,
        kernel_size=31,):
    super().__init__()
    # Project the dimension of features from that of input into d_model.
    self.prenet = nn.Linear(40, d_model)

    #self.encoder_layer = nn.TransformerEncoderLayer( d_model=d_model, dim_feedforward=256, nhead=8)
    # self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
    self.block=ConformerBlock(d_model, num_heads, max_len, expansion_factor, kernel_size, dropout)
    self.conformer=self._make_layer(d_model, num_heads, max_len, expansion_factor, kernel_size, dropout,num_blocks)
    self.pred_layer = nn.Sequential(
      nn.Linear(d_model, d_model),
      nn.ReLU(),
      nn.Linear(d_model, n_spks),
    )

  def _make_layer(self, d_model, num_heads, max_len, expansion_factor, kernel_size, dropout,num_blocks):
        layers = []

        for i in range(1, num_blocks):
            layers.append(ConformerBlock(d_model, num_heads, max_len, expansion_factor, kernel_size, dropout))
        return nn.Sequential(*layers)


  def forward(self, mels):

      #mels: (batch size, length, 40)
      #return out: (batch size, n_spks)

    # out: (batch size, length, d_model)
    out = self.prenet(mels)
    # out: (length, batch size, d_model)
    #out = out.permute(1, 0, 2)
    #out = self.encoder_layer(out)

    out = self.conformer(out)

    # out: (batch size, length, d_model)
    #out = out.transpose(0, 1)
    # mean pooling
    stats = out.mean(dim=1)

    # out: (batch, n_spks)
    out = self.pred_layer(stats)
    return out


def get_cosine_schedule_with_warmup(
  optimizer: Optimizer,
  num_warmup_steps: int,   #The number of steps for the warmup phase.
  num_training_steps: int, #The total number of training steps.
  num_cycles: float = 0.5,  #
  last_epoch: int = -1,
):

  def lr_lambda(current_step):
    # Warmup
    if current_step < num_warmup_steps:
      return float(current_step) / float(max(1, num_warmup_steps))
    # decadence
    progress = float(current_step - num_warmup_steps) / float(
      max(1, num_training_steps - num_warmup_steps)
    )
    return max(
      0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    )

  return LambdaLR(optimizer, lr_lambda, last_epoch)


def model_fn(batch, model, criterion, device):

  mels, labels = batch
  mels = mels.to(device)
  labels = labels.to(device)

  outs = model(mels)

  loss = criterion(outs, labels)

  preds = outs.argmax(1)

  accuracy = torch.mean((preds == labels).float())

  return loss, accuracy


def valid(dataloader, model, criterion, device):

  model.eval()
  running_loss = 0.0
  running_accuracy = 0.0
  pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")

  for i, batch in enumerate(dataloader):
    with torch.no_grad():
      loss, accuracy = model_fn(batch, model, criterion, device)
      running_loss += loss.item()
      running_accuracy += accuracy.item()
    #进度条
    pbar.update(dataloader.batch_size)
    pbar.set_postfix(
      loss=f"{running_loss / (i+1):.2f}",
      accuracy=f"{running_accuracy / (i+1):.2f}",
    )

  pbar.close()
  model.train()

  return running_accuracy / len(dataloader)


def parse_args():
    config = {
        "data_dir": "./Dataset",
        "save_path": "model.ckpt",
        "batch_size": 32,
        "n_workers": 0,
        "valid_steps": 2000,
        "warmup_steps": 1000,
        "save_steps": 2000,
        "total_steps": 200000,
    }

    return config


def main(
        data_dir,
        save_path,
        batch_size,
        n_workers,
        valid_steps,
        warmup_steps,
        total_steps,
        save_steps,
):

    device = "cuda"

    train_loader, valid_loader, speaker_num = get_dataloader(data_dir, batch_size, n_workers)
    train_iterator = iter(train_loader)
    print(f"[Info]: Finish loading data", flush=True)

    model = Classifier(n_spks=speaker_num).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"[Info]: Finish creating model!", flush=True)

    best_accuracy = -1.0
    best_state_dict = None
    #model.load_state_dict(torch.load('./model.ckpt'))
    pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

    for step in range(total_steps):
        # Get data
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        loss, accuracy = model_fn(batch, model, criterion, device)
        batch_loss = loss.item()
        batch_accuracy = accuracy.item()

        # 梯度降
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # 进度条
        pbar.update()
        pbar.set_postfix(
            loss=f"{batch_loss:.2f}",
            accuracy=f"{batch_accuracy:.2f}",
            step=step + 1,
        )

        # Do validation
        if (step + 1) % valid_steps == 0:
            pbar.close()

            valid_accuracy = valid(valid_loader, model, criterion, device)

            # keep the best model
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                best_state_dict = model.state_dict()

            pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

        # Save the best model so far.
        if (step + 1) % save_steps == 0 and best_state_dict is not None:
            torch.save(best_state_dict, save_path)
            pbar.write(f"Step {step + 1}, best model saved. (accuracy={best_accuracy:.4f})")

    pbar.close()


if __name__ == "__main__":
    main(**parse_args())
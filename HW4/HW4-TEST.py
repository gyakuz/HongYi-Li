import os
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset
import json
import csv
from pathlib import Path
from tqdm.notebook import tqdm
import random
import torch
from torch.utils.data import DataLoader
from HW4 import Classifier

class InferenceDataset(Dataset):
  def __init__(self, data_dir,segment_len=128):
    testdata_path = Path(data_dir) / "testdata.json"
    metadata = json.load(testdata_path.open())
    self.data_dir = data_dir
    self.data = metadata["utterances"]
    self.segment_len = segment_len
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    utterance = self.data[index]
    feat_path = utterance["feature_path"]
    mel = torch.load(os.path.join(self.data_dir, feat_path))
    if len(mel) > self.segment_len:
        start = random.randint(0, len(mel) - self.segment_len)
        # 长度为segment_len.
        mel = torch.FloatTensor(mel[start:start + self.segment_len])
    else:
        mel = torch.FloatTensor(mel)

    return feat_path, mel


def inference_collate_batch(batch):
  """Collate a batch of data."""
  feat_paths, mels = zip(*batch)

  return feat_paths, torch.stack(mels)


def parse_args():
    """arguments"""
    config = {
        "data_dir": "./Dataset",
        "model_path": "./model1.ckpt",
        "output_path": "./output.csv",
    }

    return config


def main(
        data_dir,
        model_path,
        output_path,
):
    """Main function."""
    device ="cuda"

    mapping_path = Path(data_dir) / "mapping.json"
    mapping = json.load(mapping_path.open())

    dataset = InferenceDataset(data_dir)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=inference_collate_batch,
    )
    print(f"[Info]: Finish loading data!", flush=True)

    speaker_num = len(mapping["id2speaker"])

    model = Classifier(n_spks=speaker_num).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"[Info]: Finish creating model!", flush=True)

    results = [["Id", "Category"]]
    for feat_paths, mels in tqdm(dataloader):
        with torch.no_grad():
            mels = mels.to(device)
            outs = model(mels)
            preds = outs.argmax(1).cpu().numpy()
            for feat_path, pred in zip(feat_paths, preds):
                results.append([feat_path, mapping["id2speaker"][str(pred)]])

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)


if __name__ == "__main__":
    main(**parse_args())

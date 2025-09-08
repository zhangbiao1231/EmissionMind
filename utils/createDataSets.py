
import random

import os
from pathlib import Path
import argparse
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # backfire root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


# 随机分区
def seq_data_iter_random(features, labels, batch_size, num_steps):
    """使用随机采样生成小批量序列"""
    offset = random.randint(0, num_steps - 1)
    features = features[offset:]
    labels = labels[offset:]

    num_subseqs = (len(features) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)

    num_batches = len(initial_indices) // batch_size
    for i in range(0, num_batches * batch_size, batch_size):
        batch_indices = initial_indices[i: i + batch_size]
        X = [torch.tensor(features[j: j + num_steps], dtype=torch.float32)
             for j in batch_indices]
        Y = [torch.tensor(labels[j + 1: j + 1 + num_steps], dtype=torch.float32)
             for j in batch_indices]
        yield torch.stack(X), torch.stack(Y)

# 顺序分区
def seq_data_iter_sequential(features, labels, batch_size, num_steps):
    """使用顺序分区生成小批量序列"""
    offset = random.randint(0, num_steps)
    num_tokens = ((len(features) - offset - 1) // batch_size) * batch_size

    Xs = features[offset: offset + num_tokens]
    Ys = labels[offset + 1: offset + 1 + num_tokens]

    Xs = Xs.reshape(batch_size, -1, features.shape[-1])  # (B, L, D)
    Ys = Ys.reshape(batch_size, -1)                      # (B, L)

    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_batches * num_steps, num_steps):
        X = Xs[:, i:i + num_steps, :]
        Y = Ys[:, i:i + num_steps]
        yield torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
class SeqDataLoader:  #@save
    """加载序列数据的迭代器"""
    def __init__(self, features, labels, batch_size, num_steps, use_random_iter):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.features, self.labels = features, labels
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.features, self.labels, self.batch_size, self.num_steps)

    def __len__(self):
        num_subseqs = (len(self.features) - 1) // self.num_steps
        return num_subseqs // self.batch_size

class TimeSeriesDataset(Dataset):
    def __init__(self, data_dir, num_steps, scaler_X = None, scaler_y = None,fit_scaler=False,window=5):
        """
        Args:
            - data_dir (str | Path): 存放 CSV 的文件夹
            - num_steps (int): 序列长度
            - scaler_X, scaler_y: 用于归一化数据集
            - fit_scaler: 是否对数据集进行归一化，默认为 False
        """
        self.data_dir = data_dir
        self.num_steps = num_steps

        self.X, self.y = self._load_all_csv(window=window)

        # 归一化
        if fit_scaler:
            self.scaler_X = MinMaxScaler().fit(self.X.reshape(-1,self.X.shape[-1]))
            self.scaler_y = MinMaxScaler().fit(self.y.reshape(-1,1))
        else:
            self.scaler_X = scaler_X
            self.scaler_y = scaler_y
        if self.scaler_X is not None:
            self.X = self.scaler_X.transform(self.X.reshape(-1,self.X.shape[-1])).reshape(self.X.shape)
        if self.scaler_y is not None:
            self.y = self.scaler_y.transform(self.y.reshape(-1,1)).reshape(self.y.shape)

    def _seq_data_from_df(self, df):
        """针对单个 CSV 生成时序样本"""
        features = df.iloc[:, 1:-1].values  # 去掉时间列，最后一列是标签
        labels = df.iloc[:, -1].values

        Xs, Ys = [], []
        for i in range(len(features) - self.num_steps-1):
            Xs.append(features[i:i+self.num_steps])
            Ys.append(labels[i+self.num_steps+1])
        return np.array(Xs), np.array(Ys)

    def _load_all_csv(self,window):
        """加载文件夹中所有 CSV 并拼接"""
        all_X, all_y = [], []
        for csv_file in sorted(self.data_dir.glob("*.csv")):
            df = pd.read_csv(csv_file)
            if len(df) > self.num_steps:
                X, y = self._seq_data_from_df(df)
                all_X.append(X)
                all_y.append(y)

        if not all_X:
            raise ValueError(f"目录 {self.data_dir} 中没有足够长的 CSV")

        return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)
    def inverse_transform(self,y_scaled):
        """
        - y_scaled: numpy array or torch tensor
        """
        if self.scaler_y is None:
            raise ValueError("scaler_y is None")
        if isinstance(y_scaled,torch.Tensor):
            y_scaled=y_scaled.detach().numpy()
        return self.scaler_y.inverse_transform(y_scaled.reshape(-1,1)).reshape(-1)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-folder", type=str, default=ROOT / "min-level/datasets/processed/train", help="train dataset path")
    parser.add_argument("--valid-folder", type=str, default=ROOT / "min-level/datasets/processed/valid", help="valid dataset path")
    parser.add_argument("--test-folder", type=str, default=ROOT / "min-level/datasets/processed/test", help="test dataset path")

    parser.add_argument("--use-random", type=bool, default=False, help="use random concat")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--num-steps", type=int, default=10, help="number of steps")
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':

    opt = parse_opt()
    # =======================================训练集==========================================#
    train_dataset = TimeSeriesDataset(ROOT/ "datasets/min-level/processed/train", num_steps=10, scaler_X=None,scaler_y=None,fit_scaler=True)
    scaler_X, scaler_y = train_dataset.scaler_X, train_dataset.scaler_y

    valid_dataset = TimeSeriesDataset(ROOT/ "datasets/min-level/processed/valid", num_steps=10,scaler_X=scaler_X,scaler_y=scaler_y,fit_scaler=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    for X, y in train_loader:
        print(f"X: {X.shape}, y: {y.shape}")
        break
    valid_loader = DataLoader(valid_dataset, batch_size=32)

    y_hat_scaled = torch.rand(5)
    print(f"y_hat_scaled: {y_hat_scaled}")

    y_hat_pred = valid_dataset.inverse_transform(y_hat_scaled)
    print(f"y_hat_pred: {y_hat_pred}")


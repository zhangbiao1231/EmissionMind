# flameMind 🔥, 1.0.0 license
"""Plotting utils."""
import contextlib
import math
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np

# from utils.general import LOGGER, increment_path

# Settings
RANK = int(os.getenv("RANK", -1))
matplotlib.rc("font", **{"size": 14})
matplotlib.use("Agg")  # for writing to files only
import pandas as pd
# 测试集验证
def plot_comparison(fname):
    df =  pd.read_csv(fname)
    file_stem = Path(fname).stem.replace("_result", "")
    save_path = Path(fname).parent / f"{file_stem}_plot.png"
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # 准确率计算
    mae = np.mean(np.abs(df["true_label"] -df["prediction"]))
    rmse = np.sqrt(np.mean((df["true_label"] -df["prediction"]) ** 2))
    ss_res = np.sum((df["true_label"] -df["prediction"]) ** 2)
    ss_tot = np.sum((df["true_label"] - np.mean(df["prediction"])) ** 2)
    R2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    # 绘制真实值（绿线）
    plt.plot(df['time_index'], df['true_label'], color='green', label='True')

    # 绘制预测值（红点）
    plt.scatter(df['time_index'], df['prediction'], color='red', label='Pred', s=10)

    # 添加图例和标签
    plt.title(f"Nox Prediction - {file_stem}")
    plt.title(f"{file_stem} \n-MAE: {mae:.2f} -RMSE: {rmse:.2f} -R2: {R2:.2f}" ,fontsize=13)# 绘制准确率信息，后续补充延迟时间
    plt.xlabel('Time (min)')
    plt.ylabel('Nox')
    plt.legend(loc="upper left")
    plt.grid(True)

    # 显示图像
    plt.tight_layout()
    plt.show()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"图像已保存：{save_path}")
# 训练过程
def plot_train(fname):
    df =  pd.read_csv(fname)
    df.columns = df.columns.str.strip()
    # 设置画布大小
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    ax.plot(df['epoch'], df['train/loss'], 'b-', label=r'Train Loss')
    ax.plot(df['epoch'], df['valid/loss'], 'r--', label='Valid Loss')

    # 添加图例和标签
    ax.set_xlabel('epoch',fontsize=14)
    ax.set_ylabel('Loss',fontsize=14)
    ax.set_title('Train Loss & Valid Accuracy vs. Epoch')
    plt.legend(loc="upper left")
    plt.grid(True)

    ax1 = ax.twinx()
    ax1.plot(df['epoch'], df['metrics/accuracy'], 'g', label='Valid Accuracy')
    ax1.set_ylabel('Accuracy',fontsize=14)
    plt.legend(loc="upper right")
    # plt.grid(True)

    # 显示图像
    plt.tight_layout()
    plt.show()

    plt.savefig("train.png", dpi=300, bbox_inches="tight")
    plt.close()
if __name__ == '__main__':
    # fname = "/Users/zebulonzhang/deeplearning/FlameMind/hengqin_3_comparison.csv"
    # plot_comparison(fname)
    fname1 = "/Users/zebulonzhang/deeplearning/FlameMind/runs/train-cls/exp18/results.csv"
    plot_train(fname1)
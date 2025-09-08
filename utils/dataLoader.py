# dataLoader.py

import os
import sys
from pathlib import Path
import pandas as pd
from os.path import join

from pathlib import Path
import shutil
import random
import pandas as pd
from os.path import join
import numpy as np
import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # backfire root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

def add_valve_features(df: pd.DataFrame,eps=1e-6,) -> pd.DataFrame:
    """
    生成阀门派生特征，包括占比、差值、比值、滚动、滞后等
    使用前值填充 NaN，避免空 DataFrame
    """
    df = df.copy()
    valve_cols = ["diffusion_valve_feedback", "premix_valve_feedback", "duty_valve_feedback"]
    avail_valves = [c for c in valve_cols if c in df.columns]
    if len(avail_valves) == 0:
        return df

    # 总开度
    df["valve_total_open"] = df[avail_valves].sum(axis=1)

    # 占比
    for c in avail_valves:
        df[f"{c}_share"] = df[c] / (df["valve_total_open"] + eps)

    return df

import numpy as np
import pandas as pd

def add_temp_features(
    df: pd.DataFrame,
    temp_cols=None,
    roll_windows=(5, 15, 60),
    lags=(1, 5),
    eps=1e-6
) -> pd.DataFrame:
    """
    为一组温度列生成衍生特征（温差、比值、滚动统计、差分、滞后、异常标记等）。
    - df: 原始 DataFrame（要求列名包含在 temp_cols）
    - temp_cols: None 或 list，若 None 则使用默认列名称
    - 返回含新增特征的 df（不做 dropna，使用前向填充）
    """
    df = df.copy()
    if temp_cols is None:
        temp_cols = [
            "compressor_inlet_temp",
            "compressor_outlet_temp",
            "turbine_exhaust_temp_10B",
            "NG_inlet_temp",
            "amb_temperature",
        ]

    # 只保留存在的列
    temp_cols = [c for c in temp_cols if c in df.columns]
    if not temp_cols:
        return df

    # ============ 基本温差/比值 ============
    # 压气机升温
    if "compressor_outlet_temp" in df.columns and "compressor_inlet_temp" in df.columns:
        df["comp_out_minus_in"] = df["compressor_outlet_temp"] - df["compressor_inlet_temp"]
        df["comp_out_over_in"] = df["compressor_outlet_temp"] / (df["compressor_inlet_temp"] + eps)

    # 排气与压气机出口
    if "turbine_exhaust_temp_10B" in df.columns and "compressor_outlet_temp" in df.columns:
        df["exhaust_minus_comp_out"] = df["turbine_exhaust_temp_10B"] - df["compressor_outlet_temp"]
        df["exhaust_over_comp_out"] = df["turbine_exhaust_temp_10B"] / (df["compressor_outlet_temp"] + eps)

    # 排气与环境
    if "turbine_exhaust_temp_10B" in df.columns and "amb_temperature" in df.columns:
        df["exhaust_minus_amb"] = df["turbine_exhaust_temp_10B"] - df["amb_temperature"]

    # 燃气影响（若有）
    if "NG_inlet_temp" in df.columns and "turbine_exhaust_temp_10B" in df.columns:
        df["exhaust_minus_NG_in"] = df["turbine_exhaust_temp_10B"] - df["NG_inlet_temp"]

    # ============ 滑动统计（均值/标准差） ============
    for w in roll_windows:
        for c in temp_cols:
            df[f"{c}_ma{w}"] = df[c].rolling(window=w, min_periods=1).mean()
            df[f"{c}_std{w}"] = df[c].rolling(window=w, min_periods=1).std().fillna(0)

    # 对温差也做滚动
    derived_diff_cols = [col for col in df.columns
                         if ("_minus_" in col) or col.endswith("_minus_in") or col.endswith("_minus_amb")
                         or col.endswith("_minus_NG_in")]
    for w in roll_windows:
        for c in derived_diff_cols:
            df[f"{c}_ma{w}"] = df[c].rolling(window=w, min_periods=1).mean()

    # ============ 一阶差分与变化率 ============
    for c in temp_cols:
        df[f"{c}_diff1"] = df[c].diff().fillna(0)
        df[f"{c}_rel_change"] = (df[f"{c}_diff1"] / (df[c].shift(1) + eps)).fillna(0)

    # 对温差也做差分
    for c in derived_diff_cols:
        df[f"{c}_diff1"] = df[c].diff().fillna(0)
        df[f"{c}_rel_change"] = (df[f"{c}_diff1"] / (df[c].shift(1) + eps)).fillna(0)

    return df
def reorder_columns(df: pd.DataFrame, target_col="NOx_in_flue_gas") -> pd.DataFrame:
    """把目标列移到最后"""
    if target_col in df.columns:
        cols = [c for c in df.columns if c != target_col] + [target_col]
        df = df[cols]
    return df

def load_and_process_data(df: pd.DataFrame,)-> pd.DataFrame:

    if "load" in df.columns:
        df["load"] = df["load"].clip(lower=0)

    df = add_valve_features(df,eps=1e-6)
    # df = add_temp_features(
    #     df,
    #     lags=(1, 5),
    #     roll_windows=(5, 15, 60),
    #     eps=1e-6,
    # )

    # 保留两位小数
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].round(2)
    return df

def split_df_by_time(df, time_col="Time", freq="1min", save_dir=None):
    """
    将不连续时间序列的 DataFrame 按连续段拆分，并保存成多个 CSV 文件

    参数:
    - df: 原始 DataFrame
    - time_col: 时间列名称
    - freq: 期望的连续频率，默认 '1min'
    - save_dir: 保存的目录

    返回:
    - segments: 拆分后的 DataFrame 列表
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    #计算相邻时间差（分钟）
    diffs = df[time_col].diff().dt.total_seconds().div(60).fillna(0)

    # 连续段的断点：diff > 期望频率
    expected_minutes = pd.Timedelta(freq).total_seconds() / 60
    segment_ids = (diffs > expected_minutes).cumsum()

    # 按 segment_id 分组
    segments = []
    save_path = save_dir
    save_path.mkdir(parents=True, exist_ok=True)

    for seg_id, seg_df in df.groupby(segment_ids):
        seg_df = seg_df.reset_index(drop=True)
        segments.append(seg_df)

        # 保存 CSV
        start = seg_df[time_col].iloc[0].strftime("%Y%m%d_%H%M")
        end = seg_df[time_col].iloc[-1].strftime("%Y%m%d_%H%M")
        filename = save_path / f"segment_{seg_id}_{start}_to_{end}.csv"
        seg_df.to_csv(filename, index=False)

    print(f"共拆分为 {len(segments)} 段，CSV 已保存到 {save_path}")
    return segments


# ============ 批处理函数 ============
def process_all_csv(raw_dir="datasets/min-level/raw",
                    out_dir="datasets/min-level/processed"):
    # raw_dir = Path(raw_dir)
    # out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for file in raw_dir.glob("*.csv"):
        try:
            print(f"Processing {file.name} ...")
            df = pd.read_csv(file)

            # 特征处理
            df = load_and_process_data(df)

            # 确保标签在最后一列
            df = reorder_columns(df, target_col="NOx_in_flue_gas")

            # 保存
            out_name = f"processed_{file.name}"
            df.to_csv(out_dir / out_name, index=False, float_format="%.2f")

            print(f"Saved to {out_dir/out_name}")
        except Exception as e:
            print(f"❌ Failed {file.name}: {e}")

def split_csv_files(data_dir, ratios=(0.75, 0.15, 0.1), seed=42):
    """
    按比例划分 CSV 文件到 train/valid/test 文件夹
    """
    # data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob("*.csv"))
    random.seed(seed)
    random.shuffle(csv_files)

    n = len(csv_files)
    n_train = int(n * ratios[0])
    n_valid = int(n * ratios[1])
    # n_test = n - n_train - n_valid

    splits = {
        "train": csv_files[:n_train],
        "valid": csv_files[n_train:n_train+n_valid],
        "test": csv_files[n_train+n_valid:],
    }

    for split, files in splits.items():
        split_dir = data_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy(f, split_dir / f.name)

    print(f"CSV 文件划分完成: train={len(splits['train'])}, valid={len(splits['valid'])}, test={len(splits['test'])}")
    return splits

def main(opt):
    file_path = opt.filepath
    df = pd.read_csv(file_path) # 2023
    save_dir = opt.input
    out_dir = opt.output



    # df = pd.read_csv(file_path / "GT2_2024_selected_features_1s.csv")
    # save_dir = opt.data_root / "second-level/raw"
    # #================================================================================#
    segments = split_df_by_time(df, time_col="Time", freq="1min", save_dir=save_dir)
    # segments = split_df_by_time(df, time_col="Time", freq="1s", save_dir=save_dir)
    # # ================================================================================#
    process_all_csv(raw_dir=save_dir,
                    out_dir=out_dir)
    # process_all_csv(raw_dir=save_dir,
    #                 out_dir=opt.data_root / "second-level/processed")
    # ================================================================================#
    split_csv_files(data_dir=out_dir, ratios=opt.ratios, seed=42)
    # split_csv_files(data_dir=opt.data_root / "second-level/processed", ratios=(0.75, 0.15, 0.1), seed=42)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str,
                        default=ROOT / "datasets/GT2_2024_selected_features_1min.csv",
                        help="source data path")
    parser.add_argument("--input", type=str,
                        default=ROOT / "datasets/min-level/raw",
                        help="raw data path")
    parser.add_argument("--output", type=str,
                        default=ROOT / "datasets/min-level/processed",
                        help="data path for processed data")
    parser.add_argument("--ratios", type=float, nargs="+",
                        default=(0.75, 0.15, 0.1),
                        help="ratios of train/valid/test set, e.g., 0.75 0.15 0.1")
    opt = parser.parse_args()
    return opt
if __name__ == '__main__':
    opt = parse_opt()
    main(opt)



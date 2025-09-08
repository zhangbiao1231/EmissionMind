# flame-mind 🔥, 1.0.0 license

import argparse
import os
import sys
from pathlib import Path
import time

import torch

import numpy as np
import pandas as pd

from collections import deque
from tqdm import tqdm

import matplotlib.pyplot as plt
plt.ioff()
from sklearn.metrics import (mean_absolute_error,
                             root_mean_squared_error,
                             r2_score,)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.plots import plot_comparison
from utils.general import (
    LOGGER,
    Profile,
    colorstr,
    intersect_dicts,
    increment_path,
    print_args,
)

from utils.general import try_gpu

RANK = int(os.getenv("RANK", -1))

def run(
        source="",  # dataset dir
        weights="",  # model.pt path(s)
        project="",  # save to project/name
        name="",  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=False,  # use FP16 half-precision inference
        model=None,
        dataloader=None,
        criterion=None,
        maxlen=10,
):
    device = try_gpu()

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    csv = save_dir / "results.csv"

    # Load model
    weights = str(weights[0]) if isinstance(weights, (list, tuple)) else str(weights)
    ckpt = torch.load(weights, map_location="cpu",weights_only=False)  # 直接加载model
    model = ckpt["model"].to(device)
    LOGGER.info(f"Loaded full model from {weights}")  # report

    # Recover scaler
    import pickle
    scalers = pickle.loads(ckpt['scaler'])
    scaler_X, scaler_y = scalers["X"], scalers["y"]

    # Predictor Class
    class Predictor:
        def __init__(self, model, maxlen=10, dims=80, scalers=None, device=device):
            self.device = device
            self.model = model.to(self.device)
            self.maxlen = maxlen
            self.dims = dims
            self.scalers = scalers
            self.buffer = deque([torch.zeros(self.dims) for _ in range(maxlen)], maxlen=maxlen)
            self.results = []

        def update_buffer(self, new_data):
            self.buffer.append(new_data)

        def predict_next(self):
            X_input = np.array(self.buffer) # (maxlen, dims)
            X_scaled = self.scalers["X"].transform(X_input)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, maxlen, dims)

            # Forward
            with torch.no_grad():
                y_hat, _ = self.model(X_tensor,) # (1, maxlen)
            y_hat = y_hat[:,-1].squeeze().cpu().item()  # (maxlen, ) -> (1, )
            self.results.append(y_hat)
            return self.results

    # Run inference
    # Inference on data
    def run_inference(df, predictor, file, maxlen):
        start_time = time.time()
        n = len(df)  # number of samples

        for t in tqdm(range(n), desc=f"{file} 推理中", ncols=80):
            current_data = df.iloc[t, :-1].values  # shape = (10,),
            predictor.update_buffer(current_data)
            if t >= maxlen - 1 and t < n-1:
                predictor.predict_next()

        end_time = time.time()  # ⏱️ 结束计时
        elapsed = end_time - start_time
        return predictor.results,elapsed

    def evaluate_inference(file, df, results, maxlen, save_dir):
        # 提取真实标签
        true_labels = df.iloc[maxlen:,-1].values.astype(np.float32) # (maxlen, 1)

        preds_labels = np.array(results, dtype=np.float32).reshape(-1, 1)  # 变成 (n_samples, 1)
        print(preds_labels.shape)
        pred_labels = scaler_y.inverse_transform(preds_labels).reshape(-1) # (maxlen, 1)

        if len(true_labels) != len(pred_labels):
            m = min(len(true_labels), len(pred_labels))
            pred_labels, true_labels = pred_labels[:m], true_labels[:m]

        assert len(true_labels) == len(pred_labels), "预测数量与真实标签数量不一致！"

        # 推理准确率
        mae = mean_absolute_error(true_labels, pred_labels)
        rmse = root_mean_squared_error(true_labels, pred_labels)
        R2 = r2_score(true_labels, pred_labels)

        elapsed = df.attrs.get("elapsed", 1e-6)  # 容错（确保非零）
        time_per_sample = elapsed / len(true_labels) * 1000.0

        print(f"文件: {file:<22} 数量: {len(true_labels):<10} 单位推理时间:{time_per_sample:<8.2f}ms"
              f"MAE: {mae:<7.2f} RMSE:{rmse:<7.2f} R2:{R2:<7.2f}")

        # 保存预测结果
        df_compare = pd.DataFrame({
            'time_index': list(range(maxlen, maxlen + len(true_labels))),
            'true_label': true_labels,
            'prediction': pred_labels,
        })
        df_compare.to_csv(save_dir / f"{Path(file).stem}_result.csv", index=False)

        return mae, rmse, R2, time_per_sample, len(true_labels)
    # Load files
    files = []
    source_path = Path(source)
    if source_path.is_file():
        files = [source_path.name]
        source = source_path.parent
    else:
        files = [f for f in os.listdir(source) if f.endswith(".csv")]

    all_results = []
    for file in files:
        print(f"\n开始推理文件: {file}")
        df = pd.read_csv(Path(source) / file)
        if "Time" in df.columns or "timestamp" in df.columns:
            df = df.drop(columns=["Time"], errors="ignore")
        predictor = Predictor(model, maxlen=maxlen, dims=df.shape[-1], scalers=scalers, device="cpu")
        results, elapsed= run_inference(df, predictor,file, maxlen)

        df.attrs["elapsed"] = elapsed  # 传入评估函数

        mae, rmse, R2, time_per_sample,length = evaluate_inference(file, df, results, maxlen, save_dir)

        all_results.append({
            "file": file,
            "count": length,
            "time_per_sample": time_per_sample,
            "mae": mae,
            "rmse": rmse,
            "R2": R2,
            "elapsed": elapsed,
        })
    # 表头
    print(f"\n{'FileName':<60}{'Quantities':<20}{'MAE':<12}{'RMSE':<12}{'R2':<12}{'UnitTime':<12}")
    print("-" * 120)

    # 每行数据
    for r in all_results:
        print(f"{r['file']:<30}{r['count']:<20}{r['mae']:<12.2f}{r['rmse']:<12.2f}{r['R2']:<12.2f}{r['time_per_sample']:<12.2f} ")

    if RANK in {-1, 0}:
        LOGGER.info(
            f"\nResults saved to {colorstr('bold', save_dir)}",
        )
    # 绘制图像并保存
    for csv_file in save_dir.glob("*_result.csv"):
        plot_comparison(fname=csv_file)

def parse_opt():
    """Parses command line arguments for YOLOv5 inference settings including model, source, device, and image size."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=ROOT / "datasets/min-level/processed/test",
                        help="dataset path")
    # parser.add_argument("--model", type=str, default=None, help="initial weights path")
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "runs/train-reg/exp20/weights/best.pt",
                        help="model.pt path(s)")
    parser.add_argument("--project", default=ROOT / "runs/inference-reg", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    # parser.add_argument("--maxlen", default=10, type=int, help="max length of buffer")
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt

def main(opt):
    """Executes YOLOv5 model inference with options for ONNX DNN and video frame-rate stride adjustments."""
    # check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
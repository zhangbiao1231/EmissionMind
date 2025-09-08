import pandas as pd
from pathlib import Path
import sys
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # backfire root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))
def select_features(input_file, output_file, feature_cols):
    """
    从原始CSV提取指定特征并保存
    :param input_file: str, 原始CSV文件路径
    :param output_file: str, 新CSV文件路径
    :param feature_cols: list, 需要保留的列名
    """
    df = pd.read_csv(input_file)

    # 只保留需要的列
    df_selected = df[feature_cols]

    # 保存到新文件
    df_selected.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"已保存: {output_file}, shape={df_selected.shape}")

    return df
if __name__ == "__main__":
    # 示例使用
    data_dir = ROOT / "processed_data"
    input_file =data_dir/ "select_data_after_firing_GT2_2023_resample_1s.csv"
    output_file = ROOT/"datasets"/ "GT2_2023_selected_features_1s.csv"

    # 你需要保留的特征（按你的需求修改）
    feature_cols = [
        "Time", # 时间列
        "load", # 负荷
        "diffusion_valve_feedback", # 扩散气
        "premix_valve_feedback", # 预混气
        "duty_valve_feedback", # 值班气
        # "diffusion_fuel_mass_flow_set",
        # "premix_fuel_mass_flow_set",
        # "duty_fuel_mass_flow_set",
        "compressor_inlet_temp", # 压气机入口温度
        "compressor_outlet_temp", # 压气机排气温度
        "turbine_exhaust_temp_10B", # 排气温度
        "NG_inlet_temp", # 燃气温度
        "amb_humidity", # 空气湿度
        "amb_temperature", # 空气温度
        "NOx_in_flue_gas", # NOx 值

        # "GT_load_control_mode", #
        # "GT_initiate_control_mode", #
        # "GT_temp_control_mode", #
        # "GT_pres_ratio_control_mode", #
        # "GT_speed_control_mode", #
    ]
    select_features(input_file, output_file, feature_cols)

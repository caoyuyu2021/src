from init_env import init_path
import pandas as pd
from models.iTransformer import iTransformer
from utils.predict import predict
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    ts_data = pd.read_csv(init_path() + "data/energy.csv").iloc[-300:, :]
    ts_data['time'] = pd.to_datetime(ts_data['time'])
    # 构造参数字典
    params = {
        "task_args": {
            "columns": ['load', 'temp'],
            "target": ['load', 'temp'],
            "features": 'M',
        },
        "predict_args": {
            "time_col": 'time',
            "target_col": ['temp'],
            "freq": 'h',
            "model_name": iTransformer,
            "model_path": init_path() + "outputs/best_models/iTransformer/checkpoint.pth",
            "x_true": ts_data,
            "scaler_path": init_path() + 'outputs/scalers/iTransformer',
            "device": 'cpu'
        },
        "model_args": {
            'seq_len': 6,
            'pred_len': 1,
            "label_len": 3,
            'output_attention': False,
            'd_model': 32,
            'n_heads': 8,
            'd_ff': 32,
            'dropout': 0.1,
            'e_layers': 1,
            'activation': "relu",
        },
    }
    y_pred = predict(**params)
    print(y_pred)
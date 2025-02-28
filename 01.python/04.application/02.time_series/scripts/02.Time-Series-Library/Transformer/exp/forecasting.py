from init_env import init_path
import pandas as pd
from models.Transformer import Transformer
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
            "model_name": Transformer,
            "model_path": init_path() + "outputs/best_models/Transformer/checkpoint.pth",
            "x_true": ts_data,
            "scaler_path": init_path() + 'outputs/scalers/Transformer',
            "device": 'cpu'
        },
        "model_args": {
            'seq_len': 6,
            'pred_len': 1, 
            'label_len': 3,
            'output_attention': True,
            'embed': 'timeF', 
            'freq': 'h',
            'd_model': 512,
            'enc_in': 2,
            'dec_in': 2,
            'dropout': 0.1,
            'factor': 3,
            'n_heads': 8,
            'd_ff': 256,
            'e_layers': 2,
            'd_layers': 2,
            'c_out': 2
        },
    }
    y_pred = predict(**params)
    print(y_pred)
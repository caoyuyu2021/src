from init_env import init_path
import pandas as pd
from utils.predict import predict
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    ts_data = pd.read_csv(init_path() + "data/energy.csv").iloc[:6, :]
    # 构造参数字典
    params = {
        "task_args": {
            "columns": ['load', 'temp'],
            "target": ['load', 'temp'],
            "features": 'M',
        },
        "predict_args": {
            "time_col": 'time',
            "freq": 'h',
            "model_path":
            init_path() + "outputs/best_models/Transformer/transformer.onnx",
            "x_true": ts_data,
            "scaler_path": init_path() + "outputs/scalers/Transformer",
            'pred_len': 1,
            "label_len": 3,
        },
    }
    y_pred = predict(**params)
    print(y_pred)
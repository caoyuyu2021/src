from init_env import init_path
import pandas as pd
import torch.nn as nn
from models.Transformer import Transformer
from utils.loader import loader
from utils.divider import divider
from utils.generator import generator
from utils.train import train
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # 数据加载
    ts_data = pd.read_csv(init_path() + "data/energy.csv")
    ts_data = loader(data_path=None, data=ts_data, time_col='time')

    # 数据集划分
    params1 = {
        "df": ts_data,
        "train_ratio": 0.8,
        "valid_ratio": 0.1,
        "x_feature_list": ['load', 'temp'],
        "y_feature_list": ['load', 'temp'],
        "freq": 'h',
        "scaler_path": init_path() + "outputs/scalers/Transformer"
    }
    x_scaler, y_scaler, train_data, valid_data, test_data = divider(**params1)

    # 利用前seq_len个数据，预测下pred_len个数据
    params2 = {
        "seq_len": 6,
        "pred_len": 1,
        "label_len": 3,
        "batch_size": 32,
    }
    X_train, y_train, X_train_stamp, y_train_stamp, train_loader = generator(
        train_data, **params2)
    X_valid, y_valid, X_valid_stamp, y_valid_stamp, valid_loader = generator(
        valid_data, **params2)
    print("X_size: {0}, y_size: {1}, X_train_stamp: {2}, loader_len: {3}".format(
    X_train.shape, y_train.shape, X_train_stamp.shape, len(train_loader)))
    print("X_size: {0}, y_size: {1}, X_valid_stamp: {2}, loader_len: {3}".format(
    X_valid.shape, y_valid.shape, X_valid_stamp.shape, len(valid_loader)))

    # 模型训练
    params3 = {
        "train_args": {
            "features": 'M',
            "model_name": Transformer,
            "train_loader": train_loader,
            "valid_loader": valid_loader,
            "n_epochs": 20,
            "learning_rate": 0.001,
            "loss": nn.MSELoss(),
            "patience": 3,
            "lradj": 'cosine',
            "model_path": init_path() + "outputs/best_models/Transformer",
            "device": 'cuda',
            "verbose": True,
            "plots": True,
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
    model = train(**params3)
    print('模型训练完成！')
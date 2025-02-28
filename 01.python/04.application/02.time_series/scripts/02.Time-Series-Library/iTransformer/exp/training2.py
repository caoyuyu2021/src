from init_env import init_path
import pandas as pd
import torch.nn as nn
from models.iTransformer import iTransformer
from utils.loader import loader
from utils.divider import divider
from utils.generator import generator
from utils.train import train
from utils.test import test
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # 数据加载
    ts_data = pd.read_csv(init_path() + "data/data620.csv", encoding='gbk')
    ts_data = loader(data_path=None, data=ts_data, time_col='时间')

    # 数据集划分
    params1 = {
        "df": ts_data,
        "train_ratio": 0.7,
        "valid_ratio": 0.15,
        "x_feature_list": ['A相有功功率', 'B相有功功率', 'C相有功功率', 'A相无功功率',
                           'B相无功功率', 'C相无功功率', '总有功功率'],
        "y_feature_list": ['A相有功功率', 'B相有功功率', 'C相有功功率', 'A相无功功率',
                           'B相无功功率', 'C相无功功率', '总有功功率'],
        "freq": 'h',
        "scaler_path": init_path() + "outputs/scalers/iTransformer"
    }
    x_scaler, y_scaler, train_data, valid_data, test_data = divider(**params1)

    # 利用前seq_len个数据，预测下pred_len个数据
    params2 = {
        "seq_len": 720,
        "pred_len": 168,
        "label_len": 72,
        "batch_size": 32,
    }
    X_train, y_train, X_train_stamp, y_train_stamp, train_loader = generator(
        train_data, **params2)
    X_valid, y_valid, X_valid_stamp, y_valid_stamp, valid_loader = generator(
        valid_data, **params2)
    X_test, y_test, X_test_stamp, y_test_stamp, test_loader = generator(
        test_data, **params2)
    print("X_size: {0}, y_size: {1}, X_train_stamp: {2}, loader_len: {3}".format(
        X_train.shape, y_train.shape, X_train_stamp.shape, len(train_loader)))
    print("X_size: {0}, y_size: {1}, X_valid_stamp: {2}, loader_len: {3}".format(
        X_valid.shape, y_valid.shape, X_valid_stamp.shape, len(valid_loader)))

    # 模型训练
    params3 = {
        "train_args": {
            "features": 'M',
            "model_name": iTransformer,
            "train_loader": train_loader,
            "valid_loader": valid_loader,
            "n_epochs": 50,
            "learning_rate": 0.001,
            "loss": nn.MSELoss(),
            "patience": 7,
            "lradj": 'cosine',
            "model_path": init_path() + "outputs/best_models/iTransformer",
            "device": 'cuda',
            "verbose": True,
            "plots": True,
        },
        "model_args": {
            'seq_len': 720,
            'pred_len': 168,
            "label_len": 72,
            'output_attention': False,
            'd_model': 1024,
            'n_heads': 8,
            'd_ff': 512,
            'dropout': 0.2,
            'e_layers': 2,
            'activation': "relu",
        },
    }
    model = train(**params3)
    print('模型训练完成！')

    # 模型测试
    params4 = {
        "test_args": {
            "features": 'M',
            "model": model,
            "x_test": X_test,
            "x_test_stamp": X_test_stamp,
            "y_test": y_test,
            "y_test_stamp": y_test_stamp,
            'drawing_pred': 6,
            'label_len': 72,
            'pred_len': 168,
            'device': 'cuda',
            'test_path': init_path() + "outputs/results/iTransformer"
        }
    }
    res = test(**params4)
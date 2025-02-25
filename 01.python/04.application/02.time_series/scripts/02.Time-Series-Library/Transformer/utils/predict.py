import torch
import pandas as pd
import numpy as np
import joblib
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from utils.loader import loader

# 时间格式编码
def time_features_from_frequency_str(freq_str: str):
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    class TimeFeature:
        def __init__(self):
            pass

        def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
            pass

        def __repr__(self):
            return self.__class__.__name__ + "()"


    class SecondOfMinute(TimeFeature):
        """Minute of hour encoded as value between [-0.5, 0.5]"""

        def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
            return index.second / 59.0 - 0.5


    class MinuteOfHour(TimeFeature):
        """Minute of hour encoded as value between [-0.5, 0.5]"""

        def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
            return index.minute / 59.0 - 0.5


    class HourOfDay(TimeFeature):
        """Hour of day encoded as value between [-0.5, 0.5]"""

        def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
            return index.hour / 23.0 - 0.5


    class DayOfWeek(TimeFeature):
        """Hour of day encoded as value between [-0.5, 0.5]"""

        def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
            return index.dayofweek / 6.0 - 0.5


    class DayOfMonth(TimeFeature):
        """Day of month encoded as value between [-0.5, 0.5]"""

        def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
            return (index.day - 1) / 30.0 - 0.5


    class DayOfYear(TimeFeature):
        """Day of year encoded as value between [-0.5, 0.5]"""

        def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
            return (index.dayofyear - 1) / 365.0 - 0.5


    class MonthOfYear(TimeFeature):
        """Month of year encoded as value between [-0.5, 0.5]"""

        def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
            return (index.month - 1) / 11.0 - 0.5


    class WeekOfYear(TimeFeature):
        """Week of year encoded as value between [-0.5, 0.5]"""

        def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
            return (index.isocalendar().week - 1) / 52.0 - 0.5

    
    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)
    
def time_features(dates, freq='h'):
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])

def predict(task_args, predict_args, model_args):
    # 参数配置
    columns = task_args['columns']
    target = task_args['target']
    features = task_args['features']
    time_col = predict_args['time_col']
    target_col = predict_args['target_col']
    freq = predict_args['freq']
    model_name = predict_args['model_name']
    x_true = predict_args['x_true']
    scaler_path = predict_args['scaler_path']
    model_path = predict_args['model_path']
    device = predict_args['device']  # 可选'cuda'和'cpu'
    pred_len = model_args['pred_len']
    label_len = model_args['label_len']
    seq_len = model_args['seq_len']
    x_true.to_excel('../data/班竹变断路器SF6气体密度输入数据表.xlsx')

    # 检查可用device
    device = torch.device(device)

    # 读取归一化参数
    x_scaler = joblib.load(scaler_path + "/x_scaler.pkl")
    y_scaler = joblib.load(scaler_path + "/y_scaler.pkl")

    # 加载模型
    model = model_name(**model_args)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    # 构造数据集
    x_true = x_true.reset_index(drop=True)
    x_true_ = loader(data_path=None, data=x_true, time_col=time_col)  # 原始数据
    time_columns = x_true_.index  # 获取时间列
    x_true_ = x_true_.copy()[columns]

    # X时间编码
    x_stamp = pd.to_datetime(x_true_.index)
    x_stamp = time_features(x_stamp, freq=freq)
    x_stamp = x_stamp.transpose(1, 0)

    # y时间编码，包含未来的时间
    timedelta = x_true_.index[-1] - x_true_.index[-2]  # 时间差
    y_stamp = pd.date_range(start=x_true_.index[0],
                            end=x_true_.index[-1] +
                            timedelta*(pred_len),
                            freq=freq)
    y_stamp = time_features(y_stamp, freq=freq)
    y_stamp = y_stamp.transpose(1, 0)

    # X归一化
    x_true_[columns] = x_scaler.transform(x_true_)
    x_true_ = x_true_.values.astype('float32')

    # 生成预测张量
    X_true, X_stamp, Y_stamp = [], [], []
    sample_freq = 1
    for index in range(0, len(x_true) - seq_len + 1, sample_freq):
        # 起点
        s_begin = index
        # 终点(起点 + 回视窗口)
        s_end = s_begin + seq_len
        # (终点 - 先验序列窗口)
        r_begin = s_end - label_len
        # (终点 + 预测序列长度)
        r_end = r_begin + label_len + pred_len

        # 数据维度
        feat = x_true_[s_begin: s_end]
        X_true.append(np.array(feat))

        # 时间维度
        xs = x_stamp[s_begin: s_end]
        ys = y_stamp[r_begin: r_end]
        X_stamp.append(np.array(xs))
        Y_stamp.append(np.array(ys))
    X_true = torch.as_tensor(X_true).float()
    X_stamp = torch.as_tensor(X_stamp).float()
    Y_stamp = torch.as_tensor(Y_stamp).float()

    # 模型预测
    model.eval()
    with torch.no_grad():
        X_true = X_true.to(device)
        X_stamp = X_stamp.to(device)
        Y_stamp = Y_stamp.to(device)

        # decoder输入
        B, _, _ = X_true.shape
        dec_inp = torch.zeros(
            (B, pred_len + label_len, len(target))).float().to(device)
        y_pred = model(X_true, X_stamp, dec_inp, Y_stamp)
        y_pred = y_pred.cpu().detach().numpy()
        f_dim = -1 if features == 'MS' else 0
        y_pred = y_pred[:, -pred_len:, f_dim:]

    # y_pred的形状为 (batch_size, pred_len, feature_dim)
    batch_size, pred_len, feature_dim = y_pred.shape
    time_index = time_columns[seq_len-1:]

    # 初始化一个空的 DataFrame，每行是初始时间，每列是递增的预测步
    # 列名格式为 target_i，例如 target_1 表示预测步1，target_2 表示预测步2，依此类推
    columns = [f"{t}_{i+1}" for i in range(pred_len) for t in target]
    result_df = pd.DataFrame(index=time_index, columns=columns)

    # 填充 DataFrame，每一行是一个时间点的预测序列
    for i in range(batch_size):
        # 当前时间点的预测
        pred_data = y_scaler.inverse_transform(
            y_pred[i, :, :])  # 形状 (pred_len, feature_dim)
        pred_flattened = pred_data.flatten()  # 将预测结果展平

        # 将展平的预测步依次填入当前行
        result_df.iloc[i] = pred_flattened

    # 将原始数据与预测数据合并输出
    result_df = result_df.reset_index().rename(columns={'index': time_col})
    select_columns = [time_col] + target
    y_pred = pd.merge(x_true[select_columns], result_df, on=time_col, how='left')

    # 获取第一个预测结果
    def last_prediction(prediction, pred_len, time_col, target):
        pre_target = [i+'_'+str(pred_len) for i in target]
        prediction_target = prediction[pre_target]
        time = prediction[[time_col]]
        # 向下平移数据
        prediction_shift = prediction_target.shift(pred_len)
        prediction_shift.columns = target
        # 对齐时间
        last_prediction = pd.concat([time, prediction_shift], axis=1)
        
        return last_prediction
    
    # 获取第一个预测结果
    params = {
        "prediction": y_pred,
        "pred_len": pred_len,
        "time_col": time_col,
        "target": target
    }
    y_pred = last_prediction(**params)

    data = pd.concat([x_true[time_col], x_true[target_col], y_pred[target_col]], axis=1)
    data.columns = [time_col, target_col[0], target_col[0]+'_prediction']
    data = data.iloc[seq_len+pred_len-1:, :].reset_index(drop=True)
    data[target_col[0]+'_residual'] = data[target_col[0]] - data[target_col[0]+'_prediction'] # 残差

    return data
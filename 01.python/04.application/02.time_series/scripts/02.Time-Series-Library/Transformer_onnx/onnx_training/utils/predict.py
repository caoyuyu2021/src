import onnx
import onnxruntime as ort
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

def predict(task_args, predict_args):
    # 参数配置
    columns = task_args['columns']
    target = task_args['target']
    features = task_args['features']
    time_col = predict_args['time_col']
    freq = predict_args['freq']
    x_true = predict_args['x_true']
    scaler_path = predict_args['scaler_path']
    model_path = predict_args['model_path']
    pred_len = predict_args['pred_len']
    label_len = predict_args['label_len']

    # 读取归一化参数
    x_scaler = joblib.load(scaler_path + "/x_scaler.pkl")
    y_scaler = joblib.load(scaler_path + "/y_scaler.pkl")

    # 加载模型
    onnx_model = onnx.load(model_path)

    # 将模型序列化为字符串
    onnx_model_serialized = onnx_model.SerializeToString()

    # 使用指定的提供者创建推理会话
    sess_options = ort.SessionOptions()
    providers = ['CPUExecutionProvider']
    sess = ort.InferenceSession(onnx_model_serialized,
                                sess_options,
                                providers=providers)

    # 未设置时间编码
    if freq == None:
        # 参数名称
        input_name1 = sess.get_inputs()[0].name
        input_name2 = sess.get_inputs()[1].name
        output_name = sess.get_outputs()[0].name
    
        # 生成固定长度的时间范围
        x_true = loader(data_path=None, data=x_true, time_col=time_col)  # 原始数据
        x_true = x_true[columns]
    
        # 转换类型
        x_true[columns] = x_scaler.transform(x_true)  # 归一化
        x_true = x_true.values.astype('float32')
        x_true = np.array(x_true, dtype=np.float32)[np.newaxis, :]
    
        # 模型推理
        B, _, _ = x_true.shape
        dec_inp = np.zeros((B, pred_len + label_len, len(target))).astype(np.float32)# 占位符
        y_pred = sess.run(
            [output_name], {
                input_name1: x_true,
                input_name2: dec_inp,
            })[0]
        f_dim = -1 if features == 'MS' else 0
        y_pred = y_pred[:, -pred_len:, f_dim:]
        y_pred = y_scaler.inverse_transform(y_pred[-1, :, :])  # 反归一化
    # 设置了时间编码
    else:
        # 参数名称
        input_name1 = sess.get_inputs()[0].name
        input_name2 = sess.get_inputs()[1].name
        input_name3 = sess.get_inputs()[2].name
        input_name4 = sess.get_inputs()[3].name
        output_name = sess.get_outputs()[0].name
    
        # 生成固定长度的时间范围
        x_true = loader(data_path=None, data=x_true, time_col=time_col)  # 原始数据
        x_true = x_true[columns]
        timedelta = x_true.index[-1] - x_true.index[-2]  # 时间差
        if label_len != 0:
            y_stamp = pd.date_range(start=x_true.index[-label_len],
                                    end=x_true.index[-label_len] +
                                    timedelta*(label_len+pred_len-1),
                                    freq=freq)
        else:
            y_stamp = pd.date_range(start=x_true.index[-1]+timedelta*(label_len+1),
                                    end=x_true.index[-1] +
                                    timedelta*(label_len+pred_len),
                                    freq=freq)
        x_stamp = time_features(pd.to_datetime(x_true.index), freq=freq)  # x时间戳数据
        x_stamp = x_stamp.transpose(1, 0)
        y_time = y_stamp
        y_stamp = time_features(y_stamp, freq=freq)  # y时间戳数据
        y_stamp = y_stamp.transpose(1, 0)
    
        # 转换类型
        x_true[columns] = x_scaler.transform(x_true)  # 归一化
        x_true = x_true.values.astype('float32')
        x_true = np.array(x_true, dtype=np.float32)[np.newaxis, :]
        x_stamp = np.array(x_stamp, dtype=np.float32)[np.newaxis, :]
        y_stamp = np.array(y_stamp, dtype=np.float32)[np.newaxis, :]
    
        # 模型推理
        B, _, _ = x_true.shape
        dec_inp = np.zeros((B, pred_len + label_len, len(target))).astype(np.float32)# 占位符
        y_pred = sess.run(
            [output_name], {
                input_name1: x_true,
                input_name2: x_stamp,
                input_name3: dec_inp,
                input_name4: y_stamp,
            })[0]
        f_dim = -1 if features == 'MS' else 0
        y_pred = y_pred[:, -pred_len:, f_dim:]
        y_pred = y_scaler.inverse_transform(y_pred[-1, :, :])  # 反归一化
    
        # 输出为dataframe
        y_pred = pd.DataFrame(data=y_pred, index=y_time[-pred_len:], columns=target)

    return y_pred
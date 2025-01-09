from models.Transformer import Transformer
from utils.transform import transform
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # 构造参数字典
    params = {
        "transform_args": {
            "batch_size": 1,
            "seq_len": 6,
            "mask_size": 4,
            "model_name": Transformer,
            "model_path": "../outputs/best_models/Transformer/checkpoint.pth",
            "export_onnx_path": '../outputs/best_models/Transformer/transformer.onnx',
        },
        "model_args": {
            'seq_len': 6,
            'pred_len': 1, 
            'label_len': 3,
            'output_attention': True,
            'embed': 'timeF', 
            'freq': 'h',
            'd_model': 256,
            'enc_in': 2,
            'dec_in': 1,
            'dropout': 0.1,
            'factor': 3,
            'n_heads': 8,
            'd_ff': 128,
            'e_layers': 1,
            'd_layers': 1,
            'c_out': 1
        },
    }
    transform(**params)
    print('模型保存完成！')
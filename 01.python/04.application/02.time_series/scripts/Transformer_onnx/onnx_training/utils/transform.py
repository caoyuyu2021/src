import torch
import warnings

warnings.filterwarnings("ignore")

def transform(transform_args, model_args):
    # 参数配置
    model_name = transform_args['model_name'] # torch模型名称
    model_path = transform_args['model_path'] # torch模型保存路径
    export_onnx_path = transform_args['export_onnx_path'] # ONNX模型保存路径
    B = transform_args['batch_size']
    L = transform_args['seq_len']
    D_x = model_args['enc_in'] # 输入维度
    D_y = model_args['c_out'] # 输出维度
    D_mask = transform_args['mask_size'] # 时间特征维度
    pred_len = model_args['pred_len']
    label_len = model_args['label_len']

    # 加载torch模型
    torch_model = model_name(**model_args)
    state_dict = torch.load(model_path)
    torch_model.load_state_dict(state_dict)

    # 生成随机张量
    x_true = torch.rand(B, L, D_x)
    dec_inp = torch.rand(B, pred_len+label_len, D_y)
    if D_mask == None:
        x_stamp = None
        y_stamp = None
    else:
        x_stamp = torch.rand(B, L, D_mask)
        y_stamp = torch.rand(B, pred_len+label_len, D_mask)

    # 切换为推理模式
    torch_model.eval()

    # 保存模型为onnx
    torch.onnx.export(
        model=torch_model,
        args=(x_true, x_stamp, dec_inp, y_stamp),
        f=export_onnx_path,
        verbose=False,
        opset_version=17,
        # 是否执行常量折叠优化
        do_constant_folding=True,
        # 输入名
        input_names=['x_true', 'x_stamp', 'dec_inp', 'y_stamp'],
        # 输出名
        output_names=['output'],
        # 指定模型的动态轴
        dynamic_axes={
            'x_true': {
                0: 'batch_size'
            },
            'x_stamp': {
                0: 'batch_size'
            },
            'dec_inp': {
                0: 'batch_size'
            },
            'y_stamp': {
                0: 'batch_size'
            },
            'output': {
                0: 'batch_size'
            }
        })
    return 0
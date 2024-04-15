import torch
import math
import torch.nn as nn

# 正弦位置编码
class SinusoidalPositionalEncoder(nn.Module):
    def __init__(self, seq_len, d_model, dropout=0.1):
        super(SinusoidalPositionalEncoder, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(1, seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        # 通过register_buffer注册为模型的缓冲区，因此在模型保存和加载时，这个位置编码会被保留
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe
        x = self.dropout(x)

        return x

# 可学习位置编码，参数的值会在训练过程中通过反向传播进行学习
class LearnablePositionalEncoder(nn.Module):
    def __init__(self, seq_len, d_model, dropout=0.1):
        super(LearnablePositionalEncoder, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        pe = nn.parameter.Parameter(torch.zeros((1, seq_len, d_model)))
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe
        x = self.dropout(x)

        return x

# Transformer模型
class Transformer(nn.Module):
    def __init__(self, sequence_len, target_len, num_encoder_layers, num_decoder_layers, input_dim, out_dim, 
                 d_model, num_heads, feedforward_dim, dropout=0.1, positional_encoding="sinusoidal"):
        super(Transformer, self).__init__()
        # sequence_len输入时间步，target_len输出时间步
        self.sequence_len = sequence_len
        self.target_len = target_len

        #位置编码
        if positional_encoding == "sinusoidal":
            self.positional_encoder = SinusoidalPositionalEncoder(seq_len=self.sequence_len, 
                                                                  d_model=d_model, 
                                                                  dropout=dropout)
            self.positional_decoder = SinusoidalPositionalEncoder(seq_len=self.target_len, 
                                                                  d_model=d_model, 
                                                                  dropout=dropout)
        elif positional_encoding == "learnable":
            self.positional_encoder = LearnablePositionalEncoder(seq_len=self.sequence_len, 
                                                                 d_model=d_model, 
                                                                 dropout=dropout)
            self.positional_decoder = LearnablePositionalEncoder(seq_len=self.target_len, 
                                                                 d_model=d_model, 
                                                                 dropout=dropout)
        else:
            raise Exception("Positional encoding type not recognized: use 'sinusoidal' or 'learnable'.")

        # 将输入序列进行线性变换，以便传递给Transformer模型的编码器
        self.encoder_input_layer = nn.Linear(input_dim, d_model)

        # 通过输入序列在多头自注意力层和前馈神经网络层之间进行处理，输出编码后的序列，可以被堆叠多次
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, # 模型的隐藏层大小
                                                   nhead=num_heads, # 多头自注意力机制中的头数
                                                   dim_feedforward=feedforward_dim, # 全连接前馈神经网络中间层的维度
                                                   dropout=dropout, # 用于在多头自注意力机制和全连接前馈神经网络中进行dropout的概率
                                                   batch_first=False) # batch_first: 若`True`，则为(batch, seq, feture)，若为`False`，则为(seq, batch, feature)

        # 用于创建多层Transformer编码器的类
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, # 用于构建编码器的层对象
                                             num_layers=num_encoder_layers) # 编码器中的层数

        # 将输出序列进行线性变换，以便传递给Transformer模型的解码器
        self.decoder_input_layer = nn.Linear(out_dim, d_model)

        # 通过输出序列在多头自注意力层和前馈神经网络层之间进行处理，输出编码后的序列，可以被堆叠多次
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, 
                                                   nhead=num_heads, 
                                                   dim_feedforward=feedforward_dim, 
                                                   dropout=dropout, 
                                                   batch_first=False)

        # 用于创建多层Transformer编码器的类
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, 
                                             num_layers=num_decoder_layers)

        # 用于将Transformer解码器的输出映射到最终的目标维度
        self.output_layer = nn.Linear(d_model, out_dim)

        # 初始化权重
        self.init_weights()

    # 用来初始化模型的权重的，它初始化了编码器输入层、解码器输入层和输出层的权重
    def init_weights(self):
        initrange = 0.1
        # 将张量的值从均匀分布[-initrange, initrange]中抽取
        self.encoder_input_layer.weight.data.uniform_(-initrange, initrange)
        self.decoder_input_layer.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_() # 偏置置零
        self.output_layer.weight.data.uniform_(-initrange, initrange)

    # generate_mask函数用于生成一个上三角矩阵，用于在自注意力机制中屏蔽未来时间步的信息。
    def generate_mask(self, dim1, dim2):
        # 其中dim1和dim2分别是上三角矩阵的行数和列数
        return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)

    def forward(self, src, trg, memory_mask=None, trg_mask=None):
        # 输入数据的维度为(batch_size, sequence_length, input_dim)
        src = self.encoder_input_layer(src)
        src = self.positional_encoder(src)
        # transformer维度要求(sequence_length, batch_size, input_dim)
        src = src.permute(1, 0, 2)
        encoder_output = self.encoder(src)

        trg = self.decoder_input_layer(trg)
        trg = self.positional_decoder(trg)
        trg = trg.permute(1, 0, 2)

        if memory_mask is None:
            memory_mask = self.generate_mask(self.target_len, self.sequence_len).to(src.device)
        if trg_mask is None:
            # trg_mask的目的是在解码器的自注意力机制中防止模型访问未来的信息。
            trg_mask = self.generate_mask(self.target_len, self.target_len).to(trg.device)

        decoder_output = self.decoder(tgt=trg, # 目标序列张量，形状为(target_seq_len, batch_size, model_dim)
                                      memory=encoder_output, # 编码器的输出张量，形状为(src_seq_len, batch_size, model_dim)
                                      tgt_mask=trg_mask, # 目标序列的掩码，形状为(target_seq_len, target_seq_len)或者None
                                      memory_mask=memory_mask) # 编码器输出的掩码，形状为(target_seq_len, src_seq_len)或者 None

        output = self.output_layer(decoder_output)
        # 将维度还原为(batch_size, sequence_length, input_dim)
        output = output.permute(1, 0, 2)

        return output
# lstm_model.py
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=5, num_layers=1, batch_first=True)
        self.linear = nn.Linear(in_features=5, out_features=out_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step_output = lstm_out[:, -1, :]
        output = self.linear(last_time_step_output)
        return output
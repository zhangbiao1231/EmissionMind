import math
from pyexpat import features

import torch
from torch import nn
from torch.nn import functional as F
#============================================================================================#
# 为一个完整的循环神经网络模型定义一个RNNModel类。
# ⚠️ rnn_layer只包含隐藏的循环层，我们还需要创建一个单独的输出层。
class RNNModel(nn.Module):
    """循环神经网络模型"""
    def __init__(self, rnn_layer=None, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.num_hiddens = self.rnn.hidden_size
        self.output_features = 1
        self.fc = nn.Linear(self.num_hiddens, self.num_hiddens // 2)
        # self.fc = nn.Sequential(
        #     nn.Linear(self.num_hiddens, self.num_hiddens // 2),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(self.num_hiddens // 2, self.num_hiddens // 4),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(self.num_hiddens // 4, self.num_hiddens // 8),
        #     nn.ReLU(),)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.output_layer = nn.Linear(self.num_hiddens // 2, self.output_features)

    def forward(self, X, state=None):
        # X:(batch_size, time_steps, input_size)
        X = X.permute(1, 0, 2)
        # X:(time_steps, batch_size, input_size)
        Y, state = self.rnn(X, state)
        # Y: (time_steps， batch_size, hidden_size)
        middel_layer = self.fc(Y)
        output = self.output_layer(middel_layer)
        # output = self.output_layer(Y)
        out = output.permute(1, 0, 2).squeeze(-1)  # → [B, T]
        return out, state

    def begin_state(self, batch_size, device):
        if not isinstance(self.rnn, nn.LSTM):
            # state 对于nn.GRU或nn.RNN作为张量
            # 非 nn.LSTM 模型（如 nn.RNN 或 nn.GRU）,函数返回一个三维张量。
            return torch.zeros((self.rnn.num_layers,
                                batch_size, self.num_hiddens),
                               device=device)
        else:
            # state 对于nn.LSTM 是个元组
            # • 如果模型是 nn.LSTM，则返回一个包含两个张量的元组，分别表示 LSTM 的隐藏状态 (hidden state) 和单元状态 (cell state)。
            # • 每个张量的形状相同，均为 (num_directions * num_layers, batch_size, hidden_size)，其中：
            ##  • hidden state：储存每一层每个时间步的隐藏状态。
            ##  • cell state：储存每一层每个时间步的记忆状态，用于控制长期依赖信息。
            return (torch.zeros((self.rnn.num_layers, batch_size, self.num_hiddens),
                                device=device),
                    torch.zeros((self.rnn.num_layers, batch_size, self.num_hiddens),
                                device=device))
class get_rnn_layer(nn.Module):
    def __init__(self, input_size, num_hiddens, num_layers, dropout,**kwargs):
        super(get_rnn_layer, self).__init__(**kwargs)
        self.input_size = input_size
        self.num_hiddens=num_hiddens
        self.num_layers=num_layers
        self.dropout = dropout
    def construct_rnn(self,selected_model):
        if selected_model.lower() == "rnn":
            return nn.RNN(input_size=self.input_size,
                          hidden_size=self.num_hiddens,
                          num_layers=self.num_layers,
                          dropout=self.dropout)
        elif selected_model.lower() == "gru":
            return nn.GRU(input_size=self.input_size,
                          hidden_size=self.num_hiddens,
                          num_layers=self.num_layers,
                          dropout=self.dropout)
        else:
            return nn.LSTM(input_size=self.input_size,
                          hidden_size=self.num_hiddens,
                          num_layers=self.num_layers,
                          dropout=self.dropout)
if __name__ == '__main__':
    #============================================================================================#
    # 简洁实现: 直接实例化循环神经网络模型（RNN）
    batch_size, time_steps, input_size = 32, 10, 6
    X = torch.randn(batch_size, time_steps, input_size) # X : (32, 10, 6)

    # create model
    get_rnn_layer = get_rnn_layer(input_size=input_size,
                                      num_hiddens=32,
                                      num_layers=2,
                                      dropout=0.5)
    rnn_layer = get_rnn_layer.construct_rnn(selected_model="rnn")

    net = RNNModel(rnn_layer=rnn_layer)  # out_feature = 1
    print(f'输入形状:{X.shape}')
    state = net.begin_state(device="cpu",batch_size=X.shape[0])
    Y, new_state = net(X, state)

    print(f'输出形状:{Y.shape}\n更新隐状态H长度:{len(new_state)}\n更新隐状态H形状:{new_state[0].shape}')
    # 输出形状：（时间步数, 批量大小, 1）
    # 隐变量形状：（批量大小, 隐藏单元数）


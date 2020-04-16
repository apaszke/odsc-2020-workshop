import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.cpp_extension as cpp

cpp_lstms = cpp.load(name='lstms',
                     sources=['cpp/lstm.cpp'],
                     extra_cflags=['-O3'],
                     verbose=True)


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.empty(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        nn.init.uniform_(self.weight_ih, -stdv, stdv)
        nn.init.uniform_(self.weight_hh, -stdv, stdv)
        nn.init.constant_(self.bias, 0)

    def forward(self, input, state):
        return cpp_lstms.lstm_cell(input, state, self.weight_ih, self.weight_hh, self.bias)


class LSTMLayer(nn.Module):
    def __init__(self, *cell_args):
        super().__init__()
        self.cell = LSTMCell(*cell_args)

    def forward(self, input, state):
        return cpp_lstms.lstm(input, state, self.cell.weight_ih, self.cell.weight_hh, self.cell.bias)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        assert num_layers >= 1
        self.layers = nn.ModuleList([LSTMLayer(input_size, hidden_size)] +
                                    [LSTMLayer(hidden_size, hidden_size)
                                     for _ in range(num_layers - 1)])
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward(self, input, states):
        output_states = []
        output = input
        for i, layer in enumerate(self.layers):
            output, out_state = layer(output, (states[0][i], states[1][i]))
            # Apply the dropout layer except the last layer
            if i < self.num_layers - 1:
                output = self.dropout(output)
            output_states.append(out_state)
        return output, (torch.stack([s[0] for s in output_states]),
                        torch.stack([s[1] for s in output_states]))

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        hx, cx = state
        gates = (torch.mm(input, self.weight_ih.t()) +
                 torch.mm(hx, self.weight_hh.t()) + self.bias)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy


class LSTMLayer(nn.Module):
    def __init__(self, *cell_args):
        super().__init__()
        self.cell = LSTMCell(*cell_args)

    def forward(self, input, state):
        outputs = []
        for i in input.unbind(0):
            state = self.cell(i, state)
            outputs.append(state[0])
        return torch.stack(outputs), state


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

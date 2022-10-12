# -*- coding: utf-8 -*-

import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class biLSTM(nn.Module):

    def __init__(self, hisize, layers, insize, dropout):
        super(biLSTM, self).__init__()
        self.rnn = nn.LSTM(
            hidden_size=hisize,
            num_layers=layers,
            input_size=insize,
            bidirectional=True,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x, len):
        x = pack_padded_sequence(x, len, batch_first=True, enforce_sorted=False)
        y, _ = self.rnn(x)
        y = pad_packed_sequence(y, batch_first=True)
        return y[0]

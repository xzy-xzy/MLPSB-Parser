import torch as tch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .bilstm import biLSTM

class CharLSTM(nn.Module):

    def __init__(self, in_dim, out_dim, char_emb):
        super(CharLSTM, self).__init__( )
        self.lstm = biLSTM(
            hisize = out_dim // 2,
            layers = 1,
            insize = in_dim,
            dropout = 0
        )
        self.emb = char_emb

    def Emb(self, input, lens):
        batch_size = len(lens)
        word_len = [ ]
        word_list = [ ]     # word_list is a List of (word_len, dim)
        for i in range(batch_size):
            L = [self.emb.CharField(x, False) for x in input[i]]
            word_list += L
            word_len += [x.size(0) for x in L]
        max_word_len = np.max(word_len)
        word_cnt = len(word_len)
        for i in range(word_cnt):
            word_list[i] = F.pad(word_list[i], (0, 0, 0, max_word_len - word_len[i]))
        input = tch.stack(word_list, dim=0)     # input is (word_cnt, word_len, dim)
        res = self.lstm(input, word_len)
        B, N = res.size(0), res.size(1)
        res = res.view(B, N, 2, -1)
        emb = [ ]
        for i in range(word_cnt):
            emb += [tch.cat((res[i, word_len[i] - 1, 0], res[i, 0, 1]), dim=0)]    # emb is a List of (dim)
        max_len = np.max(lens)
        t_list = [ ]
        st = 0
        for i in range(batch_size):
            piece = emb[st:st + lens[i]]
            st += lens[i]
            piece = tch.stack(piece, dim=0)     # (sent_len, dim)
            piece = F.pad(piece, (0, 0, 0, max_len - lens[i]))
            t_list += [piece]
        T = tch.stack(t_list, dim=0)            # (batch_size, sent_len, dim)
        return T



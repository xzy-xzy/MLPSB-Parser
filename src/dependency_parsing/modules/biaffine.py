import torch as tch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from .arc import Arc
from .label import Label
from .mlp import MLP


def Clone(model, n):
    return nn.ModuleList([copy.deepcopy(model) for _ in range(n)])


class BiaffineLayer(nn.Module):
    def __init__(self, din, dim_arc, dropout, siamese, attn_type):
        super(BiaffineLayer, self).__init__()
        self.arc = Arc(din=din, dim=dim_arc, drp=dropout, type=attn_type)
        # self.label = Label(din=din, dim=dim_label, num_out=num_rel, drp=dropout)
        self.ex_arc = Arc(din=din, dim=dim_arc, drp=dropout)
        # self.ex_label = Label(din=din, dim=dim_label, num_out=num_rel, drp=dropout)
        self.vec_arc = MLP(in_ch=din, out_ch=dim_arc, drp=dropout)
        # self.vec_label = MLP(in_ch=din, out_ch=dim_label, drp=dropout)
        # self.V_label = nn.Linear(num_rel, 1, bias=False)
        self.proj = nn.Linear(dim_arc, din, False)
        self.norm = nn.LayerNorm(din)
        self.INF = float('inf')
        self.FAC = din ** 0.5
        self.siamese = siamese

    def forward(self, X, mask):
        '''
        X: (batch_size, len, din)
        mask: (batch_size, len, len)
        '''
        if self.siamese == "P":
            E_arc = self.ex_arc.Score(X)  # (B, len, len)
            E_arc = E_arc / self.FAC  # FIX
            E_arc.masked_fill_(mask, -self.INF)
            E_arc = F.softmax(E_arc, dim=2)

        S_arc = self.arc.Score(X)  # (B, len, len)
        S_arc = S_arc / self.FAC  # FIX
        S_arc.masked_fill_(mask, -self.INF)
        S_arc = F.softmax(S_arc, dim=2)

        V_arc = self.vec_arc(X)  # (B, len, D)
        Y_arc = tch.matmul(S_arc, V_arc)  # (B, len, D)
        Y_arc = self.proj(Y_arc)
        Y = self.norm(X + Y_arc)

        if self.siamese == "P":
            return E_arc, Y
        else:
            return S_arc, Y


class MultiLayerBiaffine(nn.Module):
    def __init__(self, layer_num, din, dim_arc, dim_label, num_rel, dropout, siamese, label_drop, attn_type):
        super(MultiLayerBiaffine, self).__init__()
        self.n = layer_num
        self.layers = Clone(BiaffineLayer(din, dim_arc, dropout, siamese, attn_type), layer_num)
        self.arc = Arc(din=din, dim=dim_arc, drp=dropout)
        self.label = Label(din=din, dim=dim_label, num_out=num_rel, drp=dropout, label_drop=label_drop)
        self.INF = float('inf')
        self.FAC = din ** 0.5
        self.siamese = siamese

    def get_mask(self, X, lens):
        B, L = X.size(0), X.size(1)
        mask = X.new_zeros((B, L), dtype=tch.bool)
        for i in range(B): mask[i, 0:lens[i]].fill_(True)  # (batch_size, len) False-mask
        mask = (~mask).unsqueeze(1).expand(B, L, L)  # (batch_size, len, len) True-mask
        return mask

    def get_label_score(self, X, fa):
        return self.label.Score(X, fa)

    def forward(self, X, lens):
        '''
        X: (batch_size, len)
        lens: (batch_size)
        '''
        mask = self.get_mask(X, lens)
        W_arc = [ ]
        for i in range(self.n):
            L_arc, X = self.layers[i](X, mask)
            W_arc.append(L_arc)
        F_arc = self.arc.Score(X)
        F_arc = F_arc / self.FAC
        F_arc.masked_fill_(mask, -self.INF)
        F_arc = F.softmax(F_arc, dim=2)
        W_arc.append(F_arc)

        if self.siamese == "N":
            return F_arc, X
        else:
            return W_arc, X
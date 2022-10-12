# -*- coding: utf-8 -*-

import torch as tch
import torch.nn as nn
import torch.nn.functional as F

class Weight_Average(nn.Module):

    def __init__(self, n):
        super(Weight_Average, self).__init__()
        self.n = n
        self.W = nn.Parameter(tch.zeros(n))
        self.rate = nn.Parameter(tch.tensor([1.0]))

    def forward(self, input):
        W = F.softmax(self.W, dim=0)
        S = self.rate * sum(w * i for w, i in zip(W, input))
        return S

class Weight_Average_Norate(nn.Module):

    def __init__(self, n, fixed):
        super(Weight_Average_Norate, self).__init__()
        self.n = n
        self.W = nn.Parameter(tch.zeros(n), requires_grad=not fixed)

    def forward(self, input):
        W = F.softmax(self.W, dim=0)
        S = sum(w * i for w, i in zip(W, input))
        return S

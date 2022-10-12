import torch as tch
import torch.nn as nn
from .mlp import MLP

class Arc(nn.Module):

    def __init__(self, din, dim, drp, type="biaffine"):
        super(Arc, self).__init__( )
        self.dim = dim
        self.MLP_head = MLP(din, dim, drp)
        self.MLP_depe = MLP(din, dim, drp)
        if type == "biaffine":
            self.U1_arc = nn.Linear(dim, dim, bias=False)
            self.U2_arc = nn.Linear(dim, 1, bias=False)
        if type == "general":
            self.U1_arc = nn.Linear(dim, dim, bias=False)
        if type == "concat":
            self.U1_arc = nn.Linear(dim * 2, dim, bias=False)
            self.U2_arc = nn.Linear(dim, 1, bias=False)
            self.Tanh = nn.Tanh( )
        self.type = type
        # self.U3_arc = nn.Linear(dim, 1, bias=False)

    def get_W(self, f_h, f_d):
        '''
        f_h: (batch, len, dim)
        f_d: (batch, len, dim)
        '''
        if self.type == "biaffine":
            tmp = self.U1_arc(f_d)      # (B, N, D)
            w_arc = tch.matmul(tmp, f_h.transpose(1, 2))  # (B, N, N) [dim1 is dep, dim2 is head]
            b_h = self.U2_arc(f_h).transpose(1, 2)  # (B, 1, N)
            # b_d = self.U3_arc(f_d)      # (B, N, 1)
            W = w_arc + b_h     # + b_d       # (B, N, N)

        elif self.type == "general":
            tmp = self.U1_arc(f_d)      # (B, N, D)
            W = tch.matmul(tmp, f_h.transpose(1, 2))  # (B, N, N) [dim1 is dep, dim2 is head]

        elif self.type == "dot":
            W = tch.matmul(f_d, f_h.transpose(1, 2))

        elif self.type == "concat":
            N = f_d.size(1)
            A = f_h.unsqueeze(1).repeat(1, N, 1, 1)     # (B, N, N, D) [1,2,3,1,2,3,1,2,3]
            B = f_d.unsqueeze(2).repeat(1, 1, N, 1)     # (B, N, N, D) [1,1,1,2,2,2,3,3,3]
            W = self.U2_arc(self.Tanh(self.U1_arc(tch.cat([A, B], dim=3)))).squeeze(3)

        return W

    def get_F(self, fea):
        '''
        fea: (batch, len, din)
        '''
        f_h = self.MLP_head(fea)    #(batch, len, dim)
        f_d = self.MLP_depe(fea)    #(batch, len, dim)
        return f_h, f_d

    def Score(self, X):
        a, b = self.get_F(X)
        return self.get_W(a, b)

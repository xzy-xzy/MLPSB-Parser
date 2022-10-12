import torch as tch
import torch.nn as nn
from .mlp import MLP

class Label(nn.Module):

    def __init__(self, din, dim, num_out, drp, label_drop):
        super(Label, self).__init__( )
        self.MLP_labelh = MLP(din, dim, drp)
        self.MLP_labeld = MLP(din, dim, drp)
        self.K = num_out
        self.U1_label = nn.Linear(dim, num_out * dim, bias=False)
        self.U2_label = nn.Linear(2 * dim, num_out, bias=True)
        self.FAC = dim ** 0.5

    def get_F(self, fea):
        '''
        fea: (batch, din)
        '''
        f_h = self.MLP_labelh(fea)    #(batch, len, dim)
        f_d = self.MLP_labeld(fea)    #(batch, len, dim)
        return f_h, f_d

    def get_W(self, x, y):
        '''
        x,y: (batch, len, dim)
        '''
        tmp = tch.einsum('abc, efc -> efab', self.U1_label, y)  # (batch, len, outnum, dim)
        score = tch.einsum('abcd, aed -> abec', tmp, x)  # (batch, len, len, outnum)
        return score

    def Score(self, X, fa):
        '''
        X: (B, N, D)
        fa: (B, N)
        '''
        h, d = self.get_F(X)    # (B, N, D)
        B, N, D = h.size( )
        heads = [tch.index_select(x, 0, y) for x, y in zip(h, fa)]
        h = tch.stack(heads, dim=0)
        tmp = self.U1_label(h)  # (B, N, K * D)
        tmp = tmp.view(B, N, self.K, D)     # (B, N, K, D) (B, N, D)
        score1 = tch.einsum('abcd, abd -> abc', tmp, d)     # (B, N, K)
        score2 = self.U2_label(tch.cat([h, d], dim=2))      # (B, N, K)
        score = (score1 + score2)
        score = score.softmax(dim=2)
        return score


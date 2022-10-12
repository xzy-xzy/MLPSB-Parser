import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, in_ch, out_ch, drp, use_drop=False):
        super(MLP, self).__init__()
        self.Linear = nn.Linear(in_ch, out_ch)
        self.ReLU = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(drp)
        # print(use_drop)
        if use_drop:
            self.seq = nn.Sequential(self.Linear, self.ReLU, self.dropout)
        else:
            self.seq = nn.Sequential(self.Linear, self.ReLU)

    def forward(self, x):
        return self.seq(x)

import torch.nn as nn
from .weight_average import Weight_Average

class BertTool(nn.Module):

    def __init__(self, layer, in_ch, out_ch):
        super(BertTool, self).__init__()
        self.ave = Weight_Average(layer)
        self.linear = nn.Linear(in_ch, out_ch, bias=False)
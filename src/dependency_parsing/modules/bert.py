import torch as tch
import torch.nn as nn
from transformers import XLNetModel, XLNetTokenizer
from transformers import BertModel, BertTokenizer

class XLNetEmbedding(nn.Module):

    def __init__(self, path, layer, tool, dv, grad=False, append=None):
        super(XLNetEmbedding, self).__init__()
        self.bert = XLNetModel.from_pretrained(path, output_hidden_states=True).requires_grad_(grad).to(dv)
        self.device = dv
        self.tz = XLNetTokenizer.from_pretrained(path)
        if append:
            print(append)
            self.tz.add_special_tokens({'additional_special_tokens':append})
            self.bert.resize_token_embeddings(len(self.tz))
        self.spec = [4, 3]
        self.layer = layer
        self.grad = grad
        self.pad = 5
        self.hisize = self.bert.config.hidden_size
        self.average = tool.ave
        self.linear = tool.linear

    def convert(self, input):
        id, lens = [ ], [ ]
        for x in input:
            idseg, lseg = [ ], [ ]
            for piece in x:
                rep = self.tz.__call__(piece)['input_ids'][:-2]
                idseg += rep
                lseg.append(len(rep))
            id.append(idseg + self.spec)
            lens.append(lseg)

        max_len = max([len(x) for x in id])
        token = [[self.pad] * (max_len - len(x)) + x for x in id]
        mask = [[0] * (max_len - len(x)) + [1] * len(x) for x in id]
        token = tch.tensor(token).to(self.device)
        mask = tch.tensor(mask).to(self.device)

        seq_len = max([len(x) for x in lens])
        batch_size = len(input)
        lens = [x + [0] * (seq_len - len(x)) for x in lens]
        lens = tch.tensor(lens).to(self.device)

        return token, mask, batch_size, lens, seq_len

    def forward(self, input):
        #self.bert.eval( )
        token, bert_mask, batch_size, lens, seq_len = self.convert(input)
        len_mask = lens.gt(0)   #(batch, seq_len)
        output = self.bert(token, attention_mask=bert_mask)
        res = output.hidden_states
        res = res[-self.layer:]     #(layer, batch, max_len, dim)
        res = self.average(res)     #(batch, max_len, dim)
        B = bert_mask.size(0)
        for i in range(B):
            bert_mask[i][-1] = 0
            bert_mask[i][-2] = 0
        res = res[bert_mask.gt(0)].split(lens[len_mask].tolist( ))  #(word_num, each_word_len, dim)
        res = tch.stack([x.mean(0) for x in res])   #(word_num, dim)
        emb = res.new_zeros(batch_size, seq_len, self.hisize)
        emb = emb.masked_scatter_(len_mask.unsqueeze(-1), res)  #(batch, seq_len, dim)
        emb = self.linear(emb)
        return emb

class BertEmbedding(nn.Module):

    def __init__(self, path, layer, tool, dv, grad=False, append=None):
        super(BertEmbedding, self).__init__()
        self.bert = BertModel.from_pretrained(path, output_hidden_states=True).requires_grad_(grad).to(dv)
        self.device = dv
        self.tz = BertTokenizer.from_pretrained(path)
        if append:
            print(append)
            self.tz.add_special_tokens({'additional_special_tokens':append})
            self.bert.resize_token_embeddings(len(self.tz))
        self.cls = 101
        self.sep = 102
        self.max_len = 510
        self.layer = layer
        self.grad = grad
        self.pad = 0
        self.hisize = self.bert.config.hidden_size
        self.average = tool.ave
        self.linear = tool.linear

    def convert(self, input):
        id, lens = [ ], [ ]
        for x in input:
            L = 0
            idseg, lseg = [ ], [ ]
            for piece in x:
                rep = self.tz.__call__(piece)['input_ids'][1:-1]
                L += len(rep)
                if L > self.max_len:
                    break
                idseg += rep
                lseg.append(len(rep))
            id.append([self.cls] + idseg + [self.sep])
            lens.append(lseg)

        id_lens = [len(x) for x in id]
        max_len = max(id_lens)
        # print(max_len)
        token = [x + [self.pad] * (max_len - len(x)) for x in id]
        mask = [[1] * len(x) + [0] * (max_len - len(x)) for x in id]
        token = tch.tensor(token).to(self.device)
        mask = tch.tensor(mask).to(self.device)

        seq_len = max([len(x) for x in lens])
        batch_size = len(input)
        lens = [x + [0] * (seq_len - len(x)) for x in lens]
        lens = tch.tensor(lens).to(self.device)

        return token, mask, batch_size, lens, seq_len, id_lens

    def forward(self, input):
        #self.bert.eval( )
        token, bert_mask, batch_size, lens, seq_len, id_lens = self.convert(input)
        len_mask = lens.gt(0)   #(batch, seq_len)
        output = self.bert(token, attention_mask=bert_mask)
        res = output.hidden_states
        res = res[-self.layer:]     #(layer, batch, max_len, dim)
        res = self.average(res)     #(batch, max_len, dim)
        B = bert_mask.size(0)
        for i in range(B):
            bert_mask[i][0] = 0
            bert_mask[i][id_lens[i] - 1] = 0
        res = res[bert_mask.gt(0)].split(lens[len_mask].tolist( ))  #(word_num, each_word_len, dim)
        res = tch.stack([x.mean(0) for x in res])   #(word_num, dim)
        emb = res.new_zeros(batch_size, seq_len, self.hisize)
        emb = emb.masked_scatter_(len_mask.unsqueeze(-1), res)  #(batch, seq_len, dim)
        emb = self.linear(emb)
        return emb

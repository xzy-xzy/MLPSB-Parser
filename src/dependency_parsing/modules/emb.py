import torch as tch
import torch.nn as nn
import random
import math

def rand_vec(D):
    x = [ ]
    for i in range(D): x.append(-1 + 2 * random.random( ))
    norm = 0
    for i in range(D): norm += x[i] * x[i]
    norm = math.sqrt(norm)
    for i in range(D): x[i] = x[i] / norm
    return x


class Emb(nn.Module):
    def __init__(self, need_vec, freeze=None, device=None, vec_file=None, index_file=None, D=None, append_list=None):
        super(Emb, self).__init__( )
        emblist = [ ]
        self.dict = { }
        self.dim = 0
        self.num = 0
        self.unk = -1

        if vec_file != None:
            f = open(vec_file)
            f = f.readlines( )
            self.dim = len(f[0].strip( ).split( )) - 1
            self.num = len(f)
            for i in range(self.num):
                y = f[i].strip( ).split( )
                self.dict[y[0]] = i
                emblist.append(list(map(float, y[1:self.dim+1])))
                if y[0]=='<unk>': self.unk = i
        else:
            self.dim = D

        self.unvec_num = 0

        if self.unk == -1:
            self.dict['<unk>'] = self.num
            self.unk = self.num
            self.num += 1
            # emblist.append(rand_vec(self.dim))
            self.unvec_num += 1

        if index_file != None:
            f = open(index_file)
            f = f.readlines( )
            for x in f:
                y = x.strip( )
                if y not in self.dict.keys( ):
                    self.dict[y] = self.num
                    self.num += 1
                    # emblist.append(rand_vec(self.dim))
                    self.unvec_num += 1

        if append_list != None:
            for x in append_list:
                if x not in self.dict.keys( ):
                    self.dict[x] = self.num
                    self.num += 1
                    # emblist.append(rand_vec(self.dim))
                    self.unvec_num += 1

        self.need_vec = need_vec
        if need_vec:
            if self.unvec_num == self.num:
                self.emb = nn.Embedding(self.unvec_num, self.dim)
                # print("1")
            else:
                unvec = nn.Parameter(tch.FloatTensor(self.unvec_num, self.dim))
                nn.init.orthogonal_(unvec)
                edvec = tch.tensor(emblist)
                allvec = tch.cat((edvec, unvec), dim=0).to(device)
                self.emb = nn.Embedding.from_pretrained(allvec, freeze=freeze)
                # print("2")
            # self.zero = tch.zeros(self.dim, requires_grad=False).to(device)
        self.device = device

    def single_to_ID(self, x):
        return self.dict[x] if x in self.dict.keys() else self.unk

    def to_ID(self, input, lens):
        batch_size = len(lens)
        S = [[self.single_to_ID(x) for x in input[i]] for i in range(batch_size)]
        return S

    def Emb(self, input, lens):
        # batch_size = len(lens)
        max_len = max(lens)
        input = self.to_ID(input, lens)
        # print(max_len)
        # print(max([len(x) for x in input]))
        S = [x + [0] * (max_len - l) for l, x in zip(lens, input)]
        input = tch.tensor(S).to(self.device)
        return self.emb(input)

    def CharMean(self, str):
        L = [self.dict[x] if x in self.dict.keys( ) else self.unk for x in str]
        L = [self.emb(tch.tensor(x).to(self.device)) for x in L]
        L = tch.stack(L, dim=0)
        L = tch.mean(L, dim=0)
        return L

    def CharField(self, str, mean):
        L = [self.dict[x] if x in self.dict.keys( ) else self.unk for x in str]
        L = self.emb(tch.tensor(L).to(self.device))
        if mean: L = tch.mean(L, dim=0)
        return L

    def CharEmb(self, input, lens):
        batch_size = len(lens)
        max_len = max(lens)
        S = [ ]
        for i in range(batch_size):
            L = [self.CharMean(x) for x in input[i]]
            Z = [self.zero] * (max_len - lens[i])
            S.append(tch.stack(L + Z, dim=0))
        S = tch.stack(S, dim=0)
        return S
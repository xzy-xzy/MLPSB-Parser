import torch as tch
import torch.nn as nn
import numpy as np
from utils import eisner, eisner_tch
from modules.emb import Emb
from modules.charlstm import CharLSTM
from modules.bilstm import biLSTM
from utils.getitem import Getitem
from modules.weight_average import Weight_Average_Norate
from utils.tree_check import tree_check
from utils.cle import get_head
from torch.nn.functional import pad
import time


class Dependency(nn.Module):

    def __init__(self, share):
        super(Dependency, self).__init__( )
        self.share = share
        DIR = share['dir']
        CON = share['con']
        corp = share['corp']
        data = share['data']
        self.CON = CON
        self.DIR = DIR
        if CON.use_word:
            self.word_emb = Emb(
                need_vec=True,
                freeze=CON.wordemb_freeze,
                device=DIR.device,
                vec_file=corp + 'wordvec.txt',
                index_file=corp + 'wordlog.txt',
            )
        if CON.use_char:
            self.char_emb = Emb(
                need_vec=True,
                freeze=True,
                device=DIR.device,
                vec_file=data + CON.charemb_path
            )
        if CON.use_pos:
            self.pos_emb = Emb(
                need_vec=True,
                freeze=False,
                device=DIR.device,
                index_file=corp + 'poslog.txt',
                D=CON.dim_pos
            )
        self.rel_dic = Emb(
            need_vec=False,
            vec_file=corp + 'rellog.txt'
        )
        self.num_rel = self.rel_dic.num
        self.punct_id = self.rel_dic.dict['punct'] if "punct" in self.rel_dic.dict.keys( ) else self.rel_dic.dict['P']
        self.bilstm = biLSTM(
            hisize=CON.hisize,
            layers=CON.lstm_layers,
            insize=CON.dim_emb,
            dropout=CON.drop_rate
        )
        self.dropout = nn.Dropout(CON.drop_rate)
        self.aver_arc = Weight_Average_Norate(CON.biaff_layers + 1, CON.w_fixed)
        self.ignore_index = -100
        self.loss = nn.NLLLoss(ignore_index=self.ignore_index, reduction="sum")
        self.biaffine_time = 0
        self.scoring_time = 0
        self.mst_time = 0
        self.parsing_time = 0
        self.time_check = CON.time_check


    def get_sum(self, X, aim, lens, op=False):
        '''
        X: (B, N, C)
        '''
        '''
        B, N = X.size(0), X.size(1)
        aim = [[x[i] if i < l - 1 else self.ignore_index for i in range(N)] for x, l in zip(aim, lens)]
        aim = X.new_tensor(aim, dtype=tch.long)
        aim = aim.view(B * N)
        X = X.view(B * N, -1)
        return self.loss(tch.log(X), aim)
        '''

        B = X.size(0)
        L = [ ]
        for i in range(B):
            for j in range(lens[i] - 1):
                # print(i, j, aim[i][j], -tch.log(X[i, j, aim[i][j]]), X[i][j])
                L.append(-tch.log(X[i, j, aim[i][j]]))
        return sum(L)

    def sync(self):
        if self.time_check:
            tch.cuda.synchronize(self.DIR.device)

    def _train(self, W_arc, final_X, fa, rel, L):
        loss_arc = self.get_sum(W_arc, fa, L) * self.CON.arc_rate
        B, N = final_X.size(0), final_X.size(1)
        pad_fa = [x + [0] * (N + 1 - L[i]) for i, x in enumerate(fa)]
        pad_fa = final_X.new_tensor(pad_fa, dtype=tch.long)
        W_label = self.share["biaff"].get_label_score(final_X, pad_fa)
        loss_label = self.get_sum(W_label, rel, L, True) * self.CON.label_rate
        if self.CON.div_way == 0:
            loss = (loss_arc + loss_label) / (sum(L) - len(L))
        elif self.CON.div_way == 1:
            loss = (loss_arc + loss_label) / len(L)
        elif self.CON.div_way == 2:
            loss = (loss_arc + loss_label)
        return loss

    def _test(self, W_arc, final_X, fa, rel, L):
        self.sync( )
        start = time.time( )
        tot, r_UAS, r_LAS = 0, 0, 0
        if self.CON.use_gold_head:
            m, l = max(L), len(fa)
            for i in range(l):
                b = len(fa[i])
                for j in range(m - b):
                    fa[i].append(0)
            my_fa = tch.tensor(fa).to(self.DIR.device)

        else:
            self.sync( )
            mst_start = time.time( )
            t = 0
            if self.CON.MST_algorithm == "cle":
                my_fa, bad = tree_check(W_arc, L)
                my_fa = my_fa.cpu( )

                if len(bad) != 0:
                    bad = [i for i in range(len(L))]
                    bad_arc = tch.index_select(W_arc, dim=0, index=tch.tensor(bad).to(self.DIR.device))
                    # my_fa = eisner.eisner_dp(W_arc, L)
                    # my_fa = eisner_tch.eisner_dp_tch(W_arc, L, self.DIR.device)
                    bad_l = [L[i] for i in bad]
                    # bad_fa = eisner_tch.eisner_dp_tch(bad_arc, bad_l, self.DIR.device)
                    bad_fa = [ ]
                    m = bad_arc.size(1)
                    for x, y, i in zip(bad_arc, bad_l, bad):
                        self.sync( )
                        a = time.time( )
                        zl_fa = get_head(x[:y].cpu( ))
                        self.sync( )
                        t += time.time( ) - a
                        zl_fa = tch.tensor(zl_fa + [0] * (m - y))
                        # print(zl_fa)
                        # print(my_fa[i])
                        bad_fa.append(zl_fa)

                    for id, f in zip(bad, bad_fa):
                        my_fa[id] = f

                    # print(t)

            elif self.CON.MST_algorithm == "eisner":
                my_fa = eisner_tch.eisner_dp_tch(W_arc, L, self.DIR.device)

            self.sync( )
            self.mst_time += time.time( ) - mst_start

        self.sync( )
        lb_start = time.time( )
        W_label = self.share["biaff"].get_label_score(final_X, my_fa.to(self.DIR.device))
        self.sync( )
        self.scoring_time += time.time( ) - lb_start

        my_rel = tch.max(W_label, dim=2)[1].cpu( )
        self.sync( )
        self.parsing_time += time.time( ) - start

        for my_fa, L, fa, rel, my_rel in zip(my_fa, L, fa, rel, my_rel):
            if not self.CON.test_limit(L):
                continue
            rt_np = [rel[i] != self.punct_id for i in range(L-1)]
            rt_fa = [rt_np[i] and my_fa[i] == fa[i] for i in range(L-1)]
            rt_rel = [rt_fa[i] and my_rel[i] == rel[i] for i in range(L-1)]
            tot += sum(rt_np)
            r_UAS += sum(rt_fa)
            r_LAS += sum(rt_rel)
        return tot, r_UAS, r_LAS

    def forward(self, data, train=True):
        self.sync( )
        start = time.time( )
        wd = [x[0] for x in data]
        pos = [x[1] for x in data]
        fa = [x[2] for x in data]
        lens = [len(x) for x in wd]
        max_L = max(lens)
        rel = self.rel_dic.to_ID([x[3] for x in data], lens)
        emblist = [ ]
        if self.CON.use_word:
            emblist.append(self.word_emb.Emb(wd, lens))
        if self.CON.use_char:
            emblist.append(self.char_emb.CharEmb(wd, lens))
        if self.CON.use_bert:
            bert_e = self.share['bert_emb'](wd)
            if bert_e.size(1) != max_L:
                print("truncated")
                bert_e = pad(bert_e, (0, 0, 0, max_L - bert_e.size(1)), "constant", 0)
            emblist.append(bert_e)
        if self.CON.use_pos:
            emblist.append(self.pos_emb.Emb(pos, lens))
        emb = tch.cat(emblist, dim=2)
        emb = self.dropout(emb)
        if self.CON.use_lstm:
            fea = self.bilstm(emb, lens)
        else:
            fea = emb
        self.sync( )
        bf_start = time.time( )
        W_arc, final_X = self.share['biaff'](fea, lens)
        self.sync( )
        self.biaffine_time += time.time( ) - bf_start
        if self.CON.siamese != "N":
            W_arc = self.aver_arc(W_arc)
        if train:
            return self._train(W_arc, final_X, fa, rel, lens)
        else:
            self.sync( )
            self.scoring_time += time.time( ) - start
            self.parsing_time += time.time( ) - start
            return self._test(W_arc, final_X, fa, rel, lens)

    def getloss(self, x):
        return self.forward(x)

    def test(self, epoch, test_loader, i, name):
        # print(self.aver_arc.W.softmax(dim=0))
        tot, r_UAS, r_LAS = 0, 0, 0
        # start = time.time( )
        self.biaffine_time = 0
        self.scoring_time = 0
        self.parsing_time = 0
        self.mst_time = 0
        with tch.no_grad():
            for round, data in enumerate(test_loader):
                print("\r%s Inc=%d" % (name, round), end=' ')
                t, u, l = self.forward(Getitem(data, i), train=False)
                tot += t
                r_UAS += u
                r_LAS += l
        # print(time.time() - start)
        print( )
        if self.time_check:
            print("biaffine time: %.4f ms" % (self.biaffine_time * 1000))
            print("scoring time: %.4f s" % self.scoring_time)
            print("mst time: %.4f s" % self.mst_time)
            print("whole parsing time: %.4f s" % self.parsing_time)
        UAS = 100.0 * r_UAS / tot
        LAS = 100.0 * r_LAS / tot
        print(name + " UAS/LAS: %.4f %.4f" % (UAS, LAS))
        info = "epoch %d: %.4f %.4f\n" % (epoch, UAS, LAS)

        return info, UAS, LAS
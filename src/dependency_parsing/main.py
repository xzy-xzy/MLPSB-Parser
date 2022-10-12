import torch as tch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import DataLoader
from config import Config
from dir import Dir
from utils.dataset import Joint_Dataset
from modules.bert import BertEmbedding, XLNetEmbedding
from modules.berttool import BertTool
from modules.biaffine import MultiLayerBiaffine
from modules.emb import Emb
from utils.getitem import Getitem
from dependency import Dependency
import time
import sys
import math

'''
Basic Setting
'''

datafolder = '../../data/'
CON = Config(datafolder, sys.argv)
share = { }

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tch.manual_seed(seed)
    tch.cuda.manual_seed_all(seed)
    tch.backends.cudnn.deterministic = True

type = CON.attn_type
if type == "biaffine":
    type = ""
else:
    type = type + ","

set_seed(CON.seed - 1)
name = CON.corp + "," + CON.pretrain_name + "," + \
       "n=%d,lr=[%.2e,%.2e,%.2e],seed=%d,epoch=%d," \
       % (CON.biaff_layers, CON.bert_lr, CON.biaff_lr, CON.learning_rate, CON.seed, CON.epoch) + \
       "arc_rate=%.2f," % (CON.arc_rate) + \
       "word_d=%d,hisize=%d,arc_d=%d,label_d=%d," % (CON.dim_word, CON.hisize, CON.msize_arc, CON.msize_label) + \
       "attn=%s,sia=%s,lstm=%d,div=%d," % (CON.attn_type, CON.siamese, CON.use_lstm, CON.div_way) + \
       "pos=%d," % (CON.use_pos) + "batchsize=%d" % (CON.batchsize)

CON.biaff_layers -= 1
DIR = Dir(name, "", CON)

if CON.corp in ["PTB", "CTB"]:
    corp_type = "PC"
    corpfolder = ["../../corpus/" + CON.corp + "/"]
else:
    corp_type = "UD"
    corpfolder = ["../../corpus/UD2.2/" + CON.corp + "/"]

'''
Shared Models
'''

if CON.use_bert:
    bert_tool = BertTool(
        layer=CON.bert_layers,
        in_ch=CON.bert_dim,
        out_ch=CON.dim_word
    )
    X = BertEmbedding if CON.pretrain_type == "bert" else XLNetEmbedding
    bert_emb = X(
        path=datafolder + "/" + CON.pretrain_name,
        layer=CON.bert_layers,
        tool=bert_tool,
        dv=DIR.device,
        grad=CON.bert_train,
        append=['<root>']
    )
    DIR.Insert_model(bert_tool, "bert_tool", True)
    DIR.Insert_model(bert_emb, "bert_emb", CON.bert_train, CON.bert_lr)
    share['bert_emb'] = bert_emb

num_rel = len(open(corpfolder[0]+'rellog.txt').readlines( ))
# print(num_rel)
biaffine = MultiLayerBiaffine(
    layer_num=CON.biaff_layers,
    din=CON.hiout if CON.use_lstm else CON.dim_emb,
    dim_arc=CON.msize_arc,
    dim_label=CON.msize_label,
    num_rel=num_rel,
    dropout=CON.drop_rate,
    siamese=CON.siamese,
    label_drop=CON.label_drop,
    attn_type=CON.attn_type
)
size = 0
for name, param in biaffine.named_parameters( ):
    t = 1
    for x in param.size( ):
        t = t * x
    size += t
print("biaffine size: %.2f M" % (size / 1000000))
DIR.Insert_model(biaffine, "biaffine", True, CON.biaff_lr)
share['biaff'] = biaffine

share['dir'] = DIR
share['con'] = CON
share['corp'] = corpfolder[0]
share['data'] = datafolder

'''
Models & Tasks
'''

modelclass = [(Dependency, "DE", True)]
for x in modelclass:
    if x[2]:
        DIR.Insert_model(x[0](share), x[1], True)
        DIR.Insert_task(DIR.modelnumber - 1)
DIR.Load( )
# DIR.Show_param( )

'''
Corpus
'''

P = np.load(corpfolder[0] + "train.npy", allow_pickle=True)
MAX_L = 50
S = [0] * MAX_L
len_list = [ ]
for x in P:
    aim = len(x[0]) - 1
    len_list.append(aim)
    if aim >= MAX_L:
        aim = MAX_L - 1
    if S[aim] == 0:
        S[aim] = [ ]
    S[aim].append(x)
for i in range(MAX_L):
    if S[i] == 0:
        S[i] = [ ]

# print(sum(len_list) / len(len_list))
corplist = [x for x, y in zip(corpfolder, modelclass) if y[2]]
if corp_type ==  "PC":
    trainset = [Joint_Dataset(corplist, "train.npy", DIR, CON, [S[i]]) for i in range(MAX_L)]
elif corp_type == "UD":
    trainset = [Joint_Dataset(corplist, "train.npy", DIR, CON, [S[i]]) for i in range(2, MAX_L)]
devset = Joint_Dataset(corplist, "dev.npy", DIR, CON)
testset = Joint_Dataset(corplist, "test.npy", DIR, CON)
out_size = CON.batchsize
in_size = out_size if CON.bert_dim <= 768 else 8
if corp_type ==  "PC":
    dev_batchsize = (850 if CON.corp == "PTB" else 401) if CON.bert_dim <= 768 else 400
    test_batchsize = (1208 if CON.corp == "PTB" else 955) if CON.bert_dim <= 768 else 600
elif corp_type == "UD":
    dev_batchsize = 200 if CON.corp == "cs" else (400 if CON.corp == "en" else 800)
    test_batchsize = dev_batchsize
train_loader = [DataLoader(dataset=y, batch_size=in_size if x == MAX_L - 1 else out_size,
                           shuffle=True, collate_fn=lambda x:x) for x, y in enumerate(trainset) if y.len > 0]
dev_loader = DataLoader(dataset=devset, batch_size=dev_batchsize, collate_fn=lambda x:x)
test_loader = DataLoader(dataset=testset, batch_size=test_batchsize, collate_fn=lambda x:x)



'''
Process
'''

def Test( ):
    DIR.Is_train(False)
    testlist = [ ]
    for i, ID in enumerate(DIR.tasklist):
        testlist.append(DIR.modellist[ID].test(-1, test_loader, i, "test"))
    DIR.Deal_result(None, testlist)

def Process( ):

    L_reset = 0
    epoch = DIR.rangenow
    if epoch == 0:
        DIR.Save_model(-1, False)
    while epoch < DIR.rangenext:
        FLAG = False
        set_seed(epoch + CON.seed + L_reset)
        print("epoch%d:" % epoch)
        DIR.Is_train(True)

        if CON.time_check:
            tch.cuda.synchronize(DIR.device)
        start = time.time( )

        num = 0
        for L, loader in enumerate(train_loader):
            if FLAG: break
            for round, data in enumerate(loader):
                num += len(data)
                losslist = [ ]
                for i, ID in enumerate(DIR.tasklist):
                    losslist.append(DIR.modellist[ID].getloss(Getitem(data, i)))
                f_list = [x for x in losslist if x]
                loss = sum(f_list)
                print("\rTrain Len=%d Inc=%d Loss=%.4f" % (L+1, round, loss), end=' ')
                if math.isnan(loss) or math.isinf(loss):
                    print("Loss Error. Reload.")
                    L_reset += 1
                    DIR.Zero_grad( )
                    FLAG = True
                    break
                elif loss > 0:
                    loss.backward( )
                if CON.corp == "ru" and L == 0:     # ru
                    continue
                if num >= CON.batchsize:
                    DIR.Step( )
                    DIR.Zero_grad( )
                    num = 0
        if num > 0:
            DIR.Zero_grad( )
            DIR.Step( )

        if CON.time_check:
            tch.cuda.synchronize(DIR.device)
            print("training time: %.4f s" % (time.time( ) - start))

        if not FLAG:
            print( )
            DIR.Is_train(False)
            devlist = [ ]
            # testlist = [ ]
            for i, ID in enumerate(DIR.tasklist):
                devlist.append(DIR.modellist[ID].test(epoch, dev_loader, i, "dev"))
                # testlist.append(DIR.modellist[ID].test(epoch, test_loader, i, "test"))
            upd = DIR.Deal_result(devlist, None)
            DIR.Save_model(epoch, upd)
            # print("epoch %d Test Finished" % epoch)
            epoch += 1
        else:
            DIR.Load( )

    DIR.Remove_opt( )


if DIR.train_mode == "train":
    Process( )
Test( )
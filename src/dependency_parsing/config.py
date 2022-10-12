import json

class Config:
    def __init__(self, datafolder, argv):

        self.pretrain_name = "XLNet_base"
        f = open(datafolder + self.pretrain_name + "/config.json", "r")
        info = json.load(f)
        if self.pretrain_name[0:4].lower( ) == "bert":
            self.pretrain_type = "bert"
            self.bert_dim = info["hidden_size"]
        elif self.pretrain_name[0:5].lower( ) == "xlnet":
            self.pretrain_type = "XLNet"
            self.bert_dim = info["d_model"]
        else:
            assert False

        self.corp = "PTB"
        self.use_pos = self.corp != "PTB"
        self.learning_rate = 5e-4
        self.bert_lr = 1e-5
        self.biaff_lr = 5e-4
        self.biaff_layers = 5
        self.hisize = 512
        self.msize_arc = 512
        self.msize_label = 128
        self.lstm_layers = 3
        self.bert_layers = 4

        self.siamese = "P"  # (P, T, N)
        self.use_lstm = True

        self.dim_pos = 50
        self.dim_word = 300
        self.dim_char = 300

        self.use_word = (self.pretrain_type == 'word')
        self.use_char = False
        self.charemb_path = None
        self.use_bert = (self.pretrain_type != 'word')
        self.dim_emb = (self.use_word + self.use_bert) * self.dim_word + self.use_pos * self.dim_pos + self.use_char * self.dim_char
        self.hiout = 2 * self.hisize

        self.drop_rate = 1.0 / 3
        self.arc_rate = 0.55
        self.label_rate = 0.45
        self.wordemb_freeze = True
        self.bert_train = True
        self.w_fixed = False
        self.label_drop = False
        self.div_way = 2 if self.corp == "PTB" else 1
        self.attn_type = "biaffine"
        self.MST_algorithm = "eisner" if self.corp in ["PTB", "CTB"] else "cle"
        self.time_check = False

        self.batchsize = 24
        self.seed = 1
        self.norm = 1
        self.epoch = 20 if self.corp == "CTB" else 8
        self.use_gold_head = False
        self.test_limit = (lambda x: True)
        # self.test_limit = (lambda x: x - 1 >= 28)
        # self.test_limit = (lambda x: x - 1 < 28)

        if len(argv) >= 2 and argv[1].isdigit( ):
            self.biaff_layers = int(argv[1])
        if len(argv) >= 3 and argv[2].isdigit( ):
            self.seed = int(argv[2])


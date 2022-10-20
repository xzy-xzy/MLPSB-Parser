import torch as tch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.nn.utils as uti
import sys
import os

class Dir:
    def __init__(self, t_name, prename, CON):
        self.name = t_name
        self.prename = prename
        self.CON = CON
        self.modelfolder = '../model/' + self.name + '/'
        self.resfolder = '../result/' + self.name + '/'
        self.pre_modelfolder = '../model/' + self.prename + '/'
        self.checkpoint = self.modelfolder + 'checkpoint.txt'
        self.bestpoint = self.modelfolder + 'bestpoint.txt'
        self.modeldir = [ ]
        self.devdir = [ ]
        self.testdir = [ ]
        self.modellist = [ ]
        self.modelname = [ ]
        self.tasklist = [ ]
        self.optlist = [ ]
        self.schlist = [ ]
        self.chlist = [ ]
        self.frombest = [ ]
        self.lrlist = [ ]
        self.decaylist = [ ]
        self.modelnumber = 0
        self.norm = CON.norm

        self.train_mode = "train"
        self.precheck_mode = 0
        self.save_model = 1
        self.save_result = 1
        self.have_model = 1
        self.from_checkpoint = 1
        self.bestmode = 0
        self.metric = -1e9
        self.UAS = 0
        self.LAS = 0
        self.cortic = ""

        print(t_name)
        self.device = tch.device("cuda:1" if tch.cuda.is_available() else "cpu")
        print(self.device)

        self.rangenow = 0
        self.rangenext = CON.epoch

        if self.from_checkpoint == 1 and os.path.exists(self.checkpoint):
            x = open(self.checkpoint, 'r')
            s = x.readline( ).strip( )
            if s != "":
                self.rangenow = int(s) + 1

        if os.path.exists(self.bestpoint):
            x = open(self.bestpoint, 'r')
            s = x.readline( ).strip( )
            if s != "":
                self.metric = float(s.split( )[1])
            self.cortic = x.readline( ).strip( )

        if not os.path.exists(self.modelfolder): os.makedirs(self.modelfolder)
        if not os.path.exists(self.resfolder): os.makedirs(self.resfolder)

        if len(sys.argv) >= 2 and sys.argv[1] == "test":
            self.train_mode = "test"
            self.save_model = 0
        if len(sys.argv) >= 3 and sys.argv[2] == "pre":
            self.precheck_mode = 1
        if len(sys.argv) >= 3 and sys.argv[2] == "testb":
            self.train_mode = "test"
            self.best_mode = 1
            self.save_model = 0

    def Insert_model(self, x, name, change, lr=None):
        self.modellist.append(x)
        x.to(self.device)
        LR = lr if lr else self.CON.learning_rate
        opt = optim.Adam(x.parameters( ), lr=LR)
        # sch = sched.ReduceLROnPlateau(opt, mode='max', factor=self.CON.step_decay, patience=self.CON.step_patience, verbose=True)
        self.optlist.append(opt)
        # self.schlist.append(sch)
        self.decaylist.append(LR / self.CON.epoch)
        self.lrlist.append(lr if lr else self.CON.learning_rate)
        mf = self.modelfolder if change else self.pre_modelfolder
        self.modeldir.append(mf + name)
        self.devdir.append(self.resfolder + name + "_dev_result.txt")
        self.testdir.append(self.resfolder + name + "_test_result.txt")
        self.chlist.append(change)
        self.modelname.append(name)
        self.frombest.append(True if self.bestmode else (not change))
        if not change: self.modellist[self.modelnumber].train(mode=False)
        self.modelnumber += 1

    def Insert_task(self, x):
        self.tasklist.append(x)

    def Is_train(self, P):
        for i in range(self.modelnumber):
            if self.chlist[i]: self.modellist[i].train(mode=P)

    def Zero_grad(self):
        for i in range(self.modelnumber): self.modellist[i].zero_grad( )

    def Step(self):
        for i in range(self.modelnumber):
            if self.chlist[i]: uti.clip_grad_norm_(self.modellist[i].parameters( ), max_norm=self.norm, norm_type=2)
        for i in range(self.modelnumber):
            if self.chlist[i]: self.optlist[i].step( )

    def Schedule(self, val=None):
        for i in range(self.modelnumber):
            if self.chlist[i]:
                for x in self.optlist[i].param_groups:
                    x['lr'] -= self.decaylist[i]

    def Save_model(self, epoch, upd):
        if self.save_model == 1:
            for i in range(self.modelnumber):
                if self.chlist[i]:
                    tch.save(self.modellist[i].state_dict( ), self.modeldir[i] + '_LATEST')
                    tch.save(self.optlist[i].state_dict( ), self.modeldir[i] + '_opt_LATEST')
                    # tch.save(self.schlist[i].state_dict( ), self.modeldir[i] + '_sch_LATEST')
            checkfile = open(self.checkpoint, 'w')
            checkfile.write("%d" % epoch)
            checkfile.close( )
            if upd:
                for i in range(self.modelnumber):
                    if self.chlist[i]:
                        tch.save(self.modellist[i].state_dict(), self.modeldir[i] + '_BEST')
                        tch.save(self.optlist[i].state_dict(), self.modeldir[i] + '_opt_BEST')
                        # tch.save(self.schlist[i].state_dict(), self.modeldir[i] + '_sch_BEST')
                bestfile = open(self.bestpoint, 'w')
                bestfile.write("%d %.8f %.8f %.8f\n" % (epoch, self.metric, self.UAS, self.LAS))
                bestfile.write("cortest: " + self.cortic)
                bestfile.close( )
        print("epoch %d Model&CheckPoint Updated\n" % epoch)

    def Load(self):
        for i in range(self.modelnumber):
            try:
                FB = '_BEST' if self.frombest[i] else '_LATEST'
                self.modellist[i].load_state_dict(tch.load(self.modeldir[i] + FB))
                if self.rangenow != self.rangenext and self.train_mode != "test":
                    self.optlist[i].load_state_dict(tch.load(self.modeldir[i] + '_opt' + FB))
                print(self.modelname[i] + " Found")
            except Exception as e:
                print(e)
                print(self.modelname[i] + " Not Found")

    def Remove_opt(self):
        for i in range(self.modelnumber):
            for FB in ['_BEST', '_LATEST']:
                name = self.modeldir[i] + '_opt' + FB
                if os.path.isfile(name):
                    os.remove(name)

    def Show_param(self):
        for i in range(self.modelnumber):
            model = self.modellist[i]
            for name, param in model.named_parameters( ):
                print(name, param)

    def Deal_result(self, devlist, testlist, aim=0):
        if devlist:
            for i, ID in enumerate(self.tasklist):
                g = open(self.devdir[ID], 'a')
                g.write(devlist[i][0])
            UAS, LAS = devlist[aim][1], devlist[aim][2]
            self.Schedule(UAS + LAS)
            upd = (UAS + LAS > self.metric)
            if upd:
                self.metric = UAS + LAS
                self.UAS, self.LAS = UAS, LAS
            return upd
        if testlist:
            for i, ID in enumerate(self.tasklist):
                g = open(self.testdir[ID], 'a')
                g.write(testlist[i][0])
from torch.utils.data import Dataset
import numpy as np

class Joint_Dataset(Dataset):

    def __init__(self, corplist, filename, DIR, CON, data=None):

        datalist = [np.load(corp + filename, allow_pickle=True) for corp in corplist] if not data else data
        n = len(datalist)
        data = datalist[0]
        data = [[x] for x in data]
        L = len(data)
        for i in range(1, n):
            corp = datalist[i]
            assert len(corp) == L
            for j in range(L):
                data[j].append(corp[j])
        self.data = data
        if DIR.precheck_mode: self.data = self.data[0:2]
        self.len = len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

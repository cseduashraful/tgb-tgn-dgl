import torch
from torch.utils.data import Dataset

class TemporalGraphDataset(Dataset):
    def __init__(self, src, dst, t, msg, batch = None):
        """
        Initialize the dataset with source nodes, destination nodes, time values, and features.
        """
        self.src = src
        self.dst = dst
        self.t = t
        self.msg = msg
        self.batch = batch
        # self.uniqueDest = torch.unique(dst)
        # self.dstCnt = len(self.uniqueDest)

    def __len__(self):
        """
        Return the total number of data points.
        """
        return len(self.src)
    
    def __getNegativeDest(self, exclude):

        while True:
            ridx = torch.randint(0, self.dstCnt, (1,)).item()
            negdst = self.uniqueDest[ridx]
            if negdst.item() != exclude:
                return negdst




    def __getitem__(self, idx):
        """
        Return a single data point as a dictionary.
        """
        if self.batch is None:
            return {
                'src': self.src[idx],
                'dst': self.dst[idx],
                't': self.t[idx].float(),
                'msg': self.msg[idx],
                'idx': idx
            }
        else:
            # dst = self.dst[idx].item()

            return {
                'src': self.src[idx],
                'dst': self.dst[idx],
                # 'ndst': self.__getNegativeDest(dst),
                't': self.t[idx].float(),
                'msg': self.msg[idx],
                'b':self.batch[idx],
                'idx': idx
            }


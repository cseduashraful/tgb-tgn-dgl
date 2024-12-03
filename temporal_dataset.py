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

# Example Data
# data = {
#     "src": torch.tensor([0, 1, 2, 3, 4]),
#     "dst": torch.tensor([1, 2, 3, 4, 5]),
#     "t": torch.tensor([10, 20, 30, 40, 50]),
#     "msg": torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])  # Features as 2D vectors
# }

# # Initialize the Dataset
# dataset = TemporalGraphDataset(data['src'], data['dst'], data['t'], data['msg'])

# # Create the DataLoader
# batch_size = 2
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # Iterate through the DataLoader
# for batch in dataloader:
#     print("Batch:")
#     print(batch)

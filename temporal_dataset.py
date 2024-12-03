import torch
from torch.utils.data import Dataset

class TemporalGraphDataset(Dataset):
    def __init__(self, src, dst, t, msg):
        """
        Initialize the dataset with source nodes, destination nodes, time values, and features.
        """
        self.src = src
        self.dst = dst
        self.t = t
        self.msg = msg

    def __len__(self):
        """
        Return the total number of data points.
        """
        return len(self.src)

    def __getitem__(self, idx):
        """
        Return a single data point as a dictionary.
        """
        return {
            'src': self.src[idx],
            'dst': self.dst[idx],
            't': self.t[idx].float(),
            'msg': self.msg[idx],
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

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class s2fDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.trans = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        data, label = self.x[idx], self.y[idx]
        label = torch.tensor(label, dtype=torch.float32)
        tensor = self.trans(data)

        return tensor, label


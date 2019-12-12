import torch
from torch.utils.data import Dataset


class SignatureDataset(Dataset):
    """Signatures dataset."""

    def __init__(self, images, labels):
        """
        :param images: list of signatures as PIL images
        :param labels: list of true labels
        """
        self.X = torch.stack(images)
        self.y = labels

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'X': self.X[idx], 'y': self.y[idx]}
        return sample

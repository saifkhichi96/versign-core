import torch


class FeatureExtractor:
    def __init__(self, model):
        self.net = model

    def extract(self, dataset):
        with torch.no_grad():
            features = self.net(dataset.X)
            return features.numpy()

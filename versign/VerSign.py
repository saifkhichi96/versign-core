import numpy as np
import torch

from PIL import Image
from sigver.featurelearning.models import SigNet
from torchvision.transforms import *

from .train_test import Classifier
from .utils import preprocess_signature


class VerSign:
    def __init__(self, model, imsize):
        """Default public constructor.

        Parameters:
            model : a trained PyTorch model for extracting signature features
            imsize (tuple) : dimensions (w,h) of the input images to the model
        """
        state_dict, _, _ = torch.load(model)
        self.__model = SigNet().eval()
        self.__model.load_state_dict(state_dict)
        self.__imsize = imsize
        self.__clf = Classifier()

    def transform(self, im):
        """Pre-process a signature image and convert it to a tensor.
        """
        im = Image.fromarray(preprocess_signature(np.array(im), (self.__imsize)))
        im = Compose([
            Resize(self.__imsize),
            ToTensor(),
        ])(im)
        return im.view(-1, self.__imsize[0], self.__imsize[1])

    def fit(self, X):
        """Fits model on the training data.

        All input images are assumed to be from a single class. This method
        trains a OneClassSVM on writer-independent features in these signature
        images.

        Parameters:
            X (list) : list of signature images for training

        Returns:
            a OneClassSVM trained on input data
        """
        with torch.no_grad():
            x = self.__model(torch.stack([self.transform(Image.open(i).convert('L')) for i in X])).numpy()
            self.__clf.fit(x)

    def predict(self, X, thresh=0.5):
        """Classifies signatures as genuine or forged.

        Parameters:
            X (list) : list of signature images for testing

        Returns:
            y (list) : list of predicted labels
        """
        with torch.no_grad():
            x = self.__model(torch.stack([self.transform(Image.open(i).convert('L')) for i in X])).numpy()
            if isinstance(thresh, list):
                y_pred = {}
                for t in thresh:
                    y_pred[t] = [1 if y > t else -1 for y in self.__clf.decision_function(x)]
                return y_pred
            else:
                return [1 if y > thresh else -1 for y in self.__clf.decision_function(x)]

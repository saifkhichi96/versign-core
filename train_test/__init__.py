import os
import shutil
from os import listdir
from os.path import splitext, join, exists

import cv2
import numpy as np
import scipy.io as io
import torch
from PIL import Image
from sigver.featurelearning.models import SigNet
from sklearn import svm
from torch.utils.data import Dataset
from torchvision.transforms import *

from preprocess import preprocess_signature
from segment import extract_from_grid


class SignatureDataset(Dataset):
    """Signatures dataset."""

    def __init__(self, user_id, input_dir, labels=None,
                 transform=Compose([Resize((150, 220)), ToTensor()])):
        """
        :arg user_id unique id for the user
        :arg input_dir name of directory containing signatures of a single user
        :arg labels (optional) list of true labels
        :arg preprocess (optional) a callable function which takes and pre-processes a PIL image
        :arg transform (optional) transforms to apply on input image
        """
        self.id = user_id
        self.X = []
        self.y = [] if labels is None else labels

        valid_exts = ['.jpg', '.jpeg', '.tif', '.tiff', '.png']
        image_files = [i for i in sorted(listdir(input_dir)) if
                       splitext(i)[1].lower() in valid_exts and not i.startswith('.')]

        for fn in image_files:
            infile = join(input_dir, fn)
            im = Image.open(infile)
            im = preprocess_signature(im, (150, 220))
            im = transform(im)
            im = im.view(-1, 150, 220)

            self.X.append(im)
            if not labels:
                self.y.append(-1 if fn.lower().startswith('f') else 1)

        self.X = torch.stack(self.X)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'X': self.X[idx], 'y': self.y[idx], 'id': self.id}
        return sample


class FeatureExtractor:
    def __init__(self, model):
        state_dict, classification_layer, forg_layer = torch.load(model)
        self.net = SigNet().eval()
        self.net.load_state_dict(state_dict)

    def extract(self, dataset):
        with torch.no_grad():
            features = self.net(dataset.X)
            return features.numpy()


class Classifier:
    def __init__(self):
        self.clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=2 ** -11)

    def fit(self, x):
        self.clf.fit(x)

    def predict(self, x):
        return self.clf.predict(x)

    def decision_function(self, x):
        y = np.array(self.clf.decision_function(x))

        # Rescale in range [0,1]
        y -= np.min(y)
        y /= np.max(y)

        return y

    def fit_and_predict(self, x_train, x_test):
        self.fit(x_train)

        y_train = self.predict(x_train)
        train_error = y_train[y_train == -1].size

        y_test = self.predict(x_test)
        y_prob = self.decision_function(x_test)

        return y_test, y_prob, train_error


def train(images_dir, features_dir, model, labels=None, preprocess=None):
    # Create dataset
    user_id = images_dir.split('/')[-1]
    dataset = SignatureDataset(user_id, images_dir, labels, preprocess, Compose([Resize((150, 220)),
                                                                                 RandomRotation(5),
                                                                                 ToTensor()]))

    # Extract features
    extractor = FeatureExtractor(model)
    features = extractor.extract(dataset)

    # Write features to file
    if features_dir:
        io.savemat(join(features_dir, user_id + '.mat'), {'features': features})

    return features


def test(data_path, model, save_path, labels=None, preprocess=None):
    # Read train features from disk
    user_id = data_path.split('/')[-1]
    train_mat = join(save_path, user_id + '.mat')
    if not exists(train_mat):  # training data for user must exist
        return None

    x_train = io.loadmat(train_mat)['features']

    # Extract test features
    dataset = SignatureDataset(user_id, data_path, labels, preprocess)
    extractor = FeatureExtractor(model)
    x_test = extractor.extract(dataset)

    # Perform classification
    clf = Classifier()
    y_test, y_prob, train_error = clf.fit_and_predict(x_train, x_test)
    y_true = dataset.y
    return user_id, y_test, y_true, y_prob, train_error


def is_registered(user_id, data_path="db/users/", root_dir=""):
    """
    Returns true if the specified user exists, false otherwise.

    :param user_id: identifier of the user to check
    :param data_path: relative path of users database (default: 'db/users/')
    :param root_dir: relative path of API's root directory (default: '')
    """
    path = root_dir + data_path + user_id + ".mat"
    return os.path.exists(path)


def register(user_id, im_grid, data_path="db/users/", root_dir=""):
    """
    Registers a new user with the system, if no user with given user ID already exists.

    Returns a boolean indicating whether the registration was successful or not.

    :param user_id: a unique string identifying the new user
    :param im_grid: an numpy array containing image of four reference signatures of the user in a grid
    :param data_path: relative path of users database (default: 'db/users/')
    :param root_dir: relative path of API's root directory (default: '')
    """
    if is_registered(user_id, data_path, root_dir):
        return False

    # Create required directories for new user
    images_dir = root_dir + data_path + user_id + "/"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    signatures = extract_from_grid(im_grid)
    print("found %s signatures" % len(signatures))

    # todo: augment signatures to generate more samples

    # Save all signatures
    for i, signature in enumerate(signatures):
        outfile = os.path.join(images_dir, "R%03d.png" % i)
        cv2.imwrite(outfile, signature)
        i += 1

    # Extract features from reference signatures
    features_dir = root_dir + data_path
    train(images_dir, features_dir, root_dir + "db/models/sabourin/signet.pth")
    return True


def unregister(user_id, root_dir=""):
    """
    Removes an existing user from system database.

    :param user_id: identifier of the user to remove
    :param root_dir: relative path of API's root directory (default: '')
    """
    images = root_dir + "db/users/" + user_id + "/"
    features = root_dir + "db/users/" + user_id + ".mat"
    if os.path.exists(images):
        shutil.rmtree(images)

    if os.path.exists(features):
        os.remove(features)

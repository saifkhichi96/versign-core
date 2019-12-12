from os import listdir
from os.path import splitext, join, isdir

import numpy as np
import scipy.io as io
from PIL import Image
from torchvision.transforms import *

from .train_test import FeatureExtractor, Classifier, SignatureDataset
from .utils import preprocess_signature
from .utils.segment import extract_from_check


class VerSign:
    def __init__(self, input_size, extraction_model, segmentation_model):
        """
        Default constructor.

        :param input_size: dimensions as (w,h) of the input images specified model expects
        :param extraction_model: the model to be used as feature extractor
        :param segmentation_model: the model to be used for signature segmentation from checks
        """
        self.__extractor = FeatureExtractor(extraction_model)
        self.__segmentation_model = segmentation_model
        self.__input_size = input_size
        self.__valid_exts = ['.jpg', '.jpeg', '.png', '.tiff', '.tif']

    def __preprocess_signature(self, im):
        im = preprocess_signature(im, self.__input_size)
        im = Compose([Resize(self.__input_size), ToTensor()])(im)
        return im.view(-1, self.__input_size[0], self.__input_size[1])

    def __prepare_data(self, images_dir, labels):
        x = [Image.open(join(images_dir, i)).convert("L") for i in sorted(listdir(images_dir)) if
             splitext(i)[1].lower() in self.__valid_exts and not i.startswith('.')]
        y = labels

        for i, im in enumerate(x):
            x[i] = self.__preprocess_signature(im)

        return SignatureDataset(x, y)

    def __prepare_data_from_checks(self, images_dir, labels):
        files = [i for i in sorted(listdir(images_dir)) if
                 splitext(i)[1].lower() in self.__valid_exts and not i.startswith('.')]

        x = []
        y = []
        for i, f in enumerate(files):
            try:
                im = Image.open(join(images_dir, f)).convert("L")
                im = extract_from_check(im, self.__segmentation_model)

                x.append(self.__preprocess_signature(im))
                y.append(labels[i])
            except:
                pass

        return SignatureDataset(x, y)

    def train_all(self, input_dir, output_dir=None):
        users = [i for i in sorted(listdir(input_dir)) if isdir(join(input_dir, i))]
        features = []
        for u in users:
            features.append(self.train(join(input_dir, u), user_id=u, output_dir=output_dir))

        return features

    def train(self, input_dir, user_id=None, output_dir=None):
        """
        Extract features from all the signatures in the input directory.

        All input samples are assumed to be genuine signatures. The extracted features
        can optionally be saved as a single .mat file in the output directory if both
        the user_id and output_dir are specified.

        :param input_dir: folder containing signatures of a single user
        :param user_id: unique id of the user to be used as name of the output file
        :param output_dir: folder where the extracted features will be saved
        :return: None
        """
        # Each input sample has label 1 (i.e. genuine signature)
        labels = np.ones(len([i for i in sorted(listdir(input_dir)) if
                              splitext(i)[1].lower() in self.__valid_exts and not i.startswith('.')])).tolist()

        # Prepare our dataset
        train_data = self.__prepare_data(input_dir, labels)

        # Extract features
        features = self.__extractor.extract(train_data)

        # Write features to file
        if output_dir is not None and user_id is not None:
            io.savemat(join(output_dir, user_id + '.mat'), {'features': features})

        return features

    def test_all(self, input_dir, features_dir):
        users = [i for i in sorted(listdir(input_dir)) if isdir(join(input_dir, i))]
        features = [i for i in sorted(listdir(features_dir)) if splitext(i)[1].lower() == '.mat']

        results = []
        for f in features:
            u = splitext(f)[0]
            if u in users:
                results.append(self.test(join(input_dir, u), train_mat=join(features_dir, f)))

        return np.array(results)

    def test(self, test_dir, train_vector=None, train_mat=None, is_check=False, y_true=None):
        """
        Classifies all the signatures on the images in the input directory as either
        genuine (1) or forged (-1). A OneClassSVM is trained using the training data
        provided either as a feature vector or a filename containing feature vector,
        which is then used to perform classification on all input images.

        If neither a training vector nor a file name is provided, method returns null.
        It is assumed the input files are signatures unless it is specified otherwise.

        :param test_dir: folder containing input images (either signatures or bank checks)
        :param y_true:
        :param train_vector: the feature vector obtained during training (default: None)
        :param train_mat: path of .mat file containing features extracted during training (default: None)
        :param is_check: set True if input images are bank checks instead of signatures (default: False)
        :return: vector containing classification results, or None
        """
        # Training features in at least one format are required
        if train_vector is None and train_mat is None:
            return None

        # Read training features
        x_train = []
        if train_vector is not None:
            x_train = train_vector
        elif train_mat is not None:
            x_train = io.loadmat(train_mat)['features']

        # Generate random labels if no labels are given
        if y_true is None:
            y_true = np.ones(len([i for i in sorted(listdir(test_dir)) if
                                  splitext(i)[1].lower() in self.__valid_exts and not i.startswith('.')])).tolist()

        # Prepare test dataset
        if is_check:
            test_data = self.__prepare_data_from_checks(test_dir, y_true)
        else:
            test_data = self.__prepare_data(test_dir, y_true)

        # Perform classification
        x_test = self.__extractor.extract(test_data)
        results = Classifier().fit_and_predict(x_train, x_test)
        return results[0]

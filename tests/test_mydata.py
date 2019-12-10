import os

import numpy as np
import scipy.io
from sklearn.metrics import accuracy_score
from src import classifiers

db = "db/features/"
dir = db + "mydata"
print("Dataset:", dir)
print("No. of users:", 1)

# Load training data
x_train = []
for f in os.listdir(dir + "/Reference/"):
    if f.endswith(".mat"):
        mat = scipy.io.loadmat(dir + "/Reference/" + f)
        feat = np.array(mat['feature_vector'][0][:])
        x_train.append(feat)

# Load test data
x_test = []
for f in os.listdir(dir + "/Questioned/"):
    if f.endswith(".mat"):
        mat = scipy.io.loadmat(dir + "/Questioned/" + f)
        feat = np.array(mat['feature_vector'][0][:])
        x_test.append(feat)

y_true = np.concatenate([np.ones(35) * -1, np.ones(5)])

# Get predictions from OneClassSVM
Y_test, Y_train, n_error_train, Y_prob = classifiers.OneClassSVM(x_train, x_test)

# Calculate prediction error
print("Prediction accuracy:", round(accuracy_score(y_true, Y_test) * 100, 2), "\n")

import argparse
import os

import numpy as np
import scipy.io
from PIL import Image

from . import Classifier

parser = argparse.ArgumentParser(description='Performs signature verification.')
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument("--sign", dest="sign", required=True, help="signature image to verify")
requiredNamed.add_argument("--user", dest="user", required=True, help="id of user to perform verification against")
args = parser.parse_args()

rootDir = "verisign-core/db/users/"
refDir = rootDir + args.user + "/"
if not os.path.isdir(refDir):
    print("User", args.user, "is not registered.")
    exit(0)

if not os.path.exists(args.sign):
    print("Invalid input. Cannot perform verification.")
    exit(-1)

# Create a temporary directory
tmpDir = rootDir + args.user + "/_temp/"
if not os.path.exists(tmpDir):
    os.mkdir(tmpDir)

# Move questioned signature to this directory

Image.open(args.sign).save(tmpDir + "Q001.jpg")

# Extract features from the questioned signature
os.system("python verisign-core/src/libs/sigver_wiwd/process_folder.py " + tmpDir + "/ " + tmpDir + "/")

# Load training data
x_train = []
for f in os.listdir(refDir):
    if f.endswith(".mat"):
        mat = scipy.io.loadmat(refDir + f)
        feat = np.array(mat['feature_vector'][0][:])
        x_train.append(feat)

# Load test data
x_test = []
for f in os.listdir(tmpDir):
    if f.endswith(".mat"):
        mat = scipy.io.loadmat(tmpDir + f)
        feat = np.array(mat['feature_vector'][0][:])
        x_test.append(feat)

# Get predictions from OneClassSVM
y_test, y_prob, train_error = Classifier().fit_and_predict(x_train, x_test)

# Show results
for y in y_test:
    f = open("status", "w")
    if y == 1:
        f.write(str(True))
    else:
        f.write(str(False))
    f.close()

# Remove temporary directory
# if os.path.exists(tmpDir):
#    os.removedirs(tmpDir)

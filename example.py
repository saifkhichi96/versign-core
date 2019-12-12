import shutil
from os import listdir, makedirs
from os.path import isdir, join, splitext, exists

import joblib
import numpy as np
import torch
from sigver.featurelearning.models import SigNet

from versign import VerSign
from versign.metrics import accuracy_score

# Define dataset locations
root = '../clients/db/datasets/Signatures/CustomDataset/'
train = join(root, 'Ref/')
test = join(root, 'Questioned/')
out = join(root, 'Temp/')
if not exists(out):
    makedirs(out)

# Load feature extraction model
print("Loading pre-trained model for feature extraction...")
state_dict, classification_layer, forg_layer = torch.load('models/signet.pth')
net = SigNet().eval()
net.load_state_dict(state_dict)

# Load signature segmentation model
print('loading segmentation model...')
clf = joblib.load("models/versign_segment.pkl")

v = VerSign(input_size=(150, 220), extraction_model=net, segmentation_model=clf)

# Extract features from training data
print("Extracting features from training images...")
v.train_all(train, out)

# Evaluate test data
print("Extracting features from and classifying test images...")
results = v.test_all(test, out)

# Delete extracted features from disc
shutil.rmtree(out)

# Load ground truth
print("Reading groundtruth so that test accuracy can be measured...")
ground_truth = []
users = [i for i in sorted(listdir(test)) if isdir(join(test, i))]
for user in users:
    images = [i for i in sorted(listdir(join(test, user))) if
              splitext(i)[1].lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff'] and not i.startswith('.')]

    y = []
    for im in images:
        if 'f' not in im.lower():
            y.append(1)
        else:
            y.append(-1)

    # y = np.loadtxt(join(join(test, user), 'groundtruth.txt'), int).tolist()
    ground_truth.append(y)

print("Calculating accuracy score...")
summary, details = accuracy_score(results, np.array(ground_truth))

print('STATS SUMMARY:', summary)
print('DETAILED STATS:')
for i in details.keys():
    print("\t%s: %.02f%% accuracy (FAR: %.2f, FRR: %.2f)" %
          (i, details[i]['accuracy'],
           details[i]['false-pos'] / (details[i]['right'] + details[i]['wrong']) * 100,
           details[i]['false-neg'] / (details[i]['right'] + details[i]['wrong']) * 100))

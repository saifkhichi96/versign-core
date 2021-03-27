from os import listdir, makedirs
from os.path import isdir, join, splitext, exists

import joblib
import numpy as np
import tqdm

from versign import VerSign
from versign.metrics import accuracy_scores, calc_equal_error

# Define dataset locations
root = '../../authentica/sources/db/datasets/Signatures/CustomDataset/'
train = join(root, 'Ref/')
test = join(root, 'Questioned/')

sensitivity = np.arange(0,1,0.01)
FARs = []
FRRs = []

valid_uids = []
for uid in listdir(train):
    train_data = join(train, uid)
    test_data = join(test, uid)
    if isdir(train_data) and isdir(test_data):
        valid_uids.append(uid)

for uid in tqdm.tqdm(valid_uids):
    train_data = join(train, uid)
    test_data = join(test, uid)

    # Load training data
    n_train = 8
    x_train = [join(train_data, f) for f in sorted(listdir(train_data))]

    # Load test data and labels
    x_test = [join(test_data, f) for f in sorted(listdir(test_data))]
    y_true = [(1 if 'f' not in f.split('.')[0].lower() else -1) for f in sorted(listdir(test_data))]

    # Train a writer-dependent model from training data
    v = VerSign('models/signet.pth', (150, 220))
    v.fit(x_train)

    # Predict labels of test data
    y_preds = v.predict(x_test, list(sensitivity))

    # Compare predictions with groundtruth
    far, frr = accuracy_scores(y_preds, y_true)
    FARs.append(far)
    FRRs.append(frr)

FARs = np.mean(np.array(FARs), axis=0)
FRRs = np.mean(np.array(FRRs), axis=0)

eer, plt = calc_equal_error(FARs, FRRs, sensitivity)
print(f'ERR: {eer:.2f}%')
plt.show()

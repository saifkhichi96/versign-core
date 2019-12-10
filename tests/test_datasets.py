import os

import numpy as np
import scipy.io
from sklearn.metrics import accuracy_score
from src import classifiers

datasets = ["gpds_signet/development/",
            "gpds_signet/exploitation/",
            "gpds_signet/validation/",
            "mcyt_signet/",
            "cedar_signet/"]

for dataset in datasets:
    # The directory to read from
    root = "db/features/Downloaded/" + dataset
    if not root.endswith("/"):
        root += "/"

    # Get list of users in the directory
    users = []
    for file in os.listdir(root):
        if file.endswith('.mat'):
            idx = file.split('_')[1].split('.')[0]
            if idx not in users:
                users.append(idx)

    accuracy = []
    # Run classifier for each of the users
    for user in users:
        real = 'real_' + str(user) + '.mat'
        forg = 'forg_' + str(user) + '.mat'

        # Load genuine signatures and forgeries
        G = scipy.io.loadmat(root + real)['features']  # N genuine signatures
        F = scipy.io.loadmat(root + forg)['features']  # M forged signatures

        # Training data comprises of N-4 genuine signatures
        x_train = G[:-4]

        # Test data comprises of 4 genuine and M forged signatures
        x_test = np.concatenate([G[-4:], F])
        y_true = np.concatenate([np.ones(4), np.ones((len(F))) * -1])

        # Get predictions from OneClassSVM
        y_test, y_train, n_error_train, y_prob = classifiers.OneClassSVM(x_train, x_test)

        # Calculate prediction error
        accuracy.append(accuracy_score(y_true, y_test))

    # Display results
    print("Dataset:", root)
    print("No. of users:", len(users))
    print("Prediction accuracy:", round(np.mean(accuracy) * 100, 2), "\n")

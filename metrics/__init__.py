import os

import numpy as np
import scipy.io
from matplotlib import pyplot as plt

import classifiers


def read_feature_vector(filename):
    mat = scipy.io.loadmat(filename)
    return np.array(mat['feature_vector'][0][:])


def read_train_data(train_dir):
    # Read in train files' names
    train_data = [f for f in os.listdir(train_dir) if f.endswith('mat')]
    train_data.sort()

    # Get feature vectors
    x_train = []
    for f in train_data:
        x_train.append(read_feature_vector(train_dir + '/' + f))

    return x_train


def read_test_data(test_dir):
    # Read in test files' names
    test_data = [f for f in os.listdir(test_dir) if f.endswith('mat')]
    test_data.sort()

    # Read groundtruth
    groundtruth = None
    if os.path.exists(test_dir + '/groundtruth.txt'):
        groundtruth = np.loadtxt(test_dir + '/groundtruth.txt').flatten()

    files = []
    x_test = []
    y_true = []
    for f in test_data:
        files.append(f[:-4])

        # Get feature vector
        x_test.append(read_feature_vector(test_dir + '/' + f))

        # Read label
        label = 1
        if groundtruth is not None:
            id = int(f[1:-4]) - 1
            label = groundtruth[id]
        else:
            if f.startswith('F'):
                label = -1
            elif f.startswith('G'):
                label = 1
            elif len(f) == 14:
                label = -1
            elif len(f) == 10:
                label = 1
            elif len(f) == 16:
                if int(f[4:7]) == int(f[9:12]):
                    label = 1
                else:
                    label = -1

        y_true.append(int(label))

    return files, x_test, y_true


def get_equal_error(far, frr):
    idx = np.argwhere(np.diff(np.sign(far - frr)) != 0).reshape(-1) + 0
    eer = np.mean(far[idx])
    return eer, int(np.mean(idx))


def get_user_classes(train_dir, test_dir):
    c_test = []
    for c in os.listdir(test_dir):
        if os.path.isdir(os.path.join(test_dir, c)):
            c_test.append(c)

    c_train = []
    for c in os.listdir(train_dir):
        if os.path.isdir(os.path.join(train_dir, c)):
            c_train.append(c)

    classes = []
    for c in c_test:
        if c in c_train:
            classes.append(c)

    return classes


# Define globals
data_dir = 'db/features/Facenet/'

fig = plt.figure()
ax = fig.add_subplot(111, title='ROC Curves - Facenet')

# Plot reference line
ax.plot(range(100), range(100), ':', c='gray')

for DATASET in os.listdir(data_dir):
    if not os.path.isdir(data_dir + '/' + DATASET):
        continue

    train_dir = data_dir + '/' + DATASET + '/' + 'Ref'
    test_dir = data_dir + '/' + DATASET + '/' + 'Questioned'

    NOR_OF_STEPS = 100.0
    steps = range(0, int(NOR_OF_STEPS), 1)
    classes = get_user_classes(train_dir, test_dir)

    print('Dataset:', DATASET)
    print('No. of users:', len(classes))

    # Output predictions to a csv file
    outfile = open(test_dir + '/predictions.csv', 'w')

    # Write header row
    outfile.write('Threshold, FAR, FRR\n')

    _FAR = []
    _FRR = []
    for j in steps:
        T = j / NOR_OF_STEPS
        FA = 0
        FR = 0
        NOR_F = 0
        NOR_G = 0
        x_train = []
        x_test = []
        files = []
        Y_true = []
        for c in classes:
            DIR_TRAIN = train_dir + '/' + c
            DIR_TEST = test_dir + '/' + c

            # Load training data
            x_train += read_train_data(DIR_TRAIN)

            # Load test data
            _files, _x_test, _Y_true = read_test_data(DIR_TEST)
            files += _files
            x_test += _x_test
            Y_true += _Y_true

            NOR_F += len([_ for _ in Y_true if _ == -1])
            NOR_G += len([_ for _ in Y_true if _ == 1])

        # Get predictions from OneClassSVM
        Y_test, Y_train, n_error_train, Y_prob = classifiers.OneClassSVM(x_train, x_test)

        for i in range(len(Y_prob)):
            tfile = files[i]
            prob = Y_prob[i][0]
            if prob > T:
                pred = 1
                if Y_true[i] == -1:
                    FA += 1
            else:
                pred = -1
                if Y_true[i] == 1:
                    FR += 1

        FAR = 100. * FA / NOR_F
        FRR = 100. * FR / NOR_G

        _FAR.append(FAR)
        _FRR.append(FRR)
        outfile.write(str(T) + ',' + str(FAR) + ',' + str(FRR) + '\n')

    FAR = np.array(_FAR)
    FRR = np.array(_FRR)

    # Calculate Equal Error Rate (EER)
    EER, idx = get_equal_error(FAR, FRR)
    print("EER:", str(round(EER, 3)) + '%')

    # Plot actual data
    ax.plot(FAR, FRR, '-', label=DATASET + '(' + str(round(EER, 3)) + '%)')

    # Plot EER
    ax.plot(EER, EER, '.', c='black')

# Show the graph
plt.legend()
plt.xlim([0, 100])
plt.ylim([0, 100])
plt.savefig(data_dir + '/figure.png')
plt.show()

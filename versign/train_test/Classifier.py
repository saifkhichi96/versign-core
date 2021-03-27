import numpy as np
from sklearn import svm


class Classifier:
    def __init__(self, nu=0.1, gamma=2**-11):
        self.clf = svm.OneClassSVM(nu=nu, kernel="rbf", gamma=gamma)

    def fit(self, x):
        self.clf.fit(x)

    def predict(self, x):
        return self.clf.predict(x)

    def decision_function(self, x):
        y = np.array(self.clf.decision_function(x))
        return (y-np.min(y)) / (np.max(y)-np.min(y))

    def fit_and_predict(self, x_train, x_test):
        self.fit(x_train)

        y_train = self.predict(x_train)
        train_error = y_train[y_train == -1].size / len(x_train)

        y_test = self.predict(x_test)
        y_prob = self.decision_function(x_test)

        return y_test, y_prob, train_error

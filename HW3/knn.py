# Starting code for UVA CS 4501 Machine Learning- KNN

__author__ = '**'
import numpy as np
np.random.seed(37)
# for plot
import matplotlib.pyplot as plt
#more imports
from sklearn.neighbors import KNeighborsClassifier
## the only purpose of the above import is in case that you want to compare your knn with sklearn knn



# Load file into np arrays
# x is the features
# y is the labels
def read_file(file):
    data = np.loadtxt(file, skiprows=1)
    np.random.shuffle(data)
    x = data[:, :-1]
    y = data[:, -1].astype(int)
    return x, y

# 2. Generate the i-th fold of k fold validation
# Input:
# x is an np array for training data
# y is an np array for labels
# i is an int indicating current fold
# nfolds is the total number of cross validation folds
def fold(x, y, i, nfolds):
    # your code
    fold_size = int(len(x) / nfolds)

    x_train = np.concatenate([
        x[:i * fold_size],
        x[(i + 1) * fold_size:]
    ])
    y_train = np.concatenate([
        y[:i * fold_size],
        y[(i + 1) * fold_size:]
    ])

    x_test = x[i * fold_size:(i + 1) * fold_size]
    y_test = y[i * fold_size:(i + 1) * fold_size]

    return x_train, y_train, x_test, y_test

# 3. Classify each testing points based on the training points
# Input
# x_train: a numpy array of training data
# x_test: a numpy array
# k: the number of neighbors to take into account when predicting the label
# Output
# y_predict: a numpy array
def classify(x_train, y_train, x_test, k, use_sk=False):
    # your code
    # Euclidean distance as the measurement of distance in KNN

    # if use sklearn to verify the results
    if use_sk:
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(x_train, y_train)
        y_predict = neigh.predict(x_test)
        return y_predict

    y_predict = np.zeros(x_test.shape[0])
    for i, x_target in enumerate(x_test):
        distances = np.linalg.norm(x_train - x_target, axis=1)
        indices = np.argpartition(distances, kth=k)[:k]
        votes = y_train[indices]
        label = 1 if votes.sum() * 2 > k else 0
        y_predict[i] = label

    return y_predict

# 4. Calculate accuracy by comaring with true labels
# Input
# y_predict is a numpy array of 1s and 0s for the class prediction
# y is a numpy array of 1s and 0s for the true class label
def calc_accuracy(y_predict, y):
    # your code
    match = (y_predict == y).sum()
    acc = match / len(y)
    return acc

# 5. Draw the bar plot of k vs. accuracy
# klist: a list of values of ks
# accuracy_list: a list of accuracies
def barplot(klist, accuracy_list):
    # your code
    # use matplot lib to generate bar plot with K on x axis and cross validation accuracy on y-axis
    plt.bar(klist, accuracy_list)
    plt.xticks(klist)
    plt.show()
    return

# 1. Find the best K
def findBestK(x, y, klist, nfolds):
    kbest = 0
    best_acc = 0
    accuracy_list = []
    for k in klist:
        # your code here
        # to get nfolds cross validation accuracy for k neighbors
        # implement fold(x, y, i, nfolds),classify(x_train, y_train, x_test, k) and calc_accuracy(y_predict, y)
        fold_accuracy = []
        for i in range(nfolds):
            x_train, y_train, x_test, y_test = fold(x, y, i, nfolds)
            y_predict = classify(x_train, y_train, x_test, k)

            fold_accuracy.append(calc_accuracy(y_predict, y_test))

        accuracy = sum(fold_accuracy) / nfolds # CROSS VALIDATION accuracy for k neighbors
        if accuracy > best_acc:
            kbest = k
            best_acc = accuracy
        accuracy_list.append(accuracy)
        print(k, accuracy)
    # plot cross validation error for each k : implement function barplot(klist, accuracy_list)
    barplot(klist, accuracy_list)
    return kbest


if __name__ == '__main__':
    filename = 'Movie_Review_Data.txt'
    # read data
    x, y = read_file(filename)
    nfolds = 4
    klist = [3, 5, 7, 9, 11, 13]
    # Implementation covers two tasks, both part of findBestK function
    # Task 1 : implement kNN classifier for a given x,y,k
    # Task 2 : implement 4 fold cross validation to select best k from klist

    bestk = findBestK(x, y, klist, nfolds)
    # report best k, and accuracy, discuss why some k work better than others
    print('Best K is', bestk)
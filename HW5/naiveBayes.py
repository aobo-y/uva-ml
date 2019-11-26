#!/usr/bin/python

import sys
import os
import numpy as np
from sklearn.naive_bayes import MultinomialNB

###############################################################################

VOC_PATH = './dictionary.txt'

def transfer(fileDj, vocabulary):
    BOWDj = [0] * len(vocabulary)
    with open(fileDj) as f:
        text = f.read()

    for line in text.split('\n'):
        for word in line.split(' '):
            if word in ['loved', 'loves', 'loving']:
                word = 'love'

            if word in vocabulary:
                BOWDj[vocabulary[word]] += 1

    BOWDj = np.array(BOWDj)
    return BOWDj


def loadData(path):
    with open(VOC_PATH) as f:
        content = f.read()
        voc = [w for w in content.split('\n') if w]

    # ignore last UNK token
    voc = voc[:-1]

    voc = {w: idx for idx, w in enumerate(voc)}

    def load_set(set_path):
        bows, labels = [], []
        for l_name in ['pos', 'neg']:
            f_path = os.path.join(set_path, l_name)
            filenames = [fn for fn in os.listdir(f_path)]
            bows += [transfer(os.path.join(f_path, fn), voc) for fn in filenames]
            labels += [1 if l_name == 'pos' else 0] * len(filenames)

        return np.array(bows), np.array(labels)

    xtrain, ytrain = load_set(os.path.join(path, 'training_set'))
    xtest, ytest = load_set(os.path.join(path, 'test_set'))

    return xtrain, xtest, ytrain, ytest


def naiveBayesMulFeature_train(xtrain, ytrain):
    alpha = 1

    labels = [0, 1]
    voc_size = xtrain.shape[1]
    label_accum = {l: np.zeros(voc_size) for l in labels}

    for x_vct, y in zip(xtrain, ytrain):
        label_accum[y] += x_vct


    thetaNeg, thetaPos = (
        (label_accum[l] + alpha) / (label_accum[l].sum() + voc_size * alpha)
        for l in labels
    )

    return thetaPos, thetaNeg


def naiveBayesMulFeature_test(xtest, ytest,thetaPos, thetaNeg):
    yPredict = []

    theta_logs = [np.log(theta) for theta in (thetaPos, thetaNeg)]

    for x_vct in xtest:
        pos_log, neg_log = [(x_vct * theta_log).sum() for theta_log in theta_logs]

        yPredict.append(1 if pos_log > neg_log else 0)

    yPredict = np.array(yPredict)

    accuracy = (yPredict == ytest).sum() / len(ytest)

    return yPredict, accuracy


def naiveBayesMulFeature_sk_MNBC(xtrain, ytrain, xtest, ytest):
    classifier = MultinomialNB(alpha=1.)
    classifier.fit(xtrain, ytrain)
    accuracy = classifier.score(xtest, ytest)
    return accuracy


def naiveBayesBernFeature_train(xtrain, ytrain):

    return thetaPosTrue, thetaNegTrue


def naiveBayesBernFeature_test(xtest, ytest, thetaPosTrue, thetaNegTrue):
    yPredict = []

    return yPredict, accuracy


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python naiveBayes.py dataSetPath")
        sys.exit()

    print("--------------------")
    textDataSetsDirectoryFullPath = sys.argv[1]

    xtrain, xtest, ytrain, ytest = loadData(textDataSetsDirectoryFullPath)

    thetaPos, thetaNeg = naiveBayesMulFeature_train(xtrain, ytrain)
    print("thetaPos =", thetaPos)
    print("thetaNeg =", thetaNeg)
    print("--------------------")


    yPredict, accuracy = naiveBayesMulFeature_test(xtest, ytest, thetaPos, thetaNeg)
    print("MNBC classification accuracy =", accuracy)

    accuracy_sk = naiveBayesMulFeature_sk_MNBC(xtrain, ytrain, xtest, ytest)
    print("Sklearn MultinomialNB accuracy =", accuracy_sk)


    thetaPosTrue, thetaNegTrue = naiveBayesBernFeature_train(xtrain, ytrain)
    print("thetaPosTrue =", thetaPosTrue)
    print("thetaNegTrue =", thetaNegTrue)
    print("--------------------")

    yPredict, accuracy = naiveBayesBernFeature_test(xtest, ytest, thetaPosTrue, thetaNegTrue)
    print("BNBC classification accuracy =", accuracy)
    print("--------------------")

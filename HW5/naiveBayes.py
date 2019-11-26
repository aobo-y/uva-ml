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

    voc = {w: idx for idx, w in enumerate(voc)}
    # ignore last UNK token
    voc = voc[:-1]

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

    return thetaPos, thetaNeg


def naiveBayesMulFeature_test(xtest, ytest,thetaPos, thetaNeg):
    yPredict = []

    return yPredict, Accuracy


def naiveBayesMulFeature_sk_MNBC(xtrain, ytrain, xtest, ytest):

    return Accuracy




def naiveBayesBernFeature_train(xtrain, ytrain):

    return thetaPosTrue, thetaNegTrue


def naiveBayesBernFeature_test(xtest, ytest, thetaPosTrue, thetaNegTrue):
    yPredict = []

    return yPredict, Accuracy


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

    exit()

    yPredict, Accuracy = naiveBayesMulFeature_test(xtest, ytest, thetaPos, thetaNeg)
    print("MNBC classification accuracy =", Accuracy)

    Accuracy_sk = naiveBayesMulFeature_sk_MNBC(xtrain, ytrain, xtest, ytest)
    print("Sklearn MultinomialNB accuracy =", Accuracy_sk)


    thetaPosTrue, thetaNegTrue = naiveBayesBernFeature_train(xtrain, ytrain)
    print("thetaPosTrue =", thetaPosTrue)
    print("thetaNegTrue =", thetaNegTrue)
    print("--------------------")

    yPredict, Accuracy = naiveBayesBernFeature_test(xtest, ytest, thetaPosTrue, thetaNegTrue)
    print("BNBC classification accuracy =", Accuracy)
    print("--------------------")

#!/usr/bin/python

import sys
import os
import collections
import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import nltk
from nltk.corpus import stopwords

###############################################################################

DIR_PATH = os.path.dirname(__file__)
VOC_PATH = os.path.join(DIR_PATH, './dictionary.txt')

stemmer = nltk.stem.PorterStemmer()

def transfer(fileDj, vocabulary, choice=1):
    BOWDj = [0] * len(vocabulary)
    with open(fileDj) as f:
        text = f.read()

    for line in text.split('\n'):
        for word in line.split(' '):
            if choice == 1:
                if word in ['loved', 'loves', 'loving']:
                    word = 'love'
            else:
                word = stemmer.stem(word)

            if word in vocabulary:
                BOWDj[vocabulary[word]] += 1
            elif choice == 2:
                BOWDj[vocabulary['UNK']] += 1

    BOWDj = np.array(BOWDj)
    return BOWDj


def build_voc(train_path):
    counts = collections.Counter()

    min_count = 3
    stop_set = set(stopwords.words('english'))

    filepaths = []
    for l_name in ['pos', 'neg']:
        f_path = os.path.join(train_path, l_name)
        filepaths += [os.path.join(f_path, fn) for fn in os.listdir(f_path)]

    for fp in filepaths:
        with open(fp) as f:
            text = f.read()

        words = nltk.word_tokenize(text)
        for word in words:
            word = stemmer.stem(word)
            if word not in stop_set:
                counts[word] += 1

    word_list = [w for w, c in counts.items() if c > 3]
    word_list.append('UNK')

    print('Voc size:', len(word_list))
    voc = {w: i for i, w in enumerate(word_list)}

    return voc


def loadData(path, choice=1):
    with open(VOC_PATH) as f:
        content = f.read()
        voc = [w for w in content.split('\n') if w]

    if choice == 1:
        # ignore last UNK token
        voc = voc[:-1]
        voc = {w: idx for idx, w in enumerate(voc)}
    else:
        voc = build_voc(os.path.join(path, 'training_set'))

    def load_set(set_path):
        bows, labels = [], []
        for l_name in ['pos', 'neg']:
            f_path = os.path.join(set_path, l_name)
            filenames = [fn for fn in os.listdir(f_path)]
            bows += [transfer(os.path.join(f_path, fn), voc, choice) for fn in filenames]
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
    alpha = 1

    labels = [0, 1]
    voc_size = xtrain.shape[1]

    label_counts = {l: 0 for l in labels}
    label_doc_counts = {l: np.zeros(voc_size) for l in labels}


    for x_vct, y in zip(xtrain, ytrain):
        label_counts[y] += 1
        label_doc_counts[y] += (x_vct > 0).astype(int)


    thetaNegTrue, thetaPosTrue = (
        (label_doc_counts[l] + alpha) / (label_counts[l] + 2 * alpha)
        for l in labels
    )
    return thetaPosTrue, thetaNegTrue


def naiveBayesBernFeature_test(xtest, ytest, thetaPosTrue, thetaNegTrue):
    yPredict = []

    for x_vct in (xtest > 0):
        pos_log, neg_log = [
            np.log(thetaTrue[x_vct]).sum() + np.log(1 - thetaTrue[~x_vct]).sum()
            for thetaTrue in [thetaPosTrue, thetaNegTrue]
        ]

        yPredict.append(1 if pos_log > neg_log else 0)

    yPredict = np.array(yPredict)

    accuracy = (yPredict == ytest).sum() / len(ytest)

    return yPredict, accuracy

def naiveBayesBernFeature_sk_BNBC(xtrain, ytrain, xtest, ytest):
    classifier = BernoulliNB(alpha=1.)
    classifier.fit(xtrain, ytrain)
    accuracy = classifier.score(xtest, ytest)
    return accuracy

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

    # accuracy_sk = naiveBayesBernFeature_sk_BNBC(xtrain, ytrain, xtest, ytest)
    # print("Sklearn BernoulliNB accuracy =", accuracy_sk)

#!/usr/bin/python

import sys
#Your code here
import random
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.mixture import GaussianMixture

def loadData(fileDj):
    #Your code here
    with open(fileDj) as f:
        data = [
            l.split(' ') for l in
            f.read().split('\n')
            if l
        ]

    data = np.array(data, dtype='float')
    labels = data[:, -1].astype('int')
    data = data[:, :-1]
    return data, labels

## K-means functions

def getInitialCentroids(X, k):
    #Your code here
    # n_fea = X.shape[1]
    # f_max, f_min = X.max(1), X.min(1)

    # initial_centroids = [
    #     [random.uniform(f_min[f_idx], f_max[f_idx]) for f_idx in range(n_fea)]
    #     for _ in range(k)
    # ]
    # initial_centroids = np.array(initial_centroids)
    initial_idxes = np.random.choice(range(X.shape[0]), k)
    if k == 1:
        initial_idxes = [initial_idxes]
    initial_centroids = X[initial_idxes]

    return initial_centroids

def getDistance(pt1, pt2):
    #Your code here
    dist = np.linalg.norm(pt1 - pt2)
    return dist

def allocatePoints(X, cluster_centroids):
    #Your code here
    point_distances = [
        [getDistance(point, centroid) for centroid in cluster_centroids]
        for point in X
    ]

    point_clusters = np.array(point_distances).argmin(1)
    return point_clusters

def updateCentroids(X, point_clusters):
    #Your code here
    k = point_clusters.max()

    return np.array([
        X[point_clusters == c].mean(axis=0)
        for c in range(k + 1)
    ])


def visualizeClusters(X, point_clusters):
    #Your code here
    colors = cm.get_cmap('Pastel1').colors
    k = point_clusters.max()
    for c, color in zip(range(k + 1), colors):
        c_idxes = (point_clusters == c)
        plt.scatter(X[c_idxes, 0], X[c_idxes, 1], color=color)

    plt.title('k-means')
    plt.show()

def kmeans(X, k, max_iter=1000):
    cluster_centroids = getInitialCentroids(X,k)
    point_clusters = None

    for i in range(max_iter):
        new_point_clusters = allocatePoints(X, cluster_centroids)
        # no point change membership, exit
        if (new_point_clusters == point_clusters).all():
            break

        point_clusters = new_point_clusters
        cluster_centroids = updateCentroids(X, point_clusters)

        # if (i + 1) % 1 == 0:
        #     print(f'iteration {i + 1}:', {i: (point_clusters == i).sum() for i in range(k)})

    return point_clusters


def kneeFinding(X, kList):
    #Your code here
    objs = []
    for k in kList:
        point_clusters = kmeans(X, k)
        cluster_centroids = updateCentroids(X, point_clusters)
        obj = sum([
            getDistance(point, cluster_centroids[cluster])
            for point, cluster in zip(X, point_clusters)
        ])
        objs.append(obj)

    plt.plot(kList, objs)
    plt.title('k knee finding')
    plt.show()

def purity(clusters, labels):
    purities = []
    #Your code here
    k = clusters.max()

    for c in range(k + 1):
        idxes = (clusters == c)
        cluster_labels = labels[idxes]
        label_counts = collections.Counter(cluster_labels)
        max_label, max_count = label_counts.most_common(1)[0]
        purity = max_count / len(cluster_labels)
        purities.append(purity)

    return purities


## GMM functions

#calculate the initial covariance matrix
#cov_type: diag, full
def getInitialsGMM(X, k, cov_type):
    if cov_type == 'full':
        dataArray = np.transpose(X)
        covMat = np.cov(dataArray)
    else:
        covMatList = [
            np.cov(X[:, i]).item()
            for i in range(X.shape[1])
        ]
        covMat = np.diag(covMatList)

    #Your code here
    cluster_portions = np.array([1 / k] * k)
    centroids = getInitialCentroids(X, k)
    return centroids, covMat, cluster_portions


def calcLogLikelihood(X_probs):
    loglikelihood = np.log(X_probs).sum()
    return loglikelihood

# E-step
def updateEStep(X, means, cov_mat, cluster_portions):
    cov_mat_inv = np.linalg.inv(cov_mat)

    E_matrix = np.array([[
        np.exp(-0.5 * (point - mu).T @ cov_mat_inv @ (point - mu)) * cp
        for mu, cp in zip(means, cluster_portions)
    ] for point in X])

    X_probs = E_matrix.sum(1)
    E_matrix /= X_probs[:, np.newaxis]

    return E_matrix, X_probs

# M-step
def updateMStep(X, E_matrix):
    #Your code here

    # centroids = np.array([
    #     (X * E_matrix[:, c:c+1]).sum(0) / E_matrix[:, c].sum()
    #     for c in range(E_matrix.shape[1])
    # ])å

    means = E_matrix.T @ X / E_matrix.sum(0)[:, np.newaxis]

    cluster_portions = E_matrix.sum(0) / E_matrix.shape[0]

    return means, cluster_portions

def visualizeClustersGMM(X, labels, cov_type):
    #Your code here
    colors = cm.get_cmap('Pastel1').colors

    k = labels.max()
    for c, color in zip(range(k + 1), colors):
        c_idxes = (labels == c)
        plt.scatter(X[c_idxes, 0], X[c_idxes, 1], color=color)

    plt.title('GMM ' + cov_type)
    plt.show()

def gmmCluster(X, k, cov_type, max_iter=1000):
    #initial clusters
    clustersGMM = getInitialsGMM(X, k, cov_type)
    centroids, cov_mat, cluster_portions = clustersGMM

    E_matrix = None
    ll = -float('inf')

    #Your code here
    for i in range(max_iter):
        new_E_matrix, X_probs = updateEStep(X, centroids, cov_mat, cluster_portions)

        new_ll = calcLogLikelihood(X_probs)

        # stop when loglikelihood increases less than a threshold
        if new_ll - ll < 0.00001:
            break

        E_matrix = new_E_matrix
        ll = new_ll

        centroids, cluster_portions = updateMStep(X, E_matrix)

        if (i + 1) % 100 == 0:
            print(f'iteration {i + 1}:', ll)

    labels = E_matrix.argmax(1)

    visualizeClustersGMM(X, labels, cov_type)
    return labels



def main():
    # dataset path
    datadir = sys.argv[1]
    pathDataset1 = datadir + '/humanData.txt'
    pathDataset2 = datadir + '/audioData.txt'
    dataset1, labels1 = loadData(pathDataset1)
    dataset2, labels2 = loadData(pathDataset2)

    #Q2,Q3
    point_clusters = kmeans(dataset1, 2, max_iter=1000)

    visualizeClusters(dataset1, point_clusters)

    point_clusters = kmeans(dataset2, 2, max_iter=1000)
    visualizeClusters(dataset2, point_clusters)

    model = GaussianMixture(2, covariance_type='tied')
    point_clusters = model.fit_predict(dataset1)
    visualizeClusters(dataset1, point_clusters)

    #Q4
    kneeFinding(dataset1,range(1,7))

    #Q5
    purities = purity(point_clusters, labels1)
    print('k-mean purities:', purities)

    #Q7
    preds11 = gmmCluster(dataset1, 2, 'diag')
    preds12 = gmmCluster(dataset1, 2, 'full')

    #Q8
    preds21 = gmmCluster(dataset2, 2, 'diag')
    preds22 = gmmCluster(dataset2, 2, 'full')

    #Q9
    purities11 = purity(preds11, labels1)
    print('diag GMM purities of dataset1:', purities11)
    purities12 = purity(preds12, labels1)
    print('full GMM purities of dataset1:', purities12)

    purities21 = purity(preds21, labels2)
    print('diag GMM purities of dataset2:', purities21)

    purities22 = purity(preds22, labels2)
    print('full GMM purities of dataset2:', purities22)

if __name__ == "__main__":
    main()

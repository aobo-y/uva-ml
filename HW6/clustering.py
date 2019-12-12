#!/usr/bin/python

import sys
#Your code here
import random
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
#covType: diag, full
def getInitialsGMM(X,k,covType):
    if covType == 'full':
        dataArray = np.transpose(np.array([pt[0:-1] for pt in X]))
        covMat = np.cov(dataArray)
    else:
        covMatList = []
        for i in range(len(X[0])-1):
            data = [pt[i] for pt in X]
            cov = np.asscalar(np.cov(data))
            covMatList.append(cov)
        covMat = np.diag(covMatList)

    initialClusters = {}
    #Your code here
    return initialClusters


def calcLogLikelihood(X,clusters,k):
    loglikelihood = 0
    #Your code here
    return loglikelihood

#E-step
def updateEStep(X,clusters,k):
    EMatrix = []
    #Your code here
    return EMatrix

#M-step
def updateMStep(X,clusters,EMatrix):
    #Your code here
    return clusters

def visualizeClustersGMM(X,labels,clusters,covType):
    #Your code here
    pass

def gmmCluster(X, k, covType, maxIter=1000):
    #initial clusters
    clustersGMM = getInitialsGMM(X,k,covType)
    labels = []
    #Your code here
    visualizeClustersGMM(X,labels,clustersGMM,covType)
    return labels,clustersGMM


def purityGMM(X, clusters, labels):
    purities = []
    #Your code here
    return purities




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

    #Q4
    kneeFinding(dataset1,range(1,7))

    #Q5
    purities = purity(point_clusters, labels1)
    print('k-mean purities are:', purities)
    exit()

    #Q7
    labels11,clustersGMM11 = gmmCluster(dataset1, 2, 'diag')
    labels12,clustersGMM12 = gmmCluster(dataset1, 2, 'full')

    #Q8
    labels21,clustersGMM21 = gmmCluster(dataset2, 2, 'diag')
    labels22,clustersGMM22 = gmmCluster(dataset2, 2, 'full')

    #Q9
    purities11 = purityGMM(dataset1, clustersGMM11, labels11)
    purities12 = purityGMM(dataset1, clustersGMM12, labels12)
    purities21 = purityGMM(dataset2, clustersGMM21, labels21)
    purities22 = purityGMM(dataset2, clustersGMM22, labels22)

if __name__ == "__main__":
    main()

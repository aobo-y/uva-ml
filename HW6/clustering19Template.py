#!/usr/bin/python

import sys
#Your code here

def loadData(fileDj):
    data = []
    #Your code here
    return data

## K-means functions 

def getInitialCentroids(X, k):
    initialCentroids = {}
    #Your code here
    return initialCentroids

def getDistance(pt1,pt2):
    dist = 0
    #Your code here
    return dist

def allocatePoints(X,clusters):
    #Your code here
    return clusters

def updateCentroids(clusters):
    #Your code here
    return clusters


def visualizeClusters(clusters):
    #Your code here


def kmeans(X, k, maxIter=1000):
    cluster_centroids = getInitialCentroids(X,k)
    clusters = allocatePoints(X,cluster_centroids)
    cluster_centroids = updateCentroids(clusters)
    #Your code here
    return clusters


def kneeFinding(X,kList):
    #Your code here

def purity(X, clusters):
    purities = []
    #Your code here
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
    #######dataset path
    datadir = sys.argv[1]
    pathDataset1 = datadir+'/humanData.txt'
    pathDataset2 = datadir+'/audioData.txt'
    dataset1 = loadData(pathDataset1)
    dataset2 = loadData(pathDataset2)
    #Q2,Q3
    clusters = kmeans(dataset1, 2, maxIter=1000)
    visualizeClusters(clusters)
    #Q4
    kneeFinding(dataset1,range(1,7))

    #Q5
    
    purity(dataset1,clusters)

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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 10:26:24 2022

@author: zhiyuas
"""
import math
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X,y = datasets.load_iris(return_X_y=True);X = X[:,0:2]
def dist(point1,point2):
    return math.sqrt(np.sum([(p1-p2)**2 for p1, p2 in zip(point1,point2)]))
def dist_ab(point1,point2):
    return np.sum([abs(p1-p2) for p1, p2 in zip(point1,point2)])
def assign_cluster(data_coord, centroids):
    return np.argmin([dist(data_coord,centroids[i]) for i in range(len(centroids))])
def cal_cluster_centroid(old_clusters,kmedian_use=False):
    data_dim = len(old_clusters[0])
    if kmedian_use:
        return [np.median(np.array(old_clusters)[:,i]) for i in range(data_dim)]
    else:
        return [np.mean(np.array(old_clusters)[:,i]) for i in range(data_dim)]
def ob_fun(centroids,cluster_assigned,kmedian_use=False):
    dis_sum = 0
    if kmedian_use:
        for i in range(len(centroids)):
            if len(cluster_assigned[i]) == 0:
                continue
            dis_sum = dis_sum + np.sum([dist_ab(centroids[i],cluster_assigned[i][j]) for j in range(len(cluster_assigned[i]))])
        return dis_sum
    else:
        for i in range(len(centroids)):
            if len(cluster_assigned[i]) == 0:
                continue
            dis_sum = dis_sum + np.sum([dist(centroids[i],cluster_assigned[i][j]) for j in range(len(cluster_assigned[i]))])
        return dis_sum       
#initialize centroids, with 3 clusters
centroids = [X[1],X[2],X[3]]
n_cl = len(centroids)
step = 0 #interating step
kmedian_use=False #indicate which method to use
ob_scores = [0,99999]
#update centroids and iterating
while ob_scores[-1] != ob_scores[-2]:
    #A dictionary for assigned cluster mumbers
    cluster_assigned = dict([(i, []) for i in range(n_cl)])
    for row in X:
        cluster_assignment = assign_cluster(row, centroids)
        cluster_assigned[cluster_assignment].append(row.tolist())
    ob_scores.append(ob_fun(centroids,cluster_assigned,kmedian_use))
    centroids = [cal_cluster_centroid(cluster_assigned[i],kmedian_use) for i in range(n_cl)]
    step = step + 1
ob_scores = ob_scores[2:]

colors=['red','green','blue']
#plot datapoints with colored clusters
for i in range(n_cl):
    x_coord = [x[0] for x in cluster_assigned[i]]
    y_coord = [y[1] for y in cluster_assigned[i]]
    plt.scatter(x_coord,y_coord,color=colors[i])
    plt.scatter(centroids[i][0],centroids[i][1],color=colors[i],marker='x')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('iteration = '+str(step-1))
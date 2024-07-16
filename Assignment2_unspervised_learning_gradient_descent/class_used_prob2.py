#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 16:53:14 2022

@author: zhiyuas
"""
import math
import numpy as np
from sklearn import datasets
import random
import matplotlib.pyplot as plt

def dist(point1,point2):
    return math.sqrt(np.sum([(p1-p2)**2 for p1, p2 in zip(point1,point2)]))
def dist_ab(point1,point2):
    return np.sum([abs(p1-p2) for p1, p2 in zip(point1,point2)])
class K_M:
     class kmean():
         def __init__(self,n_k,tol,data):
             ob_scores = [0,9999] #objective fuction
             xy_lims = [[np.min(data[:,i]),np.max(data[:,i])] for i in range(len(data[0]))]
             #initialize centroids at the beginning
             centroids = [[random.uniform(j[0],j[1]) for j in xy_lims] for i in range(n_k)]
             iter_step = 0
             while abs(ob_scores[-1] -ob_scores[-2]) >= tol:
                 #initialize clusters every step
                 cluster_assigned = dict([(i, []) for i in range(n_k)])
                 for row in data:
                     cluster_assignment = np.argmin([dist(row,centroids[i]) for i in range(n_k)])
                     cluster_assigned[cluster_assignment].append(row.tolist())
                 dis_sum = 0
                 for i in range(n_k):
                     dis_sum = dis_sum + np.sum([dist(centroids[i],cluster_assigned[i][j]) \
                                     for j in range(len(cluster_assigned[i]))])
                 ob_scores.append(dis_sum)
                 centroids = [ [np.mean(np.array(cluster_assigned[j])[:,i])\
                               for i in range(len(cluster_assigned[j][0]))] for j in range(n_k)]
                 iter_step = iter_step + 1
                 self.ob_scores = ob_scores;self.centroids = centroids
                 self.step = iter_step;self.clusters = cluster_assigned
     class kmedian():
         def __init__(self,n_k,tol,data):
             ob_scores = [0,9999] #objective fuction
             xy_lims = [[np.min(data[:,i]),np.max(data[:,i])] for i in range(len(data[0]))]
             #initialize centroids at the beginning
             centroids = [[random.uniform(j[0],j[1]) for j in xy_lims] for i in range(n_k)]
             iter_step = 0
             while abs(ob_scores[-1] -ob_scores[-2]) >= tol:
                 cluster_assigned = dict([(i, []) for i in range(n_k)])
                 for row in data:
                     cluster_assignment = np.argmin([dist(row,centroids[i]) for i in range(n_k)])
                     cluster_assigned[cluster_assignment].append(row.tolist())
                 dis_sum = 0
                 for i in range(n_k):
                     dis_sum = dis_sum + np.sum([dist_ab(centroids[i],cluster_assigned[i][j])\
                           for j in range(len(cluster_assigned[i]))])
                 ob_scores.append(dis_sum)
                 centroids = [ [np.median(np.array(cluster_assigned[j])[:,i])\
                         for i in range(len(cluster_assigned[j][0]))] for j in range(n_k)]
                 iter_step = iter_step + 1
                 self.ob_scores = ob_scores;self.centroids = centroids
                 self.step = iter_step;self.clusters = cluster_assigned
X,y = datasets.load_iris(return_X_y=True);X = X[:,:2]
points_added = np.array([[6.181,4.016],[6.34,3.714],[6.987,1.994],[7.415,2.126]])
X_add = []
X_add.append(X)
X_add.append(np.concatenate((X,points_added),axis=0))
random.seed(2022)
colors = ['r','green','blue']
fig,axes = plt.subplots(figsize=(12,12),nrows=2, ncols=2)

tol = 1e-5; n_k = 3
for ax,data in zip(axes,X_add):
    A = K_M.kmean(n_k, tol, data);B = K_M.kmedian(n_k, tol, data)   
    for i in range(n_k):
        x_coord = [x[0] for x in A.clusters[i]];y_coord = [y[1] for y in A.clusters[i]]
        ax[0].scatter(x_coord,y_coord,color=colors[i])
        ax[0].scatter(A.centroids[i][0],A.centroids[i][1],color='cyan',marker='x')
        x_coord = [x[0] for x in B.clusters[i]];y_coord = [y[1] for y in B.clusters[i]]    
        ax[1].scatter(x_coord,y_coord,color=colors[i])
        ax[1].scatter(B.centroids[i][0],B.centroids[i][1],color='cyan',marker='x')
    ax[0].set_title('K-mean  iteration = '+str(A.step)+' ob_score = '\
                       +str('%.1f'%A.ob_scores[-1]),fontweight='bold')
    ax[1].set_title('K-median  iteration = '+str(B.step)+' ob_score = '\
                      +str('%.1f'%B.ob_scores[-1]),fontweight='bold')

for i in range(2):
        axes[1][i].set_xlabel('X')
        axes[i][0].set_ylabel('Y')
plt.savefig('prob2.png')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 13:28:02 2022

@author: zhiyuas
"""

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn.cluster import KMeans
import time
import scipy.stats as stats
font_title = 20
font_label = 16
data_raw = np.loadtxt('features.txt')[:,[0,3]]
data = stats.zscore(data_raw,axis=0)
a,b,c = np.histogram2d(data_raw[:,0],data_raw[:,1])
heatmap = plt.contour(a.transpose(),extent=[b[0],b[-1],c[0],c[-1]],cmap= cm.cool)

n_k = 2
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',\
          '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
#clustering = KMedoids(n_clusters=n_k,method='pam',init='k-medoids++').fit(data_raw)
t0 = time.time()
clustering = KMeans(n_clusters=n_k,init='k-means++',random_state=0).fit(data)
t1 = time.time()
clusters = dict([(i, []) for i in range(n_k)])
for i in range(len(data_raw)):
    clusters[clustering.labels_[i]].append(data_raw[i].tolist())
for i in range(n_k):
    x_coord = [x[0] for x in clusters[i]];y_coord = [y[1] for y in clusters[i]]
    #find centoids
    pairwise_dists = [np.sum(np.sum((np.array(clusters[i])-np.array(j))**2,axis=1)**0.5)\
                  for j in clusters[i]]
    sub_label = np.argmin(pairwise_dists)
    sub_center_coord = clusters[i][sub_label]
    centroid = [q.tolist() for q in data_raw].index(sub_center_coord)
    plt.scatter(x_coord,y_coord,color=colors[i])
    plt.scatter(data_raw[centroid,0],data_raw[centroid,1],color='k',marker='x')
    plt.text(data_raw[centroid,0],data_raw[centroid,1],s=str(centroid)+r'$^{th}$',\
             fontweight='bold',fontsize=font_label)
plt.xlabel('RMSD (Å)',fontsize=font_label)
plt.ylabel('Num. of β sheets',fontsize=font_label)
plt.clabel(heatmap, fontsize=9, inline=True)
plt.title('Time = %.2fs' %(t1-t0),fontsize=font_title)
plt.savefig('kmeans.png',dpi=220)
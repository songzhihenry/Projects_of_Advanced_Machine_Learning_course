#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 15:20:29 2022

@author: zhiyuas
"""

from scipy.cluster import hierarchy
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import time
#colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',\
#          '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
#data = np.loadtxt('all_atm_coords.txt')
ytdist = pdist(data)
n_k = 4
t0 = time.time()
Z = hierarchy.linkage(ytdist, 'single')
result = hierarchy.fcluster(Z, t=4, criterion='maxclust')
dn = hierarchy.dendrogram(Z,truncate_mode='lastp',count_sort='ascending')
t1 = time.time()
clusters = dict([(i, []) for i in range(n_k)])
for i in range(len(data)):
    clusters[result[i]-1].append(data[i].tolist())
for i in range(n_k):
    #find centoids
    cluster_points = np.array(clusters[i])
    sub_label = np.argmin([np.sum((point-np.mean(cluster_points,axis=0))**2) for point in cluster_points])
    sub_center_coord = clusters[i][sub_label]
    centroid = [q.tolist() for q in data].index(sub_center_coord)
    plt.text(0.9,0.8-i*0.1,s=str(centroid)+r'$^{th}$',color=colors[i],transform=plt.gca().transAxes)
    print (i)
plt.text(0.9,0.9,("%.2fs" % (t1 - t0)).lstrip("0"),transform=plt.gca().transAxes)
plt.savefig('hierarchy.png',dpi=220)
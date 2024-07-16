# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 20:30:46 2022

@author: Henry Song
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
import time
import scipy.stats as stats
font_title = 20
font_label = 16
data_raw = np.loadtxt('features.txt')
data = stats.zscore(data_raw,axis=0)
n_k = 4
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',\
          '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
t0 = time.time()
pca = PCA(n_components=2)
X_r = pca.fit(data).transform(data)

clustering = SpectralClustering(n_clusters=n_k,assign_labels='kmeans',\
                                affinity='nearest_neighbors',\
                                n_neighbors=6,random_state=0,n_jobs=-1).fit(X_r)
t1 = time.time()
clusters = dict([(i, []) for i in range(n_k)])
a,b,c = np.histogram2d(X_r[:,0],X_r[:,1])
heatmap = plt.contour(a.transpose(),extent=[b[0],b[-1],c[0],c[-1]],cmap= cm.cool)
for i in range(len(X_r)):
    clusters[clustering.labels_[i]].append(X_r[i].tolist())

for i in range(n_k):
    x_coord = [x[0] for x in clusters[i]];y_coord = [y[1] for y in clusters[i]]
    #find centoids
    pairwise_dists = [np.sum(np.sum((np.array(clusters[i])-np.array(j))**2,axis=1)**0.5)\
                  for j in clusters[i]]
    sub_label = np.argmin(pairwise_dists)
    sub_center_coord = clusters[i][sub_label]
    centroid = [q.tolist() for q in X_r].index(sub_center_coord)
    plt.scatter(x_coord,y_coord,color=colors[i])
    plt.scatter(X_r[centroid,0],X_r[centroid,1],color='k',marker='x')
    plt.text(X_r[centroid,0],X_r[centroid,1],s=str(centroid)+r'$^{th}$',fontweight='bold',fontsize=font_label)
plt.xlabel(r'1$^{st}$ component')
plt.ylabel(r'2$^{nd}$ component')
plt.clabel(heatmap, fontsize=9, inline=True)
plt.title('Time = %.2fs' %(t1-t0),fontsize=font_title)
plt.savefig('PCA_spectral_clustering.png',dpi=220)
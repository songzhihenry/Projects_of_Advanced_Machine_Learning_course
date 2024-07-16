# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 21:32:35 2022

@author: Henry Song
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
#import scipy.stats as stats
import time
font_title = 20
font_label = 16
data_raw = np.loadtxt('all_atm_coords.txt')
n_k = 4
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',\
          '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
t0 = time.time()
pca = PCA(n_components=2)
X_r = pca.fit(data_raw).transform(data_raw)
#data = stats.zscore(data_raw,axis=0)
#X_r = pca.fit(data).transform(data)
clustering = KMeans(n_clusters=n_k,init='k-means++',random_state=0).fit(X_r)
t1 = time.time()
a,b,c = np.histogram2d(X_r[:,0],X_r[:,1])
heatmap = plt.contour(a.transpose(),extent=[b[0],b[-1],c[0],c[-1]],cmap= cm.cool)
clusters = dict([(i, []) for i in range(n_k)])
for i in range(len(X_r)):
    clusters[clustering.labels_[i]].append(X_r[i].tolist())
centers = clustering.cluster_centers_
for i in range(n_k):
    x_coord = [x[0] for x in clusters[i]];y_coord = [y[1] for y in clusters[i]]
    centroid = np.argmin(np.sum((X_r-centers[i])**2,axis=1))
    plt.scatter(x_coord,y_coord,color=colors[i])
    plt.scatter(X_r[centroid,0],X_r[centroid,1],color='k',marker='x')
    plt.text(X_r[centroid,0],X_r[centroid,1],s=str(centroid)+r'$^{th}$',fontweight='bold',fontsize=font_label)
plt.xlabel(r'1$^{st}$ component')
plt.ylabel(r'2$^{nd}$ component')
plt.clabel(heatmap, fontsize=9, inline=True)
plt.title('Time = %.2fs' %(t1-t0),fontsize=font_title)
plt.savefig('PCA_kmeans_all_coords.png',dpi=220)
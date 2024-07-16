# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 15:12:32 2022

@author: Henry Song
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd
from numpy.linalg import inv
data_raw = np.array([[1,3],[2,5],[3,4],[4,3],[5,2],[5,1]])
data = data_raw - np.mean(data_raw,axis=0)
center = np.mean(data_raw,axis=0)
u,s,_ = svd(data.T.dot(data))
phi11 = u[0,0];phi21 = u[1,0]
line_x = np.linspace(np.min(data_raw[:,0]),np.max(data_raw[:,0]),10)#+center[0]
line_y = (line_x-center[0])*phi21/phi11+center[1]
plt.scatter(data_raw[:,0],data_raw[:,1])

cluster = {0:data_raw[:4],1:data_raw[4:]}
centroids = [np.mean(cluster[i],axis=0)[...,None] for i in range(2)]
Sb = (centroids[0]-centroids[1]).dot((centroids[0]-centroids[1]).T)
Sw = 0
for i in range(2):
    for j in range(len(cluster[i])):
        diff = cluster[i][j][...,None]-centroids[i]
        Sw = Sw + diff.dot(diff.T)
u2,s2,v2 = svd(inv(Sw).dot(Sb))
w = u2[:,0]
line_y1 = (line_x-center[0])*w[1]/w[0]+center[1]
plt.plot(line_x,line_y,label='PCA')
plt.plot(line_x,line_y1,label='LDA')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.legend(frameon=False)
plt.title('PCA vs LDA on Dimension Reduction')
plt.savefig('prob1.png')
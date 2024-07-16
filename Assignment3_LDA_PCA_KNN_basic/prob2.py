# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 18:05:06 2022

@author: Henry Song
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
data_set = np.array([[1,1],[2,2],[2,3],[3,2],[3,3],[4,4]])
y = np.array([0,0,0,1,1,1])
num = 50
x_range = np.linspace(np.min(data_set[:,0])-0.5, np.max(data_set[:,0])+0.5,num)
y_range = np.linspace(np.min(data_set[:,1])-0.5, np.max(data_set[:,1])+0.5,num)
X,Y = np.meshgrid(x_range,y_range)
K = [1,3]
labels = ['Positive','Negative']
fig, axes = plt.subplots(figsize=(13,6),nrows=1,ncols=2)
for k,ax in zip(K,axes):
    knn = neighbors.KNeighborsClassifier(k)
    knn.fit(data_set,y)
    Z = knn.predict(np.c_[X.ravel(),Y.ravel()])
    Z = Z.reshape(X.shape)
    ax.pcolormesh(X,Y,Z,cmap=cmap_light)
    ax.scatter(data_set[:3,0],data_set[:3,1],c='#FF0000',label='Positive')
    ax.scatter(data_set[3:,0],data_set[3:,1],c='#0000FF',label='Negative')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('k = '+str(k))
axes[1].legend(loc='upper left')
plt.savefig('prob2.png')
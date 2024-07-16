#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 14:27:34 2022

@author: zhiyuas
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import scipy.stats as stats
#data_raw = np.loadtxt('features.txt')
#data = stats.zscore(data_raw,axis=0)
data = np.loadtxt('all_atm_coords.txt')
pca = PCA(n_components=2)
X_r = pca.fit(data).transform(data)
a,b,c = np.histogram2d(X_r[:,0],X_r[:,1])
heatmap = plt.contourf(a.transpose(),extent=[b[0],b[-1],c[0],c[-1]])
cb = plt.colorbar(heatmap,fraction=0.1,shrink=0.7)
plt.xlabel(r'1$^{st}$ component')
plt.ylabel(r'2$^{nd}$ component')
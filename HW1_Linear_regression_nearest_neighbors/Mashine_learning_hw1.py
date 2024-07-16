#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 15:17:03 2022

@author: zhiyuas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
'''
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
df = pd.read_csv(url,header=None)
data = df.values
X,y = data[:,:-1], data[:,-1][...,None]#convert row vector to column

#(X^T*X+k*I)*X^T*y
#beta = np.linalg.inv(X.T.dot(X)+k*np.identity(len(X[0]))).dot(X.T).dot(y)

#||y-beta*X||**2+k*||X||**2
#result = np.dot(y-np.dot(beta,X),y-np.dot(beta,X))+k*np.dot(beta,beta)
#result = (y-X.dot(beta)).T.dot(y-X.dot(beta))+k*beta.T.dot(beta)
'''
def RR_np_p(A,k):
    #(A^T*A+k*I_p)^{-1}*A^T
    beta = np.linalg.inv(A.T.dot(A)+k*np.identity(len(A.T))).dot(A.T)
    #beta = ((A.T.dot(A)+k*np.identity(len(A.T))).i).dot(A.T)
    return beta
def RR_np_n(A,k):
    #A^T(A*A^T+k*I_n)^{-1}
    B = A.dot(A.T)+k*np.identity(len(A))
    beta = A.T.dot(np.linalg.inv(B))
    #beta = A.T.dot((A.dot(A.T)+k*np.identity(len(A))).i)
    return beta
n = 100;p = np.array([10,100,1000,2000])
k=1
dt1 = []
dt2 = []
exam = []
for i in range(len(p)):
    A = np.random.rand(n,p[i])
    start1 = time.time()
    a = RR_np_p(A,k)
    end1 = time.time()
    start2 = time.time()
    b = RR_np_n(A,k)
    end2 = time.time()
    dt1.append(end1 - start1)
    dt2.append(end2 - start2)
    exam.append(np.sum(a - b))
colors = ['#1f77b4', '#ff7f0e',]
plt.plot(p,dt1,label=r'$\beta = (A^{T}A+\lambda I_{p})^{-1}A^{T}$',color=colors[0])
plt.plot(p,dt2,label=r'$\beta = A^{T}(AA^{T}+\lambda I_{n})^{-1}$',color=colors[1])
plt.plot(p,dt1,'o',color=colors[0])
plt.plot(p,dt2,'o',color=colors[1])
plt.yscale('log')
plt.legend()
plt.xlabel('choice of p')
plt.ylabel('Execution time (s)')
plt.savefig('p1.png')
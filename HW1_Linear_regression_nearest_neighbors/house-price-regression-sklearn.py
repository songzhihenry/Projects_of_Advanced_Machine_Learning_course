#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 09:56:56 2022

@author: zhiyuas
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as sl
from sklearn.metrics import mean_squared_error
fontsize_label = 18
fontsize_ticks = 15
#prob1

#import data
data = np.loadtxt('housing.data')
#remove any possible ordering fx
seed = 2023; np.random.seed(seed)
randperm = np.random.randint(0,len(data),len(data))
data = data[randperm,:]
#'''
#prob2
#selection for first n records of the trainning data
n = [25,50,75,100,150,200,300]
def Vanilla_y_hat(A,y):
    #(A^T*A)^{-1}*A^T
    result = np.linalg.inv((A.T).dot(A)).dot(A.T)
    beta = result.dot(y)
    return beta
def MSE(y,beta,x):
    y_hat = x.dot(beta)
    #return sum([x**2 for x in (y-y_hat)])/len(y)
    return ((y-y_hat).T).dot(y-y_hat)[0][0]/len(y)
MSE_test = []
MSE_train = []
MSE_SL_train = []
MSE_SL_test = []
for Ntrain in n:
    #split the data into trainning and testing; then standarize data for trainning and testing,respectively
    Xtrain = stats.zscore(data[:Ntrain,:-1],axis=0); ytrain = data[:Ntrain,-1][...,None]
    Xtest = stats.zscore(data[Ntrain:,:-1],axis=0); ytest = data[Ntrain:,-1][...,None]
    Xtrain_mt = np.concatenate((np.ones((len(Xtrain),1)),Xtrain),axis=1)
    Xtest_mt = np.concatenate((np.ones((len(Xtest),1)),Xtest),axis=1)
    reg = sl.LinearRegression().fit(Xtrain, ytrain)
    beta = Vanilla_y_hat(Xtrain_mt, ytrain)
    MSE_train.append(MSE(ytrain,beta,Xtrain_mt))
    MSE_test.append(MSE(ytest,beta,Xtest_mt))
    MSE_SL_train.append(mean_squared_error(ytrain,reg.predict(Xtrain)))
    MSE_SL_test.append(mean_squared_error(ytest,reg.predict(Xtest)))
plt.plot(n,MSE_train,color='green',label='train',marker='D')
plt.plot(n,MSE_test,color='yellow',label='test',marker='D')
plt.plot(n,MSE_SL_train,color='blue',label='sklearn train')
plt.plot(n,MSE_SL_test,color='red',label='sklearn test')
plt.legend(frameon=False,fontsize=fontsize_label)
plt.ylabel('MSE',fontsize=fontsize_label)
plt.xlabel('Size of training set',fontsize=fontsize_label)
plt.savefig('4_2_fig.png')
#'''
'''
#prob3
Ntrain = 300
MSE_SL_test = []
MSE_SL_train = []
#split the data into trainning and testing; then standarize data for trainning and testing,respectively
Xtrain = stats.zscore(data[:Ntrain,:-1],axis=0); ytrain = data[:Ntrain,-1][...,None]
Xtest = stats.zscore(data[Ntrain:,:-1],axis=0); ytest = data[Ntrain:,-1][...,None]
reg = sl.LinearRegression().fit(Xtrain,ytrain)
MSE_SL_train.append(mean_squared_error(ytrain, reg.predict(Xtrain)))
MSE_SL_test.append(mean_squared_error(ytest,reg.predict(Xtest)))
degree = [2,3,4,5,6]
features = data[:,:-1]
for deg in degree:
    features = np.concatenate((features,data[:,:-1]**deg),axis=1)
    Xtrain = stats.zscore(features[:Ntrain,:],axis=0); ytrain = data[:Ntrain,-1][...,None]
    Xtest = stats.zscore(features[Ntrain:,:],axis=0); ytest = data[Ntrain:,-1][...,None]
    reg = sl.LinearRegression().fit(Xtrain,ytrain)
    MSE_SL_train.append(mean_squared_error(ytrain,reg.predict(Xtrain)))
    MSE_SL_test.append(mean_squared_error(ytest,reg.predict(Xtest)))
plt.plot([1]+degree,MSE_SL_train,color='blue',label='train',marker='D')
plt.plot([1]+degree,MSE_SL_test,color='red',label='test',marker='D')
plt.legend(frameon=False,fontsize=fontsize_label)
plt.ylabel('MSE',fontsize=fontsize_label)
plt.xlabel('Degree of expansion',fontsize=fontsize_label)
#plt.tick_params(axis='both',fontsize=fontsize_ticks)
plt.savefig('4_3_fig.png')
'''
'''
#prob4
Ntrain = 300
MSE_SL_test = []
MSE_SL_train = []
lamb = np.logspace(-10,10,10)
degree = [2,3,4,5,6]
features = data[:,:-1]
for deg in degree:
    features = np.concatenate((features,data[:,:-1]**deg),axis=1)
Xtrain = stats.zscore(features[:Ntrain,:],axis=0); ytrain = data[:Ntrain,-1][...,None]
Xtest = stats.zscore(features[Ntrain:,:],axis=0); ytest = data[Ntrain:,-1][...,None]
for lam in lamb:
    reg = sl.Ridge(lam).fit(Xtrain,ytrain)
    MSE_SL_train.append(mean_squared_error(ytrain, reg.predict(Xtrain)))
    MSE_SL_test.append(mean_squared_error(ytest,reg.predict(Xtest)))
plt.plot(lamb,MSE_SL_train,color='b',label='train',marker='D')
plt.plot(lamb,MSE_SL_test,color='r',label='test',marker='D')
plt.xscale('log')
plt.legend(frameon=False,fontsize=fontsize_label)
plt.ylabel('MSE',fontsize=fontsize_label)
plt.xlabel(r'$\lambda$',fontsize=fontsize_label)
#plt.tick_params(axis='both',fontsize=fontsize_ticks)
plt.savefig('4_4_fig.png')
'''
'''
#prob5
Ntrain = 300
lamb = np.logspace(-4,2)
#split the data into trainning and testing; then standarize data for trainning and testing,respectively
Xtrain = stats.zscore(data[:Ntrain,:-1],axis=0); ytrain = data[:Ntrain,-1][...,None]
beta_set = []
for lam in lamb:
    reg = sl.Lasso(alpha=lam).fit(Xtrain,ytrain)
    beta_set.append(reg.coef_)
beta_set = np.array(beta_set).transpose()
for i in range(len(beta_set)):
    plt.plot(lamb,beta_set[i],label=r'$\beta$'+str(i+1))
plt.xscale('log')
plt.legend(frameon=False, bbox_to_anchor=(0.585, 0.4, 0.5, 0.5))
plt.ylabel(r'$\beta$',fontsize=fontsize_label)
plt.xlabel(r'$\lambda$',fontsize=fontsize_label)
plt.savefig('4_5_fig.png')
'''


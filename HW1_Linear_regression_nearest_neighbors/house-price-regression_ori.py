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
#from sklearn.linear_model import Lasso

#prob1

#import data
#data = stats.zscore(np.loadtxt('housing.data'),axis=0)
data = np.loadtxt('housing.data')
#remove any possible ordering fx
seed = 2022; np.random.seed(seed)
randperm = np.random.randint(0,len(data),len(data))
new_data = []
for i in range(len(data)):
    new_data.append(data[randperm[i]])
data = np.array(new_data)
del new_data

def Vanilla_y_hat(A,y):
    #(A^T*A)^{-1}*A^T
    result = np.linalg.pinv((A.T).dot(A)).dot(A.T)
    beta = result.dot(y)
    return beta

def Ridge_y_hat(A,y,lam):
    #(A^T*A+lam*I)^(-1)*A^T, but use A^T(AA^T+\lam*I)^(-1) in this case
    #result = (A.T).dot(np.linalg.inv(A.dot(A.T)+lam*np.identity(len(A))))
    
    #The solution above will result in no inverse matrix when lambda lower than 1e-6, but the solution below can be calculated
    result = np.linalg.pinv((A.T).dot(A)+lam*np.identity(len(A.T))).dot(A.T)
    beta = result.dot(y)
    return beta

def MSE(y,beta,x):
    y_hat = x.dot(beta)
    #return sum([x**2 for x in (y-y_hat)])/len(y)
    return ((y-y_hat).T).dot(y-y_hat)[0][0]/len(y)
'''
#prob2
#selection for first n records of the trainning data
n = np.array([25,50,75,100,150,200,300])
#n= [25]
MSE_train = []
MSE_test = []
MSE_SL_train = []
MSE_SL_test = []
for Ntrain in n:
    #split the data into trainning and testing; then standarize data for trainning and testing,respectively
    Xtrain = stats.zscore(data[:Ntrain,:-1],axis=0); ytrain = data[:Ntrain,-1][...,None]
    Xtest = stats.zscore(data[Ntrain:,:-1],axis=0); ytest = data[Ntrain:,-1][...,None]
    #add intercept at first column
    Xtrain = np.concatenate((np.reshape(np.ones(len(Xtrain)),(len(Xtrain),1)),Xtrain),axis=1)
    Xtest = np.concatenate((np.reshape(np.ones(len(Xtest)),(len(Xtest),1)),Xtest),axis=1)
    beta = Vanilla_y_hat(Xtrain,ytrain)
    MSE_train.append(MSE(ytrain,beta,Xtrain))
    MSE_test.append(MSE(ytest,beta,Xtest))
    reg = sl.LinearRegression();
    reg.fit(Xtrain, ytrain)
    beta_sl=reg.coef_[0][...,None]
    beta_sl[0] = reg.intercept_[0]
    print((beta_sl-beta).T.dot(beta_sl-beta))
    MSE_SL_train.append(MSE(ytrain,beta_sl,Xtrain))
    y_hat = reg.predict(Xtest) - reg.intercept_[0]
    y = ytest
    MSE_SL_test.append(((y-y_hat).T).dot(y-y_hat)[0][0]/len(y))
plt.plot(n,MSE_train,color='r',label='train',marker='D')
plt.plot(n,MSE_test,color='b',label='test',linestyle='dashed',marker='D')
plt.plot(n,MSE_SL_train,color='green',label='train w Sklearn')
plt.plot(n,MSE_SL_test,color='cyan',label='test w Sklearn')
plt.legend()
'''
#'''
#prob3
Ntrain = 300
MSE_train = []
MSE_test = []
#split the data into trainning and testing; then standarize data for trainning and testing,respectively
Xtrain = stats.zscore(data[:Ntrain,:-1],axis=0); ytrain = data[:Ntrain,-1][...,None]
Xtest = stats.zscore(data[Ntrain:,:-1],axis=0); ytest = data[Ntrain:,-1][...,None]
#add intercept at first column
Xtrain = np.concatenate((np.reshape(np.ones(len(Xtrain)),(len(Xtrain),1)),Xtrain),axis=1)
Xtest = np.concatenate((np.reshape(np.ones(len(Xtest)),(len(Xtest),1)),Xtest),axis=1)
beta = Vanilla_y_hat(Xtrain,ytrain)
MSE_train.append(MSE(ytrain,beta,Xtrain))
MSE_test.append(MSE(ytest,beta,Xtest))
degree = [2,3,4,5,6];degree = [2]
for deg in degree:
    data_deg = np.concatenate((data[:,:-1],data[:,:-1]**deg,data[:,-1][...,None]),axis=1)
    Xtrain = stats.zscore(data_deg[:Ntrain,:-1],axis=0); ytrain = data_deg[:Ntrain,-1][...,None]
    Xtest = stats.zscore(data_deg[Ntrain:,:-1],axis=0); ytest = data_deg[Ntrain:,-1][...,None]
    Xtrain = np.concatenate((np.ones((len(Xtrain),1)),Xtrain),axis=1)
    Xtest = np.concatenate((np.ones((len(Xtest),1)),Xtest),axis=1)
    a = Xtrain.T.dot(Xtrain)
    beta = Vanilla_y_hat(Xtrain,ytrain)
    MSE_train.append(MSE(ytrain,beta,Xtrain))
    MSE_test.append(MSE(ytest,beta,Xtest))
plt.plot([1]+degree,MSE_train,color='r',label='train',marker='D')
plt.plot([1]+degree,MSE_test,color='b',label='test',linestyle='dashed',marker='D')
plt.legend()
#'''
'''
#prob4
Ntrain = 300
MSE_train = []
MSE_test = []
MSE_SL_test = []
MSE_SL_train = []
lamb = np.logspace(-10,10,10);#lamb = [1]#[1e-2,1e-1,1,1e1,1e2]
deg = 6
Xtrain = stats.zscore(data[:Ntrain,:-1],axis=0); ytrain = data[:Ntrain,-1][...,None]
Xtest = stats.zscore(data[Ntrain:,:-1],axis=0); ytest = data[Ntrain:,-1][...,None]
Xtrain = np.concatenate((Xtrain,Xtrain**deg),axis=1)
Xtest = np.concatenate((Xtest,Xtest**deg),axis=1)
Xtrain = np.concatenate((np.reshape(np.ones(len(Xtrain)),(len(Xtrain),1)),Xtrain),axis=1)
Xtest = np.concatenate((np.reshape(np.ones(len(Xtest)),(len(Xtest),1)),Xtest),axis=1)
for lam in lamb:
    reg = sl.Ridge(lam)
    reg.fit(Xtrain,ytrain)
    beta_sl = reg.coef_[0][...,None]
    beta_sl[0] = reg.intercept_[0]
    beta = Ridge_y_hat(Xtrain,ytrain,lam)
    print((beta_sl-beta).T.dot(beta_sl-beta)/len(beta))
    MSE_train.append(MSE(ytrain,beta,Xtrain))
    MSE_test.append(MSE(ytest,beta,Xtest))
    MSE_SL_train.append(MSE(ytrain,beta_sl,Xtrain))
    y_hat = reg.predict(Xtest) - reg.intercept_[0]
    y = ytest
    MSE_SL_test.append(((y-y_hat).T).dot(y-y_hat)[0][0]/len(y))
plt.plot(lamb,MSE_train,color='r',label='train',marker='D')
plt.plot(lamb,MSE_test,color='b',label='test',linestyle='dashed',marker='D')
plt.plot(lamb,MSE_SL_train,color='green',label='train w Sklearn',marker='D')
plt.plot(lamb,MSE_SL_test,color='cyan',label='test w Sklearn',marker='D')
plt.xscale('log')
plt.legend()
'''
'''
#prob5
Ntrain = 300
#split the data into trainning and testing; then standarize data for trainning and testing,respectively
Xtrain = stats.zscore(data[:Ntrain,:-1],axis=0); ytrain = data[:Ntrain,-1][...,None]
Xtest = stats.zscore(data[Ntrain:,:-1],axis=0); ytest = data[Ntrain:,-1][...,None]
#add intercept at first column
Xtrain = np.concatenate((np.reshape(np.ones(len(Xtrain)),(len(Xtrain),1)),Xtrain),axis=1)
Xtest = np.concatenate((np.reshape(np.ones(len(Xtest)),(len(Xtest),1)),Xtest),axis=1)
reg = sl.Lasso(alpha=0.001)
reg.fit(Xtrain,ytrain)
print(type(reg.intercept_))
print (reg.coef_)
'''


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 15:54:26 2022

@author: zhiyuas
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sl
def func(A,y,beta,lam):
    return ((A.dot(beta)-y).T.dot(A.dot(beta)-y)+lam*beta.T.dot(beta))[0][0]/2
def grad_func(A,y,beta,lam):
    return A.T.dot(A.dot(beta)-y)+lam*beta
def GD(f,gradient_f,feature, target, ini_beta, lam, learn_rate, tol):
    beta = ini_beta;ite = 0
    diff = 1e10
    #recording objective function
    ob_s = [f(feature,target,ini_beta,lam)]
    while diff > tol:
        beta = beta - learn_rate*gradient_f(feature,target,beta,lam)
        ite = ite + 1
        ob_s.append(f(feature,target,beta,lam))
        diff = abs((ob_s[-1]-ob_s[-2])/ob_s[-1])
    return ite, beta, ob_s[1:]
A = np.array([[1,2,4],[1,3,5],[1,7,7],[1,8,9]]);y=np.array([[1],[2],[3],[4]])
#f(x) = (A*beta-y)^T*(A*beta-y) innitial inputs
tol = 1e-5; learn_rate = 1; beta = np.array([0 for i in range(3)])[...,None];ite = 0;lam=0
lambdas = [0,0.1,1,10,100,200]
ites = [];ks = [];ob_score = [];cr = []
for lam in lambdas:
    LR = 1/(np.max(sl.svd(A.T.dot(A))[1])+lam)
    ite,_,ob_s = GD(func,grad_func,A,y,beta,lam,LR,tol)
    ites.append(ite);ks.append(LR);ob_score.append(ob_s)
    _,s,_ = sl.svd(A.T.dot(A)+lam*np.identity(3))
    cr.append(1-np.min(s)/np.max(s))
ob_score = [ob_s/ob_s[0] for ob_s in ob_score] #normalization
#plots
plt.figure(figsize=(14,8))
for lam,ite,ob_s,k,c in zip(lambdas,ites,ob_score,ks,cr):
    plt.plot(np.arange(1,ite+1),ob_s,label=r'LR = $\frac{{1}}{{{}+\sigma_{{max}}(A^TA)}}$='\
             .format(lam)+'%.5s'%(k*1000)+r'$\times 10^{-3}$; CR = '+'%.6s'%c)
plt.xscale('log')
plt.legend(frameon=False,fontsize=16)
plt.xlabel('Iterations',fontsize=14)
plt.ylabel('Normalized objective function scores',fontsize=14)
plt.savefig('prob3.png')
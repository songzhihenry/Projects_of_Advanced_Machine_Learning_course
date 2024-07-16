#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 17:10:48 2022

@author: zhiyuas
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sl
def func(A,y,beta,lam):
    return ((A.dot(beta)-y).T.dot(A.dot(beta)-y)+lam*beta.T.dot(beta))[0][0]/2
def grad_func(A,y,beta,lam):
    return A.T.dot(A.dot(beta)-y)+lam*beta
def GD(f,gradient_f,feature, target, ini_beta, lam, ini_learn_rate, tol):
    #backtracking line search params
    b=0.5;beta = ini_beta;ite = 0
    learn_rate = ini_learn_rate
    diff = 1e10
    #recording objective function
    #while learn_rate*gradient_f(feature,target,beta,lam).T.dot(gradient_f(feature,target,beta,lam)) > tol:
    ob_s = [f(feature,target,ini_beta,lam)]
    while diff > tol:
        learn_rate = ini_learn_rate
        while f(feature,target,beta-learn_rate*gradient_f(feature,target,beta,lam),lam) > \
        f(feature,target,beta,lam)-learn_rate*0.5*(gradient_f(feature,target,beta,lam)).T.dot(gradient_f(feature,target,beta,lam)):
            learn_rate = learn_rate*b
        beta = beta - learn_rate*gradient_f(feature,target,beta,lam)
        ite = ite + 1
        ob_s.append(f(feature,target,beta,lam))
        diff = abs((ob_s[-1]-ob_s[-2])/ob_s[-1])
    return ite, beta, learn_rate, ob_s[1:]
A = np.array([[1,2,4],[1,3,5],[1,7,7],[1,8,9]]);y=np.array([[1],[2],[3],[4]])
#1/(sigma_{max}(A^TA))
k = 1/np.max(sl.svd(A.T.dot(A))[1])
#f(x) = (A*beta-y)^T*(A*beta-y) innitial inputs
tol = 1e-5; learn_rate = 1; beta = np.array([0 for i in range(3)])[...,None];ite = 0;lam=0

lambdas = [0,0.1,1,10,100,200]
#k_lam = 1/np.max(sl.svd(A.T.dot(A))[1])
ites = [];rates = [];ks = [];ob_score = [];
for lam in lambdas:
    ite,_,learn_rate,ob_s = GD(func,grad_func,A,y,beta,lam,learn_rate,tol)
    ites.append(ite);rates.append(learn_rate);ks.append(1/(np.max(sl.svd(A.T.dot(A))[1])+lam))
    ob_score.append(ob_s)
ob_score = [ob_s/ob_s[0] for ob_s in ob_score] #normalization
#plots
plt.figure(figsize=(14,8))
for lam,ite,ob_s,LR,k in zip(lambdas,ites,ob_score,rates,ks):
    plt.plot(np.arange(1,ite+1),ob_s,label='LR: '+'%.4s'%(LR*1000)+'; '+r'1/(${}+sigma_{{max}}(A^TA)$)='.format(lam)+'%.4s'%(k*1000)+r' ($\times 10^{-3}$)')
plt.xscale('log')
#plt.yscale('log')
plt.legend(frameon=False,fontsize=16)
plt.xlabel('Iterations',fontsize=14)
plt.ylabel('Objective function scores',fontsize=14)
#plt.savefig('prob3.png')
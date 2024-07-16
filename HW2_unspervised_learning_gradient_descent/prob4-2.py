#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 13:53:35 2022

@author: zhiyuas
"""

import numpy as np
from PIL import Image
from scipy.linalg import svd
import matplotlib.pyplot as plt
def PAC_Decom_var(k_ev,image):
    m,n = image.size
    data = list(image.getdata())
    #read colors from the image
    color_sets = [np.array([x[j] for x in data]).reshape(m,n) for j in range(3)]
    #comput mean value for each color channel and centerize each channel
    mean_set = [np.mean(x) for x in color_sets]
    color_sets = [x-y for x,y in zip(color_sets,mean_set)]
    k=k_ev
    redu_sets = []
    for color_channel in color_sets:
        U,s,VT = svd(color_channel)
        redu_sets.append((U[:,:k].dot(np.diag(s[:k]).dot(VT[:k,:]))).reshape(m*n))
    redu_sets = np.rint(np.array([x+y for x,y in zip(redu_sets,mean_set)]))\
        .astype(int).transpose()
    return (np.sum((redu_sets-np.array(data))**2)/(len(data)*3))
fontsize_label = 16
#import image
im = Image.open('Lenna_(test_image).png')
#produce an empty image
best_com_num = [2,5,20,50,80,100]
cv = []
for k in best_com_num:
    cv.append(PAC_Decom_var(k,im))
plt.plot(best_com_num,np.array(cv)/cv[0]*100,label='Normalized variance')
plt.plot(best_com_num,np.array(best_com_num)*100/512,label=r'k$^{th}$ selection')
plt.xlabel('Principal Components',fontsize=fontsize_label)
plt.ylabel('Normalization of varviace and selection ratio',fontsize=12)
plt.legend(frameon=False,fontsize=fontsize_label)
plt.savefig('prob4-2.png')
    
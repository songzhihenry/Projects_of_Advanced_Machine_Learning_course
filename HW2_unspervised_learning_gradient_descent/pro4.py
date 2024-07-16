#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 16:27:35 2022

@author: zhiyuas
"""
import numpy as np
from PIL import Image
from scipy.linalg import svd
import matplotlib.pyplot as plt
def PAC_Decom(k_ev,image):
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
    #add mean back for each color channel and take int value for each elemnt
    redu_sets = np.rint(np.array([x+y for x,y in zip(redu_sets,mean_set)])).astype(int).transpose().tolist()
    redu_sets = [tuple(x) for x in redu_sets]
    return redu_sets
#import image
im = Image.open('Lenna_(test_image).png')
#produce an empty image
best_com_num = [2,5,20,50,80,100]
fig,axes = plt.subplots(figsize=(18,12),nrows=2, ncols=3)
axes = axes.flatten()
for k,ax in zip(best_com_num,axes):
    img = Image.new("RGB",im.size)
    img.putdata(PAC_Decom(k,im))
    ax.set_axis_off()
    ax.imshow(img)
    ax.set_title('First '+str(k)+'th components',fontsize=16,fontweight='bold')
#plt.savefig('prob4.png')



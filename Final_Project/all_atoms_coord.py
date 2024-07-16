# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 21:07:47 2022

@author: Henry Song
"""

import numpy as np
def get_coord(data):
    coord = []
    for i in range(3):
        coord.append([float(atm[30+i*8:30+8*(i+1)]) for atm in data])
    return np.array(coord).transpose()
'''
features = []
for snap in range(10000):
    data = [line.strip('\n') for line in open('D:/movies/{}.pdb'.format(snap))][:-2]
    coord = []
    for i in range(3):
        coord = coord + [float(atm[30+i*8:30+8*(i+1)]) for atm in data]
    features.append(coord)
features = np.array(features)
np.savetxt('all_atm_coords.txt', features)
'''
#data = [line.strip('\n') for line in open('D:/movies/0.pdb')][:-2]
#ref = get_coord(data)
#ref_mean = np.mean(ref,axis=0)
features = []
#features.append(ref.flatten())
for i in range(10000):
    data = [line.strip('\n') for line in open('D:/movies/{}.pdb'.format(i))][:-2]
    v = get_coord(data)
    #features.append((v - np.mean(v,axis=0) + ref_mean).flatten())
    features.append((v - np.mean(v,axis=0)).flatten())
features = np.array(features)
np.savetxt('all_atm_coords.txt', features)

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 11:02:27 2022

@author: willi
"""

import numpy as np
import json
import os

with open('./final_set_1533.json', 'r') as f:
    UTR_hit_list = json.load(f)



a = np.load('./lev_dist_arrays_1515/levdist_%s.npy'%UTR_hit_list[0])
lev_dists = np.zeros([len(UTR_hit_list), a.shape[0]])

for i in range(len(UTR_hit_list)):
    lev_dists[i,:] = np.load('./lev_dist_arrays_1515/levdist_%s.npy'%UTR_hit_list[i])
    

np.save('./levdist_array_1533.npy', lev_dists.astype(int))
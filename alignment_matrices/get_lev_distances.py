# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 12:43:50 2022

@author: willi
"""

import json
import numpy as np
from multiprocessing import Pool, cpu_count
import time
import os
from pathlib import Path

def lev_dist(token1, token2):
    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2
    a = 0
    b = 0
    c = 0
    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if (token1[t1-1] == token2[t2-1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]
                
                if (a <= b and a <= c):
                    distances[t1][t2] = a + 1
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    return distances[len(token1)][len(token2)]


with open('./final_set_1533.json', 'r') as f:
    UTR_hit_list = json.load(f)

with open('./utr_dot_hits_1533.json','r') as f:
  encoded_bears_hits = json.load(f)

with open('./rs_dot.json','r') as f:
  encode_bear_RS = json.load(f)


def thread_process(utr_id):
    levfile = Path('./levdist_%s.npy'%(UTR_hit_list[utr_id]))
    if levfile.exists():
        return
    else:
        ldists = [lev_dist(encoded_bears_hits[utr_id], j)  for j in encode_bear_RS]
        np.save('./levdist_%s.npy'%(UTR_hit_list[utr_id]), ldists)

        
    
if __name__ == '__main__':
    st = time.time()
    pool = Pool(processes=(48))
    
    results=pool.map(thread_process,range(0,len(UTR_hit_list)))

    pool.close()
    pool.join()
    print(time.time()-st)


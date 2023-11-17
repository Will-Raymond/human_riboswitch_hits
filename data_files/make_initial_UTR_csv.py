# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:05:50 2023

@author: willi
"""


from Bio import SeqIO  #BIOPYTHON
from Bio import pairwise2

import numpy as np
import itertools as it
import re
import os

import pandas as pd
import matplotlib.pyplot as plt

import itertools
import warnings
import time


import joblib 
import pickle

            

class kmer_DataProcessor():
    
    def __init__(self):
        self.unique_hits = set()
        self.data_file = ''
        self.all_fastas = []
        self.letter_dict = ['a','c','u','g']
        #preallocate small kmers
        self.kmer_4 = self.kmer_list(4)
        self.kmer_2 = self.kmer_list(2)
        self.kmer_3 = self.kmer_list(3)
        self.kmer_5 = self.kmer_list(5)
        self.kmer_1 = self.kmer_list(1)
        
        self.ns_dict = {'m':'a','w':'a','r':'g','y':'t','k':'g','s':'g','w':'a','h':'a','n':'a','x':'a'}
        
        self.test_seq = 'aucuguacguacguaucgaucguguacuggcaaaacguaguagcugagcaucaucuaugh'
        
    
    
    
    
    
    def create_database(self,path, u_thresh = 1, disp = False):
        '''
        Pull all sequences and throw out everything with a given threshold, if 
        u_thresh is 100% match it is fast
        otherwise this is a very slow function that needs alignment each iteration
        '''
        
        self.data_file = path
        self.all_fastas = list(SeqIO.parse(path,'fasta'))
        print('processing sequences')
        n = 0
        m = len(self.all_fastas)
        for f in self.all_fastas:
            n+= 1
            if n% 100 == 0:
                if disp:
                    print('processed: %d out of %d'%(n,m))
            if u_thresh == 1:
                newstr = self.clean_seq(str(f.seq))
                
                if newstr not in self.unique_hits:
                    self.unique_hits.add((f.id + '==='+newstr))
                    
            else:
                newstr = self.clean_seq(str(f.seq))
                best_match = self.check_percentage_alignments(newstr)
                
                if best_match < u_thresh:
                    
                    self.unique_hits.add((f.id + '==='+newstr))
        
        self.unique_hits = list(self.unique_hits)
        self.unique_ids = []
        for entry in self.unique_hits:
            self.unique_ids.append(entry.split('===')[0])
            
        self.get_unique_seqs()
        
        
    def reset_db(self,  u_thresh = 1,disp=False):
        print('resetting sequences')
        self.unique_hits = set()
        n = 0
        m = len(self.all_fastas)
        for f in self.all_fastas:
            n+= 1
            if n% 100 == 0:
                if disp:
                    print('processed: %d out of %d'%(n,m))
            if u_thresh == 1:
                newstr = self.clean_seq(str(f.seq))
                
                if newstr not in self.unique_hits:
                    self.unique_hits.add((f.id + '==='+newstr))
                    
            else:
                newstr = self.clean_seq(str(f.seq))
                best_match = self.check_percentage_alignments(newstr)
                
                if best_match < u_thresh:
                    
                    self.unique_hits.add((f.id + '==='+newstr))
                    
        self.unique_hits = list(self.unique_hits)
        self.unique_ids = []
        for entry in self.unique_hits:
            self.unique_ids.append(entry.split('===')[0])
            
        self.get_unique_seqs()        
        
                   
    def check_debruin_degeneracy():
        x=1
        
    def clean_seq(self,seq):
        '''
        clean the sequences to lowercase only a, u, g, c
        '''
        seq = seq.lower()
        
        for key in self.ns_dict.keys():            
            seq = seq.replace(key,self.ns_dict[key])
            
        seq = seq.replace('t','u')
        return seq
        
    
    def get_all_kmers(self,k):
        '''
        build the full array of kmers for the current database
        '''
        self.kmer_database = []
        for entry in self.unique_hits:
            kf = self.kmer_freq(entry.split('===')[1],k)
            self.kmer_database.append(kf)
            
        self.kmer_array = np.array(self.kmer_database)


    def get_all_kmers_length_dist(self,k,density,bins):
        '''
        build the full array of kmers for the current database
        '''
        self.kmer_database = []
        
        cum_density = np.cumsum(density)
        
        self.normalized_sizes = []
        self.normalized_seqs = []
        for entry in self.unique_hits:
            
            
            r = np.random.uniform()
            
            while len(np.where(r < cum_density)[0]) ==0:
                r = np.random.uniform()
                
            index = np.where(r < cum_density)[0][0]
                    
            length = int(bins[index])
            seq = entry.split('===')[1]
            
            if len(seq) > length:
                offset = int(np.random.uniform(0,len(seq)- length-1 ))
            else: 
                length = len(seq)
                offset = 0
                
            #print(len(seq[offset:(offset+ length)]))
            kf = self.kmer_freq(seq[offset:(offset+ length)],k)
            self.kmer_database.append(kf)
            self.normalized_sizes.append( len( seq[offset:(offset+ length)] ) )
            self.normalized_seqs.append(seq[offset:(offset+ length)])
            
            
            
        self.kmer_array = np.array(self.kmer_database)
        

    def get_all_kmers_norm(self,k):
        '''
        build the full array of kmers for the current database
        '''
        self.kmer_database_norm = []
        for entry in self.unique_hits:
            kf = self.kmer_freq(entry.split('===')[1],k)
            seq_length = len(entry.split('===')[1])
            self.kmer_database_norm.append(kf/seq_length)
            
        self.kmer_array_norm = np.array(self.kmer_database_norm)
                

    def remove_entries(self,id_list):
        to_remove_indexes = []
        new_all_fastas = []
    
        for i in range(len(self.all_fastas)):
            if self.all_fastas[i].id not in id_list:
                new_all_fastas.append(self.all_fastas[i])
        
        self.all_fastas = new_all_fastas
        
            

    def export_to_csv(self,filename,normalized=False):
        
        if normalized:
            k_arr = self.kmer_array_norm
            k_db = self.kmer_database_norm
        else:
            k_arr = self.kmer_array
            k_db = self.kmer_database
        
        k = k_arr.shape[1]
        n=1
        while k !=4:
            n+=1
            k= k/4
        k=n
        print(k)
        if k > 5:
            kmer_ind = self.kmer_list(k)
        else:
            if k == 1:
                kmer_ind = self.kmer_1
            if k == 2:
                kmer_ind = self.kmer_2
            if k == 3:
                kmer_ind = self.kmer_3            
            if k == 4:
                kmer_ind = self.kmer_4
            if k == 5:
                kmer_ind = self.kmer_5
                
        df = pd.DataFrame(k_db, columns = kmer_ind ) 
        df.insert(0,"ID",self.unique_ids,True)
        
        unique_seqs = []
        for entry in self.unique_hits:
            unique_seqs.append(entry.split('===')[1])
        
        
        df.insert(1,"SEQ",unique_seqs,True)
        df.to_csv((filename + '.csv'))
        
    def get_unique_seqs(self):
        unique_seqs = []
        for entry in self.unique_hits:
            unique_seqs.append(entry.split('===')[1])
        self.unique_seqs = unique_seqs
        self.get_sizes()
        
    def get_sizes(self):
        sizes = []
        for seq in self.unique_seqs:
            sizes.append(len(seq))
        self.all_sizes = sizes
        
    def check_percentage_alignments(self,seq):
        '''
        use global alignment to get percentage match to database 
        
        EXTREMELY SLOW
        '''
        max_align = 0
        lenseq = len(seq)
        for seq2 in self.unique_hits: 
            aligns = pairwise2.align.globalxx(seq, seq2)
            best_match = 0
            for align in aligns:
                if align[2] > best_match:
                    best_match = align[2]
                    
            percentage_match = best_match/lenseq
            if percentage_match > max_align:
                max_align = percentage_match
                
        return max_align
        

    def check_perecentage_via_freq(self,freq,thresh):
        '''
        check percentage of similarities of given kmer freq to the current database
        
        fast but inaccurate if the sequences have different sizes 
        
        '''
        similarities = []
        for kmer in self.kmer_database:
            similarities.append(np.mean( freq != kmer ))
            
        match_inds = np.where(np.array(similarities) < thresh)[0]
        return similarities, match_inds
        
        
    def kmer_list(self,k):        
        combos =[x for x in it.product(self.letter_dict, repeat=k)]       
        kmer = [''.join(y) for y in combos]
        return kmer
    
    def kmer_freq(self,seq,k):
        '''
        calculate the kmer frequences of k size for seq
        ''' 
        
        if k > 5:
            kmer_ind = self.kmer_list(k)
        else:
            if k == 1:
                kmer_ind = self.kmer_1
            if k == 2:
                kmer_ind = self.kmer_2
            if k == 3:
                kmer_ind = self.kmer_3            
            if k == 4:
                kmer_ind = self.kmer_4
            if k == 5:
                kmer_ind = self.kmer_5
                
        kmer_freq_vec = np.zeros((4**k)).astype(int)
        for i in range(len(seq)-k):
            kmer_freq_vec[kmer_ind.index(seq[i:i+k])] += 1
            
        return kmer_freq_vec

if __name__ == "__main__":
    utr5_db = kmer_DataProcessor()
    utr5_db.create_database('./5UTRaspic.Hum.fasta')
    utr5_db.get_all_kmers(3)
    utr5_db.export_to_csv('./test.csv') 
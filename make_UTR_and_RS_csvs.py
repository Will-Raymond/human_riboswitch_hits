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


################## Make the UTR csv ###########################################

utr5_file = './data_files/5UTRaspic.Hum.fasta'

utr5_db = kmer_DataProcessor()
print('Creating 5primeUTR csv from the following file: %s'%utr_file )
utr5_db.create_database(utr_file)
utr5_db.get_all_kmers(3)




import json
import tqdm
UTR_db = pd.read_csv('./data_files/test.csv')
utr_id_to_gene = json.load(open('./data_files/UTR_ID_to_gene.json','r'))
UTR_db['GENE'] =''
g_list = []
for i in range(len(UTR_db)):
  g_list.append(utr_id_to_gene[UTR_db['ID'].iloc[i]])
UTR_db['GENE'] = g_list
#parse out duplicates
cols = UTR_db.columns.tolist()
cols =  [cols[-1]] + cols[1:3]  + cols[3:-1]
UTR_db = UTR_db[cols]
UTR_db = UTR_db[UTR_db['GENE'] != 'duplicate']
UTR_db = UTR_db.reset_index(drop=True) #reset the indexes
UTR_db.columns

# Match the CCDS
ccds_file = "./data_files/ccds_2018/CCDS_nucleotide.current.fna" # @param {type:"string"}
ccds_text_file = "./data_files/ccds_2018/CCDS.current.txt" # @param {type:"string"}


ccds_attributes = pd.read_csv(ccds_text_file, delimiter='\t')

UTR_db['CCDS_ID'] = ''
UTR_db['CCDS'] = ''

r = []
for record in tqdm.tqdm(SeqIO.parse(ccds_file,'fasta')):
    r.append(record)

ccds_list = ccds_attributes['ccds_id'].values
for record in tqdm.tqdm(r):
    ccds_id = record.id.split('|')[0]
    if ccds_id in ccds_list:
        if ccds_attributes[ccds_attributes['ccds_id'] == ccds_id]['ccds_status'].values[0] != 'Withdrawn':

            gene = ccds_attributes[ccds_attributes['ccds_id'] == ccds_id]['gene'].values[0]
            if len(UTR_db[UTR_db['GENE'] == gene]['CCDS_ID']) != 0:
                #new_utr_data[new_utr_data['GENE'] == gene[0]]['CCDS_ID'] = record.id.split('|')[0]
                for ind in UTR_db[UTR_db['GENE'] == gene]['CCDS'].index:
                    UTR_db.iloc[ind,-1] = str(record.seq).lower().replace('t','u')
                    UTR_db.iloc[ind,-2] = ccds_id


UTR_db['STARTPLUS25'] = ''
UTR_db['NUPACK_25'] = ''
UTR_db['NUPACK_25_MFE'] = ''
cols = UTR_db.columns.tolist()
cols =  cols[0:3]  + [cols[-3]] + [cols[-2]]+ [cols[-1]] + [cols[-5]] + [cols[-4]]+cols[3:-5]
UTR_db = UTR_db[cols]
UTR_db.head()




# DETECT IF NUPACK IS INSTALLED
try: 
    import nupack
    nupack_installed = True
except:
    nupack_installed = False

if nupack_installed:
    def get_mfe_nupack(seq, n=100):
    
      model1 = Model(material='rna', celsius=37)
      example_hit = seq
      example_hit = Strand(example_hit, name='example_hit')
      t1 = Tube(strands={example_hit: 1e-8}, complexes=SetSpec(max_size=1), name='t1')
      hit_results = tube_analysis(tubes=[t1], model=model1,
          compute=['pairs', 'mfe', 'sample', 'ensemble_size'],
          options={'num_sample': n}) # max_size=1 default
      mfe = hit_results[list(hit_results.complexes.keys())[0]].mfe
      return mfe, hit_results

      
  
else:
    print('proceeding with dummy dot structures....')
    
    def get_mfe_nupack(seq):
        return [['....(...)....', -10]], 'test'


energies = []
dots = []
mfes = []

k = 0

max_window = 300
for i in tqdm.tqdm(range(0,len(UTR_db))):
  utr_seq = UTR_db['SEQ'][i]
  if len(UTR_db['CCDS'][i]) != 0:
    ccds = UTR_db['CCDS'][i]
    mature_mrna = utr_seq + ccds
    ### UTR + 25 NT near start
    if len(utr_seq) > max_window-25:
      seq = utr_seq[-max_window+25:] + ccds[:25]
    else:
      seq = utr_seq + ccds[:25]
    mfe,hr =  get_mfe_nupack(seq)
    mfes.append(mfe)
    UTR_db.iloc[i,3] = seq
    UTR_db.iloc[i,4] = str(mfe[0][0])
    UTR_db.iloc[i,5]  = mfe[0][1]
  k+=1


UTR_db.to_csv('./data_files/test.csv',index=False) 

1/0
################# Make the RS csv #############################################

from Bio import SeqIO  #BIOPYTHON
from Bio import pairwise2

import numpy as np
import itertools as it
import re
import os

import pandas as pd
import json
import time
import itertools as it
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import subprocess 



## All riboswitches were pulled from RNAcentral on 8.19.22, any entry with the tag "riboswitch"
## data was filtered for duplicates leaving (n = 73119)
## all ligands were extracted from RNAcentral and added to a database (RSid_to_ligand.json)

# this code parses them all to the same molecule names and then plots them.

# riboswitches considered speculative and thus labeled unknown: nhA-I motif, duf1646, raiA, synthetic, sul1,
#

RS_data = json.load(open('./riboswitch_RNAcentral_8.19.21.json.json'))

ligand_data = json.load(open('./RSid_to_ligand.json'))

unique_RS_ids = list(ligand_data.keys())

ligands = ['tpp', 'guanine', 'preq1', 'leucine','tryptophan', 'threonine','serine','cyclic di-gmp','gmp','cobalamin','molybdenum',
           'abocbl', 'nhaa-i','nhaa', 'glms','glucosamine','methionine','guanidine','fmn','thf',
           'nico','duf1646','magnesium','manganese','purine','fluoride','flavin','glycine','mg','ykok','m-box','adenine','lysine',
           'zmp','ztp','glutamine','sam','homocysteine','tyrosine','valine','b12','adocbl','alanine','guanidine','synthetic','glyq',' mn ',
           'ppgpp','guanosine',"2'dg-ii","dg-ii",'2-deoxy-d-glucose','2-dg','thf','thfa','tetrahydrofolate','raia','sul1','aminoglycoside',
           'guanidine-iii','guanidineiii','tetracycline','yybp-ykoy','proline','cyclic di-amp','cyclic-di-amp','ydao/yuaa',
           'ydao','yuaa','ydao-yuaa','histidine','aspartate','glct','glna']

ligand_alias_dict = {'': 'unknown',
                     'adenine':'adenine',
                     'adocbl':'adocbl',
                     'b12':'cobalamin',
                     'cobalamin':'cobalamin',
                     'cyclic di-gmp':'cyclic-di-GMP',
                     'cyclic-di-gmp':'cyclic-di-GMP',
                     'c-di-amp':'cyclic-di-AMP',
                     'cyclic-di-amp':'cyclic-di-AMP',
                     'dg-ii':"2'-dG-II",
                     'duf1646':'unknown',
                     'flavin':'FMN',
                     'glct':'protein',
                     'glcyine':'glcyine',
                     'glycine':'glycine',
                     'glms':'glucosamine-6-phosphate',
                     'glna':'glutamine',
                     'glucosamine':'glucosamine',
                     'glutamine':'glutamine',
                     'glyq':'tRNA',
                     'gmp':'GMP',
                     'guanidine':'guanidine',
                     'guanidine-iii':'guanidine',
                     'guanine':'guanine',
                     'histidine':'histidine',
                     'homocysteine':'homocysteine',
                     'leucine':'leucine',
                     'lysine':'lysine',
                     'magnesium':'Mg2+',
                     'mg':'Mg2+',
                     ' mn ':'Mn2+',
                     'manganese':'Mn2+',
                     'Manganese ':'Mn2+',
                     'methionine':'methionine',
                     'mfr':'purine',
                     'moco_rna_motif':'molybdenum',
                     'molybdenum':'molybdenum',
                     'obsolete cofactor?':'unknown',
                     'obsolete covaftor?':'unknown',
                     'nico':'Ni/Co',
                     'nhaa-i':'unknown',
                     'ppgpp':'(p)ppGpp',
                     'proline':'proline',
                     'purine':'purine',
                     'preq1':'preQ_1',
                     'raia':'unknown',
                     'sam':'SAM',
                     'sam_alpha':'SAM',
                     'serine':'serine',
                     'synthetic':'synthetic',
                     'sul1':'unknown',
                     'tetracycline':'tetracycline',
                     'tetrahydrofolate':'tetrahydrofolate',
                     'thf':'tetrahydrofolate',
                     'thfa':'tetrahydrofolate',
                     'threonine':'threonine',
                     'thiamine':'TPP',
                     'tpp':'TPP',
                     'trna':'tRNA',
                     'tryptophan':'tryptophan',
                     'tyrosine':'tyrosine',
                     'valine':'valine',
                     'ydao':'cyclic-di-AMP',
                     'ydao-yuaa':'cyclic-di-AMP',
                     'ykok':'Mg2+',
                     'yybp-ykoy':'Mn2+',
                     'zmp':'zmp-ztp',
                     'zmp-ztp':'zmp-ztp',
                     'ztp':'zmp-ztp',
                     'fmn':'FMN',
                     'fluoride':'fluoride',
                     'alanine':'alanine',
                     'aminoglycoside':'aminoglycoside',
                     }



amino_acids = ['arginine','histidine','lysine','aspartate','glutamine','serine','threonine','asparagine','cystine','glycine','proline',
               'alanine','valine','isoleucine','leucine','methionine','phenylalanine','tyrosine','tryptophan']
aa = False
clean_ligand_data = {}

for item in ligand_data.items():
    key,val = item
    if aa == True:
        clean_val = ligand_alias_dict[val]
        if clean_val in amino_acids:
            clean_val = 'Amino Acid'
        clean_ligand_data[key] = clean_val
       
    else:
        clean_val = ligand_alias_dict[val]
        clean_ligand_data[key] = clean_val      
    
    
    
    
    
    
    
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:05:50 2023

@author: willi
"""

'''
Identification of potential riboswitch elements in Homo
Sapiens mRNA 5’UTR sequences using Positive-Unlabeled
machine learning

By Dr. William Raymond, Dr. Jacob DeRoo, and Dr. Brian Munsky 


'''
##############################################################################
# This file makes the two CSV databases used for machine learning
# It takes the original RNA central json download to make the Riboswitch dataset
# and it takes the ccds (2018) and the 5'UTR human fasta file taken from UTRdb 1.0
##############################################################################

utr5_file = './data_files/5UTRaspic.Hum.fasta'
RS_file = './data_files/riboswitch_RNAcentral_8.19.21.json.json'
RS_ligand_file = './data_files/RSid_to_ligand.json'

UTR_csv_name = './data_files/test_UTR.csv'
RS_csv_name = './data_files/test_RS.csv'

##############################################################################
# Imports
##############################################################################
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
import json
import tqdm
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



utr5_db = kmer_DataProcessor()
print('Creating 5primeUTR csv from the following file: %s'%utr5_file )
utr5_db.create_database(utr5_file)
utr5_db.get_all_kmers(3)
utr5_db.export_to_csv(UTR_csv_name[:-4])


print('connecting the genes to the UTRdb IDs....')
UTR_db = pd.read_csv(UTR_csv_name)
utr_id_to_gene = json.load(open('./data_files/UTR_ID_to_gene.json','r'))
UTR_db['GENE'] =''
g_list = []
for i in range(len(UTR_db)):
  g_list.append(utr_id_to_gene[UTR_db['ID'].iloc[i]])
UTR_db['GENE'] = g_list
print('removing duplicate entries....')
#parse out duplicates
cols = UTR_db.columns.tolist()
cols =  [cols[-1]] + cols[1:3]  + cols[3:-1]
UTR_db = UTR_db[cols]
UTR_db = UTR_db[UTR_db['GENE'] != 'duplicate']
UTR_db = UTR_db.reset_index(drop=True) #reset the indexes
UTR_db.columns


print('matching the ccds to the utrs....')
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



print('generating dot structures with nupack THIS WILL TAKE A WHILE....')
# DETECT IF NUPACK IS INSTALLED
try: 
    import nupack
    nupack_installed = True
except:
    nupack_installed = False
    print('NUPACK is not installed on your system!! Please go to https://www.nupack.org/ and obtain a liscence to install it in your environment.')
    print('Generating the csv with dummy sequences.....')
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
        return [['....(...)....', -10.]], 'test'


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


# Patch out some things that were missed (3'UTRs)
# remove 3prime sequences
UTR_db = UTR_db[['3' != x[0] for x in  UTR_db['ID']]]

# rename duplicate ids to ID-N
ids_to_update = {}

ids = UTR_db['ID'].values.tolist()
for i in range(len(UTR_db)):
  if ids.count(UTR_db['ID'].iloc[i]) > 1:
    if UTR_db['ID'].iloc[i] not in ids_to_update.keys():
      ids_to_update[UTR_db['ID'].iloc[i]] = [i,]
    else:
      ids_to_update[UTR_db['ID'].iloc[i]] =  ids_to_update[UTR_db['ID'].iloc[i]] +  [i,]
for id in ids_to_update.keys():
 for i in range(len(ids_to_update[id])):
  UTR_db.iloc[ids_to_update[id][i],1] = id + '-' + str(i)


# Check the integrety of the file:

print('errors in IDs: %i'%len([x for x in UTR_db['ID'] if x[0] !='5']))

print('Blank Genes: %i'%len([x for x in UTR_db['GENE'] if not isinstance(x,str)]))
print('errors in Genes: %i'%len([x for x in [y for y in UTR_db['GENE'] if isinstance(y,str)] if not x.replace('-','').replace('.','').replace('_','').isalnum()]))
allowed = set(['a','g','u','c'])
print('sequences with incorrect char: %i'%len([x for x in UTR_db['SEQ'] if not set(x) <= allowed]))     
print('CCDS with incorrect char: %i'%len([x for x in UTR_db['CCDS'] if not set(x) <= allowed])) 
print('STARTPLUS25 with incorrect char: %i'%len([x for x in UTR_db['STARTPLUS25'] if not set(x) <= allowed])) 

allowed = set([')','.','('])
#print('NUPACK_DOT with incorrect char: %i'%len([x for x in UTR_db['NUPACK_DOT'] if not set(x) <= allowed]))     
print('NUPACK_25 with incorrect char: %i'%len([x for x in UTR_db['NUPACK_25'] if not set(x) <= allowed]))     
print('NUPACK_25_MFE with invalid values: %i'%len([x for x in UTR_db['NUPACK_25_MFE'] if not isinstance(x,float)]))


UTR_db.to_csv(UTR_csv_name,index=False) 
print('final database size UTR: ')
print(UTR_db.shape)

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

print('Generating RS csv....')

## All riboswitches were pulled from RNAcentral on 8.19.22, any entry with the tag "riboswitch"
## data was filtered for duplicates leaving (n = 73119)
## all ligands were extracted from RNAcentral and added to a database (RSid_to_ligand.json)

# this code parses them all to the same molecule names and then plots them.

# riboswitches considered speculative and thus labeled unknown: nhA-I motif, duf1646, raiA, synthetic, sul1,
#

RS_data = json.load(open(RS_file))

ligand_data = json.load(open(RS_ligand_file))

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


# Create the columns for the CSV
ids = []
seqs = []
descs = []
ligands = []
columns = ['ID','DESC','LIGAND','SEQ']
for item in RS_data:
    tmpid = item['id']
    if tmpid not in ligands:
        seq = item['sequence']
        if seq not in seqs:
            descs.append(item['description'])
            ids.append(item['id'])
            seqs.append(item['sequence'])
            ligands.append(clean_ligand_data[tmpid])
    

# create a pandas data frame
RS_df = pd.DataFrame(list(zip(ids,descs,ligands,seqs)), columns=columns)    
    
    
# DETECT IF NUPACK IS INSTALLED
try: 
    import nupack
    nupack_installed = True
except:
    nupack_installed = False
    print('NUPACK is not installed on your system!! Please go to https://www.nupack.org/ and obtain a liscence to install it in your environment.')
    print('Generating the csv with dummy sequences.....')
    
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
ns_dict = {'m':'a','w':'a','r':'g','y':'t','k':'g','s':'g','w':'a','h':'a','n':'a','x':'a'}

def clean_seq(seq):
    '''
    clean the sequences to lowercase only a, u, g, c
    '''
    seq = seq.lower()
    for key in ns_dict.keys():
        seq = seq.replace(key,ns_dict[key])

    seq = seq.replace('t','u')
    return seq

def kmer_list(k):
    combos =[x for x in it.product(['a','c','u','g'], repeat=k)]
    kmer = [''.join(y) for y in combos]
    return kmer

def kmer_freq(seq,k=3):
    '''
    calculate the kmer frequences of k size for seq
    '''
    kmer_ind = kmer_list(k)
    kmer_freq_vec = np.zeros((4**k)).astype(int)
    for i in range(len(seq)-k):
        kmer_freq_vec[kmer_ind.index(seq[i:i+k])] += 1

    return kmer_freq_vec

# Calculate and add the dot structures

RS_df['NUPACK_DOT'] = ''
RS_df['NUPACK_MFE'] = ''

for k in kmer_list(3):
    RS_df[k] = '' 

for i in tqdm.tqdm(range(0,len(RS_df))):
  seq = clean_seq(RS_df['SEQ'][i])
  mfe,hr =  get_mfe_nupack(seq)
  mer3 = kmer_freq(seq.lower())

  RS_df.iloc[i,4] = str(mfe[0][0])
  RS_df.iloc[i,5]  = mfe[0][1]
  RS_df.iloc[i,6:6+64] = mer3

print('inserting eukaryotic flag...')
RS_df.insert(2,'EUKARYOTIC',0)

# Manual sorting of key words that split up the RS csv, we are going to look
# for the key words in the description scraped from the RNA central entries
# below are keywords to label as eukaryotic:
eukaryote_list = ['TPP-specific','Phaeoacremonium','TPP-specific riboswitch from Arabidopsis','Paracoccidioides','Ophiocordyceps','Hyphopichia','Neonectria','Fibulorhizoctonia','Beta vulgaris', 'Caenorhabditis','Cinara','Seminavis', 'Thalictrum','Drosophila', 'Olea', 'Salix', 'Hymenolepis','Bathymodiolus','Steinernema','Zymoseptoria', 'Serpula', 'Rhizophagus', 'Dichanthelium', 'Gaeumannomyces','Yarrowia', 'Amazona','Ipomoea','Helianthus','Taphrina','Emergomyces','Picea', 'Fibroporia','Picea','Malassezia','Arthrobotrys', 'Poa','Lupinus','Tuber', 'Magnaporthe','Thielavia','Arthroderma',  'Lawsonia','Geomyces', 'Aedes','Debaryomyces','Hyaloperonospora', 'Theobroma','Acyrthosiphon', 'Komagataella', 'Solanum','Populus','Xylona','Podospora', 'Setaria',  'Leucosporidiella', 'Ricinus','Rhodnius','Brugia', 'Scheffersomyces', 'Microbotryum', 'Spathaspora', 'Anopheles', 'Chlamydomonas','Volvox','Zea', 'Coprinopsis', 'Wickerhamomyces', 'Myceliophthora', 'Pythium','Exidia', 'Byssochlamys','Madurella','Micromonas', 'Chaetomium','Meyerozyma','Botrytis', 'Setosphaeria', 'Daedalea','Prunus','Calocera', 'Fomitiporia', 'Lichtheimia','Brachypodium', 'Physcomitrella','Scedosporium','Pachysolen', 'Dactylellina','Grosmannia','Cajanus','Trichophyton', 'Perkinsus', 'Phaeodactylum','Piloderma', 'Jatropha',  'Pleurotus', 'Fragilariopsis', 'Fragilariopsis', 'Morus','Cyphellophora', 'Protochlamydia','Galerina', 'Kuraishia','Dothistroma','Capsicum', 'Heterobasidion', 'Lipomyces', 'Pyrenophora','Selaginella', 'Selaginella','Sphaeroforma', 'Nitzschia', 'Lucilia','Plasmopara','Babjeviella', 'Cyberlindnera', 'Reticulomyxa', 'Drechmeria', 'Pochonia', 'Coccomyxa','Escovopsis', 'Baudoinia','Escovopsis','Serendipita','Valsa','Parasitella','Cylindrobasidium', 'Ascoidea', 'Mortierella','Wallemia', 'Moniliophthora', 'Agaricus','Neurospora', 'Nasonia','Ciona','Ajellomyces','Phaeosphaeria','Ogataea','Rhinocladiella','Polyangium','Pyronema','Laccaria','Capsella','Gymnopus','Hypocrea','Hyphodontia','Rosa','Guillardia','Diplodia','Didymella','Paxillus','Clonorchis','Kwoniella','Claviceps','Hordeum','Stachybotrys','Neofusicoccum', 'Gossypium','Rasamsonia','Sphaerulina','Dichomitus','Punica','Eutrema','Suillus', 'Rosellinia', 'Diaporthe', 'Torrubiella', 'Nadsonia','fungal', 'Ochroconis','Toxocara', 'Coniosporium','Tortispora', 'Phaseolus','Verticillium', 'Klebsormidium', 'Glarea', 'Pneumocystis', 'Aphanomyces','Phytophthora','Cucumis','Parastrongyloides','Botryosphaeria','Rhizopogon','Chroococcidiopsis','Vitis','Emmonsia','Oidiodendron', 'Metschnikowia', 'Microdochium','Mimulus','Kribbella','Saitoella','Acremonium','Brassica','Eutypa', 'Trichoderma', 'Tolypocladium','Pisolithus','Protomyces','Monoraphidium','Citrus','Lobosporangium','Leucoagaricus','Pestalotiopsis','Schwartzia','Ananas', 'Fonsecaea','Paraphaeosphaeria','Stagonospora', 'Leptonema','Phialophora','Talaromyces','Citreicella','Penicilliopsis','Pyrenochaeta','Purpureocillium','Cladophialophora','Basidiobolus','Uncinocarpus','Neolecta','Thalassiosira','Coccidioides','Rhynchosporium','Fistulifera','Daucus','arabidopsis','aspergillus', 'Zostera','Aschersonia','Eucalyptus','Beauveria','Stemphylium',
                  'Sclerotinia','Penicillium','Marchantia','Pseudocercospora','Amborella','Mucor','Triticum',
                  'Corchorus','Colletotrichum','Cephalotus','Spinacia','Phialocephala','Absidia','Coniochaeta',
                  'Gibberella','Oryza','Capronia','Candida','Lasallia','Rachicladosporium','Nectria','Phycomyces',
                  'Rhizopus','Neosartorya','Fusarium','Exophiala','Metarhizium','Leersia','Brettanomyces','Marssonina',
                  'Mycosphaerella','Rhizoclosmatium','Lentinula','Glossina','Cicer','Thiomicrospira','Rhizoctonia','Blastomonas', 'Arabis',
                  'Macleaya','Kordiimonas','Gonium','Aureobasidium','Cordyceps','Ustilaginoidea','Saprolegnia','Choanephora','Hirsutella',
                  'Trachymyrmex','Musa','Pichia','Isaria','Alternaria','Sporothrix','Alternaria','Acidomyces','Medicago',
                  'Phaeomoniella','Hortaea','Hammondia','Macrophomina','Vigna','Clohesyomyces','Ophiostoma','Symbiodinium','Bipolaris']

elist = [y.lower() for y in eukaryote_list]

#things to remove if they are prokaryotic
remove_list = ['[Polyangium]','[Polyangium] brachysporum','candidate division KSB1','Syntrophotalea','Longilinea','Histophilus','Weeksella','Marinoscillum','Osedax symbiont','Plautia stali symbiont','Stappia','proteobacterium','Methanosalsum','artificial','Sediminimonas','spirochete','Oceanivirga', 'Nautella', 'Salibaculum', 'Kandeliimicrobium', 'Cribrihabitans', 'Pontibaca', 'Endomicrobium', 'Flavimaricola','Litorimicrobium','Lachnoanaerobaculum','Desulfatiglans','Tranquillimonas','Hyella','Kyrpidia','Citreimonas', 'Gracilimonas', 'Kordia', 'Kordia', 'Brevefilum', 'Brevefilum','Georgfuchsia', 'Wolbachia', 'Vannielia','Pleomorphomonas','Tabrizicola', 'Planktotalea', 'Albidovulum', 'Jhaorihella', 'Salinihabitans', 'Micropruina', 'Ammonifex', 'Tabrizicola', 'Meinhardsimonia', 'Pelagimonas', 'Bieblia', 'Roseicitreum', 'Poseidonocella', 'Limimaricola', 'Pontivivens','Holophaga', 'Ascidiaceihabitans', 'Holophaga', 'Romboutsia', 'Petrocella', 'Mameliella', 'Nioella', 'Oceaniglobus','Brevirhabdus', 'Romboutsia',  'Roseibaca', 'Yoonia', 'Thalassobium', 'Singulisphaera', 'Wolinella', 'Plasmodium', 'Mariprofundus', 'Thiorhodospira', 'Elusimicrobium','Kouleothrix', 'Sulfuritalea', 'Thermogutta', 'Hydrocarboniphaga', 'Singulisphaera','Alcanivorax', 'Alcanivorax', 'Desulfobacca', 'Sorangium', 'Paludisphaera', 'Sorangium', 'Alcanivorax', 'Planctomyces', 'Advenella', 'Cycloclasticus','Allochromatium', 'Thermobaculum', 'Pelotomaculum', 'Rhodomicrobium', 'Intestinimonas','Ectothiorhodospira','Magnetospira', 'Pedosphaera', 'Microterricola', 'Schleiferia', 'Phaeospirillum',  'Halomicronema', 'Sinomicrobium','Soonwooa','Methylomagnum','Verrucomicrobium', 'Yonghaparkia', 'Crinalium','Kozakia','Streptacidiphilus','Alkanindiges', 'Simkania', 'Streptacidiphilus', 'Methyloprofundus','Tangfeifania','Syntrophaceticus', 'Sunxiuqinia','Trichodesmium', 'Thermovirga', 'Methylorubrum', 'Methylorubrum', 'Microcystis', 'Clostridioides','Sneathiella', 'Gallionella','Hirschia', 'Stackebrandtia', 'Stackebrandtia','Dechlorosoma', 'Leptothrix', 'Nodularia', 'Sulfurihydrogenibium', 'Sideroxydans', 'Planktothrix', 'Scardovia', 'Aminomonas', 'Plesiocystis', 'Aromatoleum','Petrimonas', 'Microscilla', 'Scardovia', 'Mesotoga',  'Citromicrobium', 'Maricaulis', 'Dickeya', 'Sagittula', 'Microscilla''Petrimonas', 'Clostridiales', 'Pusillimonas','Thiocapsa','Alicycliphilus','Herpetosiphon','Synechocystis','Boseongicola', 'Chloroherpeton', 'Raphidiopsis','Polymorphum', 'Filifactor', 'Lautropia', 'Reinekea', 'Roseibium', 'Thalassiobium','Shigella', 'Ketogulonicigenium','Couchioplanes', 'Orenia','Zhouia','Thiohalocapsa','Fulvimarina', 'Zobellia', 'Kiritimatiella','Halothermothrix', 'Bulleidia', 'Confluentimicrobium', 'Saccharicrinis','Ethanoligenens', 'Bilophila', 'Catenulispora', 'Robiginitalea', 'Verrucosispora','Crocosphaera', 'Methylocella','Oscillochloris','Luteitalea', 'Dictyoglomus', 'Roseisalinus','Marvinbryantia', 'Gillisia', 'Pelistega', 'Hahella', 'Saccharophagus', 'Coraliomargarita', 'Actinosynnema','Abiotrophia', 'Eikenella', 'Morganella', 'Tolumonas','Sebaldella', 'Parvimonas', 'Truepera', 'Methylacidiphilum','Lacunisphaera', 'Desulfobacula', 'Ferrimicrobium', 'Acetohalobium', 'Halorhodospira','Serpens','Salinisphaera', 'Tannerella', 'Dokdonella', 'Jonquetella', 'Oligotropha','Thermobifida', 'Kingella','Thioalkalimicrobium', 'Haliangium', 'Hydrogenivirga', 'Stigmatella', 'Runella', 'Microcoleus','Pseudohaliea', 'Desulfocapsa', 'Agarivorans','Elstera','Rhodospirillum', 'Flexistipes', 'Palleronia','Desertifilum', 'Arthrospira', 'Granulicatella','Thermomonospora', 'Marichromatium', 'Thermus','Bizionia', 'Anaerofustis', 'Limnoraphis', 'Anaerophaga', 'Beijerinckia','Tepidicaulis','Rahnella', 'Aequorivita','Fimbriimonas', 'Thermodesulfobium','Crenothrix', 'Thermosinus','Brucella','Desulfarculus', 'Roseiflexus','Fischerella', 'Intrasporangium', 'Rickettsiella', 'Mesoplasma', 'Photorhabdus', 'Segniliparus', 'Cyanothece','Syntrophus', 'Desulfobulbus', 'Syntrophothermus','Oligella', 'Inquilinus', 'Thermomicrobium','Methylotenera', 'Methylomicrobium','Edwardsiella', 'Marinithermus','Starkeya', 'Maliponia', 'Marinithermus''Edwardsiella','Enhygromyxa','Acidithrix','Rhodoplanes', 'Labilithrix','Frischella', 'Isoptericola','Tamlana','Slackia', 'Bermanella', 'Jonesia', 'Pelodictyon', 'Methanocorpusculum', 'Lacinutrix', 'Thermincola', 'Cylindrospermum', 'Mumia', 'Tamlana''Thermosipho', 'Thermotoga', 'Desulfurispirillum','Shuttleworthia', 'Thermobispora','Parvularcula', 'Thermosipho','Catonella', 'Hylemonella', 'Acaryochloris', 'Picrophilus', 'Chelativorans', 'Mahella','Delftia', 'Parvibaculum',  'Prochlorothrix', 'Dechloromonas', 'Propionimicrobium', 'Ventosimonas','Desulfotignum', 'Alcaligenes', 'Thiocystis', 'Hyalangium', 'Sarcina','Halothece', 'Atopobium', 'Halothece' 'Marinitoga','Mucinivorans','synthetic','Gemella', 'Achromatium', 'Marinitoga','Elizabethkingia','Desulfatitalea','Salinispira','Nitrospina', 'Varibaculum','Hassallia','Sedimenticola', 'Thermoanaerobaculum', 'Plesiomonas','Gardnerella', 'Dehalococcoides','Haloplasma', 'Johnsonella', 'Desulfohalobium', 'Veillonella', 'Succinatimonas','Cellulophaga', 'Desmospora', 'Beutenbergia','Idiomarina','Cytophaga','Acetonema', 'Kosmotoga', 'Thermoplasma','Oceanicella', 'Desulfonatronospira', 'Desulfatibacillum','bacillum','Trueperella','Rothia', 'Zymomonas', 'Rothia', 'Salinispora','Propionispora', 'Thalassolituus', 'Pelagibaca', 'Anaeroglobus','Collimonas','Chamaesiphon', 'Robinsoniella','Chania','Eggerthia','Phycisphaera', 'Dethiosulfatarculus', 'Cyanobium', 'Scytonema','Agreia', 'Lacimicrobium', 'Bellilinea','Psychromonas', 'Barnesiella', 'Pelagicola', 'Thiolapillus', 'Oleiphilus', 'Methyloversatilis', 'Oleispira','Thalassomonas','Tistrella', 'Leminorella', 'Tateyamaria', 'Turicella', 'Rivularia','Chthonomonas','Psychroflexus','Archangium','Rhodonellum', 'Asticcacaulis', 'Stanieria', 'Lentimicrobium', 'Kaistia', 'Leisingera', 'Spiroplasma', 'Nitritalea', 'Marmoricola', 'Christensenella', 'Defluviimonas', 'Desulfuromonas','Desulfomicrobium','Caldithrix','Basilea', 'Lasius','Balneola', 'Phormidesmis', 'Fulvivirga', 'Halioglobus', 'Fervidicella','Ideonella', 'Yangia', 'Buttiauxella','Beggiatoa','Adlercreutzia', 'Moellerella', 'Proteus', 'Gallaecimonas','Mannheimia', 'Formosa','Marinactinospora', 'Aestuariivita', 'Marinactinospora','Salmonella', 'Anaerolinea', 'Pragia', 'Alloactinosynnema', 'Alkaliphilus', 'Atopostipes', 'Collinsella', 'Methylibium', 'Oceanicaulis', 'Bibersteinia', 'Methylophilus', 'Bhargavaea','Saprospira', 'Ottowia', 'Kandleria','Gynuella', 'Oceanibulbus','Actinopolyspora', 'Flammeovirga','Brochothrix', 'Pseudogulbenkiania', 'Acetomicrobium','Desulfamplus', 'Phlebia', 'Finegoldia', 'Caldicellulosiruptor', 'Desulfocarbo', 'Fimbriiglobus', 'Candidimonas', 'Malonomonas', 'Azonexus', 'Prosthecomicrobium', 'Megamonas', 'Mangrovimonas','Wohlfahrtiimonas', 'Geofilum','Austwickia', 'Caldisericum', 'Joostella','Alistipes', 'Pleurocapsa','Dysgonomonas', 'Magnetospirillum','Desulfonispora', 'Desulfonispora', 'archaeon', 'Desulfacinum', 'Bartonella', 'Chondromyces', 'Carboxydocella', 'Fibrella', 'Dubosiella', 'Xuhuaishuia','Neptunomonas','Thermanaerothrix', 'Lechevalieria', 'Brachyspira', 'Thiomonas', 'Hafnia', 'Enterorhabdus','Acholeplasma', 'Tatlockia','Escherichia','Sutterella','Sharpea', 'Izhakiella', 'Sulfurovum', 'Saccharopolyspora', 'Dyella','Tenacibaculum','Flagellimonas', 'Jeongeupia','Oceanospirillum','Yersinia','Oceanimonas','Halotalea', 'Catenovulum', 'Kutzneria', 'Snodgrassella', 'Thermoplasmatales', 'Subdoligranulum','Kineosphaera','Thiohalorhabdus', 'Ralstonia', 'Fervidicola', 'Actinobaculum', 'Oceanibaculum','Nonlabens', 'Acidiphilium', 'Eggerthella', 'Salipiger', 'Thermovenabulum', 'Salipiger', 'Sandarakinotalea','Chloroflexus', 'Cupriavidus', 'Xylanimonas', 'Calothrix', 'Megasphaera', 'Thioploca','Leclercia', 'Vitreoscilla', 'Reichenbachiella', 'Sulfurivirga', 'Thiobacimonas','Aquitalea', 'Roseivivax', 'Thiothrix', 'Arenimonas', 'Caldanaerobius', 'Marininema', 'Hapalosiphon','Sedimentitalea', 'metagenome', 'Tsukamurella', 'Limnothrix', 'Grimontia', 'Faecalibaculum', 'Desulfoplanes','Aeromonas', 'Aliifodinibius', 'Nelumbo', 'Prosthecochloris', 'Mobiluncus', 'Sulfurospirillum', 'Thermanaeromonas', 'Leptospira','Globicatella','Tropicimonas','Flaviramulus','Roseateles','Actinoalloteichus','Rubritalea','Actinomadura','Leeuwenhoekiella','Selenomonas','molybdenum','Dietzia','Syntrophomonas','Silvanigrella','Pseudorhodoplanes','Roseomonas','Seinonella','Klebsiella','Flavonifractor','Xylophilus','Methylomonas','Thermacetogenium','Insolitospirillum','Proteiniborus','Leadbetterella','Gramella','Croceivirga','Xylella','Cruoricaptor','Natranaerobius','Nakamurella','Tissierella','Mariniphaga','Nonomuraea','Friedmanniella','Fluviicola','Ornithinimicrobium','Chitinimonas','Maribius','Cedecea','Cobetia','Chlorobium', 'Sodalis','Garciella','Sandaracinus','Zhihengliuella','Wenxinia','coccus', 'bacillus','bascillus', 'unknown', 'bacterium', 'clostridium', 'sinorhizobium','streptosporangium', 'bacter', 'subtilis', 'coli', 'streptomyces', 'pseudolabrys', 'desulfovibrio',
               'unclassified', 'porphyromonas', 'azoarcus', 'Caloramator', 'Pseudonocardia', 'Pseudacidovorax','Oceanicola','Tolypothrix', 'Gloeocapsa', 'Leptotrichia','Thermoactinomyces', 'Paraglaciecola','Shinella','Saccharomonospora','Cecembia','Kurthia','Rhizobium','Burkholderia','Leptolinea','Novosphingobium','Limnochorda', 'Methylosinus','Kibdelosporangium','Actinotalea','Francisella','Microlunatus','Gemmatimonas','Woeseia','Nocardioides','Actinoplanes','Halomonas','Blautia','Bosea','Acidisphaera','Desulfosporosinus','Sphingobium','Oerskovia','Pseudomonas','Erwinia','Thermobrachium',
               'Kiloniella','Kiloniella','Thioclava','Nesterenkonia','Duganella','Hydrogenovibrio','Bordetella','Kerstersia','Aureimonas','human gut','Mitsuokella','Nocardiopsis','Bernardetia','Thalassospira','Donghicola','Williamsia','Cellulosimicrobium','Dorea','Noviherbaspirillum','Nitrosomonas',
               'Lyngbya','Gordonia','Georgenia','Listeria','Pandoraea','Loktanella','Skermanella','Vibrio','Sphingomonas',
               'Alishewanella','Kluyvera','Solemya', 'Caenispirillum','bioreactor','Brevundimonas','Mycoplasma','Luteimonas',
               'Pseudorhodoferax','Nitratireductor','Acidovorax','Spirochaeta','Shewanella','Ornatilinea','Marinomonas',
               'Legionella','Kitasatospora','Winogradskyella','Caballeronia','Pelosinus','Frondihabitans','Branchiibius',
               'Pseudogymnoascus','Pseudoxanthomonas','Devosia', 'Anaerosporomusa','Mitsuaria','Caryophanon','Holdemania',
               'Variovorax','Actinomyces','Thermotalea','Niastella','Methanomicrobiales','Shimia','Nocardia','Sinomonas',
               'Prevotella','Sphaerotilus','Mariniradius','Agromyces','Sinomonas','Endozoicomonas',
               'Frateuria','Pseudoalteromonas','Lawsonella','Geomicrobium','fluoride','cobalamin','purine',
               'SAM riboswitch','Castellaniella','endosymbiont','Castellaniella','Moraxella',
               'Halanaerobium','ZMP','Hydrogenophaga','Asaia','Myroides','Neisseria','Lentisphaera',
               'Thauera','Acidomonas','Nitrincola','glycine','FMN','PreQ1','Labrenzia','Limnohabitans',
               'lysine','Micromonospora','Oblitimonas','Methylocystis','di-GMP-II','Mastigocoleus',
               'Planomonospora','Thalassobius','Proteiniphilum','Rhodovulum','Lutispora','Alteromonas','Chitinophaga',
               'Acidimicrobium','Ochrobactrum','Treponema','Blastochloris','NiCo','Sporosarcina','Filimonas','Marivita',
               'Actinophytocola','Lactonifactor','Spirosoma','Desulfotomaculum','Gilliamella','agariphila','SAM-I','SAM/SAH','Afipia',
               'Algoriphagus','Salimicrobium','Planomicrobium','Brenneria','Hathewaya','Raoultella','Epulopiscium',
               'Anaerosphaera','Xanthomonas','Marinospirillum','Nitrospira','Saccharothrix','Mesonia','Comamonas',
               'Massilia','Xenorhabdus','glutamine','Lampropedia','Streptoalloteichus','Actinokineospora',
               'Nitrosospira','Planktothricoides','Oscillatoria','Anaerocolumna','Peptoniphilus','Vogesella',
               'Candidatus', 'Geodermatophilus','Roseovarius','Akkermansia','Seonamhaeicola','Pseudozobellia','Cnuella',
               'Luteipulveratus','Muricauda','Pantoea','Jatrophihabitans','Albidiferax','Cellulomonas','Olsenella',
               'Aurantimonas','Herbinix','Roseofilum','Lonsdalea','Kushneria','Frankia','Tatumella','Solibius','Orrella',
               'Levilinea','Marinovum','Gemmata','Smithella','Tepidimonas','Hyphomicrobium','Stenotrophomonas','Hoeflea',
               'Mastigocladus','Rubellimicrobium','Emticicia','Herbaspirillum','Weissella','Sulfuricella','Synergistes',
               'Ensifer','Knoellia','Niabella','Microbulbifer','Leifsonia','Ferroplasma','Microvirga','Siansivirga',
               'Ruegeria','Sphingorhabdus','Facklamia','Belliella','Sporomusa','Dehalogenimonas','Nitrolancea','Criblamydia',
               'Prauserella','Tetrasphaera','Thalassotalea','Yokenella','Azospirillum','Acidocella','Anaerotruncus',
               'Aeromicrobium','Pelomonas','Wenyingzhuangia','Jannaschia','Rheinheimera','Piscirickettsia',
               'Piscirickettsia','Desulfotalea','Zunongwangia','Providencia','Opitutus','Methylophaga','Pasteurella',
               'Rhodoferax','Actinopolymorpha','Microbispora','Serratia','Mycetocola','Geminocystis','Trabulsiella',
               'Aquaspirillum','Anaerobium','Martelella','Aquimixticola','Colwellia','Ferrithrix','Roseburia',
               'Andreprevotia','Desulfopila','Anaerostipes','Chishuiella','Sphingopyxis','Lentzea','Euryarchaeota','Gloeomargarita',
               'Hespellia','Jiangella','Marivirga','Ferrimonas','Haemophilus','Amycolatopsis','Asanoa','Phormidium',
               'Fibrisoma','Aquiflexum','Cryptosporangium','Caminicella','Jeotgalibaca','Tanticharoenia','Aquamicrobium',
               'Dialister','Owenweeksia','Haematospirillum','Kocuria','Glaciecola','Coxiella','Oleiagrimonas','Ahrensia',
               'Streptomonospora','Cohnella','Hyphomonas','Rubrivivax','Moorella','Kangiella','Nostoc','Moritella',
               'Aquimarina', 'Nereida','Aphanizomenon', 'Methylovorus',
               'Wenzhouxiangella', 'Riemerella', 'Lunatimonas', 'Solirubrum', 'Geitlerinema',
               'Polaromonas','Actinospica','Anabaena','Roseivirga','Aliterella','Richelia','Erysipelothrix','Crenarchaeota',
               'Solitalea','Dolosigranulum','Methylobrevis', 'Oceaniovalibus', 'Simiduia', 'Methylobrevis','Pacificimonas','Caldilinea']

# sort the remaining descriptions
a = RS_df['DESC'] 
b = [x for x in a if True in [y.lower()  in x.lower() for y in eukaryote_list ]]
c = [x for x in a if True in [y.lower()  in x.lower() for y in remove_list ]]
d = [x for x in a if True not in [y.lower()  in x.lower() for y in remove_list ]]
e = [x for x in d if True not in [y.lower() in x.lower() for y in eukaryote_list ]]

#manual removing of beta vulgaris 
f = []
for i in range(len(b)):
    if b[i].split(' ')[0].lower() in elist:
        f.append(b[i])
    if b[i].split(' ')[0].lower() in ['beta']:
        if 'beta vulgaris' in b[i].split(' ')[0].lower() :
            f.append(b[i])

g = []
h = []
for i in range(len(f)):
    if f[i] not in c:
        g.append(f[i]) #final eukaryotic descriptions
    else:
        h.append(f[i]) #final prokaryotic descriptions
        
# manually add some specific tags that were missed to the eukaryotic
        
g = g + ['Beta vulgaris subsp. vulgaris TPP riboswitch (THI element)',
 'Beta vulgaris subsp. vulgaris yybP-ykoY manganese riboswitch',
 'Dendroctonus ponderosae (mountain pine beetle) TPP riboswitch (THI element)',
 'Fibulorhizoctonia sp. CBS 109695 TPP riboswitch (THI element)',
 'Hyphopichia burtonii NRRL Y-1933 TPP riboswitch (THI element)',
 'Neonectria ditissima TPP riboswitch (THI element)',
 'Ophiocordyceps unilateralis TPP riboswitch (THI element)',
 'Paracoccidioides brasiliensis Pb01 TPP riboswitch (THI element)',
 'Paracoccidioides brasiliensis Pb18 TPP riboswitch (THI element)',
 'Phaeoacremonium minimum UCRPA7 TPP riboswitch (THI element)',
 'TPP-specific riboswitch from Arabidopsis thaliana (PDB 3D2G, chain B)']

#CHECK THAT ALL DESCRIPTIONS ARE LABELED, this should be 100% if done correctly
print('Percent of descriptions that are labeled by keywords:')
print(len(set(g + c)) / len(set(a)) *100)

# make a boolean list to add to the data frame
eukaryotic = []
for i in range(len(a)):
    if a[i] in g:
        eukaryotic.append(1)
    if a[i] in c:
        eukaryotic.append(0)
RS_df['EUKARYOTIC'] = eukaryotic


# write the final RS data frame to use for machine learning
RS_df.to_csv(RS_csv_name, index=False)
print('final database size RS: ')
print(RS_df.shape)
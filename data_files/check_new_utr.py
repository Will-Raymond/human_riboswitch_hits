# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 15:29:53 2023

@author: willi
"""
import pandas as pd
from Bio import SeqIO
from Bio import pairwise2
from tqdm import tqdm
import numpy as np

aln = np.load('newoldaln.npy')

#@title functions
import itertools as it
ns_dict = {'m':'a','w':'a','r':'g','y':'t','k':'g','s':'g','w':'a','h':'a','n':'a','x':'a'}

def get_mfe_nupack(seq):

  model1 = Model(material='rna', celsius=37)
  example_hit = seq
  example_hit = Strand(example_hit, name='example_hit')
  t1 = Tube(strands={example_hit: 1e-8}, complexes=SetSpec(max_size=1), name='t1')
  hit_results = tube_analysis(tubes=[t1], model=model1,
      compute=['pairs', 'mfe', 'sample', 'ensemble_size'],
      options={'num_sample': 100}) # max_size=1 default
  mfe = hit_results[list(hit_results.complexes.keys())[0]].mfe
  return mfe, hit_results


def clean_seq(seq):
    '''
    clean the sequences to lowercase only a, u, g, c
    '''
    seq = seq.lower()
    for key in ns_dict.keys():
        seq = seq.replace(key,ns_dict[key])

    seq = seq.replace('t','u')
    return seq.lower()

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

def get_gc(seq):
  return (seq.count('g') + seq.count('c'))/len(seq)



utr_df = pd.read_csv('C:/Users/willi/Documents/GitHub/human_riboswitch_hits/data_files/5primeUTR_final_db_small.csv')

r = [];
for record in SeqIO.parse("C:/Users/willi/Downloads/Homo_sapiens.GRCh38.107.utrs/Homo_sapiens.GRCh38.107.fa", "fasta"):
    r.append(record)
    
    
    
    
    
r2 = []
genes = []
for record in r:
    if 'five_prime_utr' in record.id:
        r2.append(record)    
        genes.append(record.id.split('|')[3])
r = r2


new_utr_data = pd.DataFrame(columns=['GENE', 'REFSEQ', 'ID', 'SEQ','CCDS_ID', 'CCDS', 'ALGN_IND', 'STARTPLUS25','NUPACK_25','NUPACK_25_MFE'])

new_utr_data['GENE'] = genes
new_utr_data['ID'] = [record.id.split('|')[2] for record in r2]
new_utr_data['SEQ'] = [clean_seq(str(record.seq)) for record in r2]

# they split their download by exons.... recombine these 
unique_ensemble_ids = new_utr_data['ID'].unique()
corrected_df_ids = []
corrected_df_gene = []
corrected_df_seq = []
for i in range(len(unique_ensemble_ids)):
    sub_df = new_utr_data[new_utr_data['ID'] == unique_ensemble_ids[i]]
    corrected_df_ids.append(unique_ensemble_ids[i])
    corrected_df_gene.append(sub_df['GENE'].iloc[0])    
    corrected_df_seq.append(''.join(sub_df['SEQ'].values))

new_utr_data = pd.DataFrame(columns=['GENE', 'REFSEQ', 'ID', 'SEQ','CCDS_ID', 'CCDS', 'ALGN_IND', 'STARTPLUS25','NUPACK_25','NUPACK_25_MFE'])
new_utr_data['GENE'] = corrected_df_gene
new_utr_data['ID'] = corrected_df_ids
new_utr_data['SEQ'] = corrected_df_seq


1/0

ccds_attributes = pd.read_csv('C:/Users/willi/Documents/GitHub/RS_redo/data_files/ccds_11.5.23/CCDS.20221027.txt',delimiter='\t')
r = []
for record in tqdm(SeqIO.parse('C:/Users/willi/Documents/GitHub/RS_redo/data_files/ccds_11.5.23/CCDS_nucleotide.20221027.fna','fasta')):
    r.append(record)

for record in tqdm(r):
    ccds_list = ccds_attributes['ccds_id'].values
    ccds_id = record.id.split('|')[0]
    if ccds_id in ccds_list:
        if ccds_attributes[ccds_attributes['ccds_id'] == ccds_id]['ccds_status'].values[0] != 'Withdrawn':
            
            gene = ccds_attributes[ccds_attributes['ccds_id'] == ccds_id]['gene'].values[0] 
            if len(new_utr_data[new_utr_data['GENE'] == gene]['CCDS_ID']) != 0:
                #new_utr_data[new_utr_data['GENE'] == gene[0]]['CCDS_ID'] = record.id.split('|')[0]
                for ind in new_utr_data[new_utr_data['GENE'] == gene]['CCDS'].index:
                    new_utr_data.iloc[ind,5] = str(record.seq).lower().replace('t','u')
                    new_utr_data.iloc[ind,4] = ccds_id
                    #s2 = new_utr_data.iloc[ind,3]
                    #s1 = str(record.seq).lower().replace('t','u')[:len(s2)]
                    #aln = pairwise2.align.globalxx(s1, s2)[0].score/max(len(s1), len(s2))
                    #new_utr_data.iloc[ind,6] = aln

for i in range(len(new_utr_data)):
    if not pd.isnull(new_utr_data.iloc[i,4]):
        new_utr_data.iloc[i]


n_mismatch = 0
n_matches = 0
n_5 = 0
seqs = []

new_utr_data = pd.read_csv()


import numpy as np
seqalns = []
seqalns2 = []
for j in tqdm(range(len(utr_df))):
#for j in tqdm(range(1000)):
    
    s1 = utr_df['SEQ'].iloc[j]
    g = utr_df['GENE'].iloc[j]
    
    if g in genes:
        new_seqs = [str(r2[i].seq).lower().replace('t','u') for i in range(len(genes)) if genes[i] == g]
        alns = []
        alns2 = []
        for k in range(len(new_seqs)):
            s2 = new_seqs[k]
            #alns.append(pairwise2.align.globalxx(s1,s2)[0].score/max(len(s1), len(s2)))
            alns2.append(pairwise2.align.globalxx(s1,s2)[0].score/len(s1))
       # seqalns.append(max(alns))
        seqalns2.append(max(alns2))
    if j%10000 == 0:
        np.save('newoldaln2.npy', seqalns)
            
            
            

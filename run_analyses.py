# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 12:52:00 2024

@author: Dr. William Raymond
"""
###############################################################################
# Description
###############################################################################

'''
This file runs all the analyses for the paper:

Identification of potential riboswitch elements in Homo
Sapiens mRNA 5’UTR sequences using Positive-Unlabeled
machine learning

By Dr. William Raymond, Dr. Jacob DeRoo, and Dr. Brian Munsky 


File sections:
    Data generation

'''

###############################################################################
# OPTIONS
###############################################################################

# filenames for the RS dataset and UTR dataset
RS_fname = './data_files/RS_final_with_euk2.csv'
UTR_fname = './data_files/5primeUTR_final_db_3.csv'


###############################################################################
# Imports
###############################################################################

import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import json

from sklearn import mixture
from pulearn import ElkanotoPuClassifier, BaggingPuClassifier
from sklearn.svm import SVC
from joblib import dump, load
from tqdm import tqdm

from rs_functions import * #functions to do feature extraction and other things

import pulearn
print('Using PUlearn version:')
print(pulearn.__version__)

###############################################################################
# Sanatize data
###############################################################################
print('______________________________________')
print('Loading and sanatizing data files....')


# Get data files
# first we have the UTR data file, UTR data file is built from "make_initial_UTR_csv.py"
# Using the 5UTRaspic.Hum.fasta
# most important headers: GENE	ID	SEQ	CCDS_ID	STARTPLUS25	NUPACK_25	NUPACK_25_MFE
# Gene - uniprot Gene ID
# ID - original UTRdb 1.0 ID sequence, defunct
# SEQ - Cleaned sequence
# CCDS_ID - CCDS_id for the 25 nucleotides added to the 5'UTR
# STARTPLUS25 - sequence of the 5'UTR + 25 nt
# NUPACK_25 - nupack MFE dot structure for 100 foldings of the dot structure
# NUPACK_25_MFE - energy of the MFE structure

UTR_db = pd.read_csv(UTR_fname)
print('UTR data file loaded: %s'%UTR_fname)

# Riboswitch data file
# ID - ID within RNA central
# DESC - Text description scraped from this given entry
# EUKARYOTIC - 1 or 0 is this a eukaryotic sequence
# LIGAND - Ligand scraped for this given entry
# SEQ - Sequence of the entry
# NUPACK_DOT - NUPACK dot structure of 100 foldings
# NUPACK_MFE - NUPACK mfe of the structure of 100 foldings
# Kmers aaa.... kmer counts for each triplet

RS_db = pd.read_csv(RS_fname)
print('RS data file loaded: %s'%RS_fname)

## We have to apply some patches to the UTR data file to remove 3'UTR sequences and duplicate IDs
## since some UTRs have multiple isoforms and UTRdb didnt handle this

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
  UTR_db.iloc[ids_to_update[id][i],4] = id + '-' + str(i)

## patch out a typo space in guanidine
for i in range(len(RS_db)):
  if RS_db.iloc[i,2] == 'guanidine ':
    RS_db.iloc[i,2] = 'guanidine'

## patch to combine adocobalamin with cobalamin since these are very similar
for i in range(len(RS_db)):
  if RS_db.iloc[i,2] == 'adocbl':
    RS_db.iloc[i,2] = 'cobalamin'
    
print('Riboswitch database size:' + str(RS_db.shape))
print('UTR database size:' + str(UTR_db.shape))




###############################################################################
# FEATURE EXTRACTION 
###############################################################################
# This block generates X_RS and X_UTR for 74 features used for the machine learning ensembles

include_mfe = True 
include_3mer = True 
include_dot = True 
include_gc = True 
be = BEARencoder(); # Custom bear encoder for counting the dot structural features.

ccds_length = 'STARTPLUS25' #select the start plus 25 sequences to use
#max_mfe = np.min([np.min(UTR_db['NUPACK_25_MFE']), np.min(RS_df['NUPACK_MFE'])])

print('Using:')
print(ccds_length)

#########################
# UTRs
#########################
print('processing UTRs......')
X_UTR = np.zeros([len(UTR_db),66+8])
dot_UTR = []
ids_UTR = []
k = 0

# get the size first X_UTR
X_utr_size = 0
for i in range(len(UTR_db)):
  if not pd.isna(UTR_db[ccds_length].iloc[i]):
    if len(clean_seq(UTR_db[ccds_length].iloc[i])) > 25:
      X_utr_size+=1

# fill up the X_UTR with extracted features
X_UTR =np.zeros([X_utr_size, 66+8])

for i in tqdm(range(len(UTR_db))):
  if not pd.isna(UTR_db[ccds_length].iloc[i]):
    seq = clean_seq(UTR_db[ccds_length].iloc[i])
    if len(seq) > 25:
      kmerf = kmer_freq(seq)
      X_UTR[k,:64] = kmerf/np.sum(kmerf)
      X_UTR[k,64] = UTR_db['NUPACK_25_MFE'].iloc[i]
      X_UTR[k,65] = get_gc(seq)
      ids_UTR.append(UTR_db['ID'].iloc[i])
      dot_UTR.append(UTR_db['NUPACK_25'].iloc[i])
      X_UTR[k,-8:] = be.annoated_feature_vector(UTR_db['NUPACK_25'].iloc[i], encode_stems_per_bp=True)
      k+=1



#########################
# RS full
#########################
print('processing all RS......')
full_RS_df = RS_db
print(len(full_RS_df))

X_RS_full = np.zeros([len(full_RS_df),66+8])
dot_RS_full = []
ids_RS_full = []
k = 0

# get the size first X_RS
X_RS_size = 0
for i in range(len(full_RS_df)):
  seq = clean_seq(full_RS_df['SEQ'].iloc[i])
  if len(seq) > 25:
    X_RS_size+=1

X_RS_full = np.zeros([X_RS_size, 66+8])

# fill up the X_UTR with extracted features
for i in tqdm(range(len(full_RS_df))):
  seq = clean_seq(full_RS_df['SEQ'].iloc[i])
  if len(seq) > 25:
    seq = clean_seq(full_RS_df['SEQ'].iloc[i])
    kmerf = kmer_freq(seq)
    X_RS_full[k,:64] = kmerf/np.sum(kmerf)
    X_RS_full[k,64] = full_RS_df['NUPACK_MFE'].iloc[i]
    X_RS_full[k,65] = get_gc(seq)
    ids_RS_full.append(full_RS_df['ID'].iloc[i])
    dot_RS_full.append(full_RS_df['NUPACK_DOT'].iloc[i])
    X_RS_full[k,-8:] = be.annoated_feature_vector(full_RS_df['NUPACK_DOT'].iloc[i], encode_stems_per_bp=True)
    k+=1


# Get the maximum values (minimum for mfe) across thte dataset to normalize against
max_mfe = min(np.min(X_RS_full[:,64]),np.min(X_UTR[:,64]))
X_RS_full[:,64] = X_RS_full[:,64]/max_mfe
X_UTR[:,64] = X_UTR[:,64]/max_mfe
max_ubs = np.max([np.max(X_UTR[:,66]),np.max(X_RS_full[:,66])])
max_bs = np.max([np.max(X_UTR[:,67]),np.max(X_RS_full[:,67])])
max_ill = np.max([np.max(X_UTR[:,68]),np.max(X_RS_full[:,68])])
max_ilr = np.max([np.max(X_UTR[:,69]),np.max(X_RS_full[:,69])])
max_lp = np.max([np.max(X_UTR[:,70]),np.max(X_RS_full[:,70])])
max_lb = np.max([np.max(X_UTR[:,71]),np.max(X_RS_full[:,71])])
max_rb = np.max([np.max(X_UTR[:,72]),np.max(X_RS_full[:,72])])

# normalize both data sets by their largest values
X_UTR[:,66] = X_UTR[:,66]/max_ubs
X_UTR[:,67] = X_UTR[:,67]/max_bs
X_UTR[:,68] = X_UTR[:,68]/max_ill
X_UTR[:,69] = X_UTR[:,69]/max_ilr
X_UTR[:,70] = X_UTR[:,70]/max_lp
X_UTR[:,71] = X_UTR[:,71]/max_lb
X_UTR[:,72] = X_UTR[:,72]/max_rb

X_RS_full[:,66] = X_RS_full[:,66]/max_ubs
X_RS_full[:,67] = X_RS_full[:,67]/max_bs
X_RS_full[:,68] = X_RS_full[:,68]/max_ill
X_RS_full[:,69] = X_RS_full[:,69]/max_ilr
X_RS_full[:,70] = X_RS_full[:,70]/max_lp
X_RS_full[:,71] = X_RS_full[:,71]/max_lb
X_RS_full[:,72] = X_RS_full[:,72]/max_rb



#########################
# RS ligand specific
#########################

# now we will split up the X_RS by ligand type for structural cross validation

print('Generating RS Ligand DFs......')
def make_ligand_df(df, ligand):
  witheld_df =  RS_db[RS_db['LIGAND'] ==ligand]
  X_witheld = np.zeros([len(witheld_df),66+8])
  dot_witheld = []
  ids_witheld = []
  k = 0
  for i in tqdm(range(len(witheld_df))):
    seq = clean_seq(witheld_df['SEQ'].iloc[i])
    if len(seq) > 25:
      if i == 0:
        X_witheld = np.zeros([1,66+8])
      else:
        X_witheld = np.vstack( [X_witheld, np.zeros([1,66+8]) ])

      seq = clean_seq(witheld_df['SEQ'].iloc[i])
      kmerf = kmer_freq(seq)
      X_witheld[k,:64] = kmerf/np.sum(kmerf)
      X_witheld[k,64] = witheld_df['NUPACK_MFE'].iloc[i]/max_mfe
      X_witheld[k,65] = get_gc(seq)
      ids_witheld.append(witheld_df['ID'].iloc[i])
      dot_witheld.append(witheld_df['NUPACK_DOT'].iloc[i])
      X_witheld[k,-8:] = be.annoated_feature_vector(witheld_df['NUPACK_DOT'].iloc[i], encode_stems_per_bp=True)


      X_witheld[k,66] = X_witheld[k,66]/max_ubs
      X_witheld[k,67] = X_witheld[k,67]/max_bs
      X_witheld[k,68] = X_witheld[k,68]/max_ill
      X_witheld[k,69] = X_witheld[k,69]/max_ilr
      X_witheld[k,70] = X_witheld[k,70]/max_lp
      X_witheld[k,71] = X_witheld[k,71]/max_lb
      X_witheld[k,72] = X_witheld[k,72]/max_rb

      k+=1
  return X_witheld, ids_witheld, dot_witheld

set(RS_db['LIGAND'])
witheld_ligands = ['cobalamin', 'guanidine', 'TPP','SAM','glycine','FMN','purine','lysine','fluoride','zmp-ztp',]

RS_df = RS_db[~RS_db['LIGAND'].isin( witheld_ligands)]
print(len(RS_df))

ligand_dfs = []
for i in range(len(witheld_ligands)):
  print('making %s....'%witheld_ligands[i])
  ligand_dfs.append(make_ligand_df(RS_db, witheld_ligands[i]))

X_RS = np.zeros([len(RS_df),66+8])
dot_RS = []
ids_RS = []
k = 0

X_RS_size = 0
for i in range(len(RS_df)):
  seq = clean_seq(RS_df['SEQ'].iloc[i])
  if len(seq) > 25:
    X_RS_size+=1

X_RS = np.zeros([X_RS_size, 66+8])

for i in tqdm(range(len(RS_df))):
  seq = clean_seq(RS_df['SEQ'].iloc[i])
  if len(seq) > 25:

    seq = clean_seq(RS_df['SEQ'].iloc[i])
    kmerf = kmer_freq(seq)
    X_RS[k,:64] = kmerf/np.sum(kmerf)
    X_RS[k,64] = RS_df['NUPACK_MFE'].iloc[i]/max_mfe
    X_RS[k,65] = get_gc(seq)
    ids_RS.append(RS_df['ID'].iloc[i])
    dot_RS.append(RS_df['NUPACK_DOT'].iloc[i])
    X_RS[k,-8:] = be.annoated_feature_vector(RS_df['NUPACK_DOT'].iloc[i], encode_stems_per_bp=True)

    X_RS[k,66] = X_RS[k,66]/max_ubs
    X_RS[k,67] = X_RS[k,67]/max_bs
    X_RS[k,68] = X_RS[k,68]/max_ill
    X_RS[k,69] = X_RS[k,69]/max_ilr
    X_RS[k,70] = X_RS[k,70]/max_lp
    X_RS[k,71] = X_RS[k,71]/max_lb
    X_RS[k,72] = X_RS[k,72]/max_rb

    k+=1
    
save_feature_sets = False
if save_feature_sets:
    maxes = np.array([max_mfe, max_ubs, max_bs, max_ill, max_ilr, max_lp, max_lb, max_rb])
    np.save('feature_maxes.npy', maxes)
    np.save('X_UTR.npy',X_UTR)
    np.save('X_RS.npy',X_RS_full)

###############################################################################
# FEATURE PLOTS
# Plots of various aspects of the X_UTR and X_RS we just constructed
###############################################################################


#@title plotting options
from cycler import cycler
########################################
dark = False
if not dark:
    colors = ['#ef476f', '#073b4c','#06d6a0','#7400b8','#073b4c', '#118ab2',]
else:
    plt.style.use('dark_background')
    plt.rcParams.update({'axes.facecolor'      : '#131313'  ,
'figure.facecolor' : '#131313' ,
'figure.edgecolor' : '#131313' ,
'savefig.facecolor' : '#131313'  ,
'savefig.edgecolor' :'#131313'})


    colors = ['#118ab2','#57ffcd', '#ff479d', '#ffe869','#ff8c00','#04756f']

font = {
        'weight' : 'bold',
        'size'   : 12}

save = False

plt.rcParams.update({'font.size': 12, 'font.weight':'bold' }   )
plt.rcParams.update({'axes.prop_cycle':cycler(color=colors)})

plt.rcParams.update({'axes.prop_cycle':cycler(color=colors)})
plt.rcParams.update({'axes.prop_cycle':cycler(color=colors)})


plt.rcParams.update({'xtick.major.width'   : 2.8 })
plt.rcParams.update({'xtick.labelsize'   : 12 })



plt.rcParams.update({'ytick.major.width'   : 2.8 })
plt.rcParams.update({'ytick.labelsize'   : 12})

plt.rcParams.update({'axes.titleweight'   : 'bold'})
plt.rcParams.update({'axes.titlesize'   : 10})
plt.rcParams.update({'axes.labelweight'   : 'bold'})
plt.rcParams.update({'axes.labelsize'   : 12})

plt.rcParams.update({'axes.linewidth':2.8})
plt.rcParams.update({'axes.labelpad':8})
plt.rcParams.update({'axes.titlepad':10})
plt.rcParams.update({'figure.dpi':120})



###############################################################################
#   Pie Chart of ligand representation
##############################################################################

amino_acids = ['arginine','histidine','lysine','aspartate','glutamine','serine','threonine','asparagine','cystine','glycine','proline',
               'alanine','valine','isoleucine','leucine','methionine','phenylalanine','tyrosine','tryptophan']
aa = False

ligand_list = RS_db[RS_db['ID'].isin(ids_RS_full)]['LIGAND'].values.tolist()

count_list = ligand_list
ligand_names = list(set(count_list))
counts = np.array([count_list.count(x) for x in ligand_names])
idx = np.argsort(counts)[::-1]

sorted_counts = counts[idx]
sorted_names = [ligand_names[i] for i in idx.tolist()]
explode = [0.02]*len(sorted_names)
sub1 = sorted_counts/np.sum(sorted_counts) < .01
colors2 = cm.Spectral_r(np.linspace(.05,.95,len(sorted_names)))

plt.figure(dpi=300)
_,f,t = plt.pie(sorted_counts, labels = sorted_names, explode = explode, autopct='%1.1f%%',
        shadow=False, startangle=90, colors=colors2, labeldistance =1.05, pctdistance=.53, textprops={'fontsize': 7})

missing_labels = []
subsum = 0
for i in range(len( t)):
    txt = t[i]
    label = f[i]

    if float(txt._text[:-1]) < 2:
        txt.set_visible(False)
        label.set_visible(False)
        missing_labels.append(label._text)
        subsum += float(txt._text[:-1])


centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.text(0.17,.4,'16.9%',color='r', size=7)
s = (.74/.4185)
xx,yy = .614,.88
plt.plot([xx,yy], [xx/s,yy/s],'r',alpha=.5)
plt.plot([0,0], [.71,1.01],'r',alpha=.5)

plt.text(.6,1,'Other (33 types)',color='r')
plt.text(.7,.85,'<2% each',color='r')

plt.text(1.1,-1.1,'N = %0.0f'%len(count_list))
plt.title('RS ligand representation')


###############################################################################
# Length distribution of the X_RS and X_UTR
###############################################################################
nbins = 40
UTR_lens = np.array([len(x) for x in dot_UTR])
RS_lens = np.array([len(x) for x in dot_RS_full])
x,bins = np.histogram(RS_lens, bins=nbins)
x2,_ = np.histogram(UTR_lens,bins=bins)
plt.figure(dpi=300)
plt.hist(RS_lens, bins=bins, density=True, alpha=1, histtype='step', lw=3)
plt.hist(UTR_lens,bins=bins, density=True, alpha=1, histtype='step', lw=3)
plt.xlim([0,325])
plt.xlabel('Length (NT)')
plt.ylabel('Probability')
plt.legend(['RS   n=%s'%str(len(RS_lens)),'UTR n=%s'%str(len(UTR_lens))], loc='upper left')
plt.title('Length distributions')


###############################################################################
# Sequence comparison plot of the 3-mers
###############################################################################
plt.errorbar(np.linspace(0,63,64),np.mean(X_RS[:,:64],axis=0), yerr=np.std(X_RS[:,:64],axis=0),ls='', marker='o', capsize=1)
plt.errorbar(np.linspace(0,63,64),np.mean(X_UTR[:,:64],axis=0), yerr=np.std(X_UTR[:,:64],axis=0),ls='', marker='o',  capsize=1)
plt.title('3-mer distribution comparisons')
plt.legend(['RS','UTR'])

plt.figure()
plt.errorbar(np.linspace(0,9,10),np.mean(X_RS[:,-10:],axis=0), yerr=np.std(X_RS[:,-10:],axis=0),ls='', marker='o',  capsize=1)
plt.errorbar(np.linspace(0,9,10),np.mean(X_UTR[:,-10:],axis=0), yerr=np.std(X_UTR[:,-10:],axis=0),ls='', marker='o', capsize=1)
plt.title('Other feature distribution comparisons')
plt.legend(['RS','UTR'])


###############################################################################
# KS distances of the X_RS and X_UTR
###############################################################################
from scipy.stats import ks_2samp

kses = [ks_2samp(X_RS[:,i], X_UTR[:,i])[0] for i in range(66+8)]
plt.figure()
plt.bar(np.linspace(0,65+8,66+8),kses)
plt.ylabel('KS distance')
plt.xlabel('Feature')

###############################################################################
# PCA of the X_RS and X_UTR
###############################################################################
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
p = pca.fit(np.vstack([X_UTR, X_RS]))
print(pca.explained_variance_ratio_)
p = pca.fit(np.vstack([X_UTR, X_RS_full]))
x_utr_t = p.transform(X_UTR)
x_rs_t = p.transform(X_RS_full)
plt.figure()
plt.scatter(x_utr_t[:,0], x_utr_t[:,1], s=5,alpha=.2)
plt.scatter(x_rs_t[:,0], x_rs_t[:,1],s=5, alpha=.2)
plt.legend(['UTR','RS'])
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.3f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.3f}%)')




###############################################################################
# TRAIN CLASSIFIER ENSEMBLE
###############################################################################

# The ensemble is trained in 3 parts: The "Other" set first (ligands with <2% representation)
# single drop out ligands second, double drop out ligands last.

# OPTIONS:
retrain  = False #retrain the ensemble
save  = False  #save the ensemble after retraining
model_name = "EKmodel_witheld_w_struct_features_9_26" #name for the model files, they will add "_ligand" to the end for each

# Train the "Other" classifier
# preallocate the accuracies, and predicted outputs of the training / validation
witheld_acc_other = []
RS_acc_other = []
UTR_acc_other = []
predicted_RSs_other = []
predicted_withelds_other = []
predicted_UTRs_other = []
estimators_other = []

for i in tqdm(range(1)):

  witheld_ligands[:i]
  X = np.vstack([X_UTR,] + [x[0] for x in ligand_dfs]) 
  X_witheld = X_RS

  if retrain:
    svc = SVC(C=10, kernel='rbf', gamma=0.4, probability=True)
    pu_estimator = ElkanotoPuClassifier(estimator=svc, hold_out_ratio=0.2)
    y = np.zeros(len(X))
    y[len(X_UTR):] = 1
    pu_estimator.fit(X, y)
  else:
    pu_estimator =load('./elkanoto_models/%s_%s.joblib'%(model_name, 'other'))


  X_t = np.vstack([ X_RS, ] + [x[0] for x in ligand_dfs]  )

  predicted_RS_other = pu_estimator.predict_proba(X_t)
  predicted_witheld_other = pu_estimator.predict_proba(X_witheld)
  predicted_UTR_other = pu_estimator.predict_proba(X_UTR)

  UTR_acc_other.append( np.sum((predicted_UTR_other[:,1] < .5))/len(X_UTR) )
  RS_acc_other.append( np.sum((predicted_RS_other[:,1] > .5))/len(X_t) )
  witheld_acc_other.append( np.sum((predicted_witheld_other[:,1] > .5))/len(X_witheld)  )

  predicted_RSs_other.append(predicted_RS_other)
  predicted_withelds_other.append(predicted_witheld_other)
  predicted_UTRs_other.append(predicted_UTR_other)
  if retrain:
    if save:
      dump(pu_estimator,'./elkanoto_models/%s_%s.joblib'%(model_name, 'other'))
  estimators_other.append(pu_estimator)

# Single drop out classifiers
witheld_acc = []
RS_acc = []
UTR_acc = []

predicted_RSs = []
predicted_withelds = []
predicted_UTRs = []
estimators = []

for i in tqdm(range(len(witheld_ligands))):

  witheld_ligands[:i]
  X = np.vstack([X_UTR, X_RS, ] + [x[0] for x in ligand_dfs[:i]] + [x[0] for x in ligand_dfs[i+1:]] )

  X_witheld = ligand_dfs[i][0]

  if retrain:
    svc = SVC(C=10, kernel='rbf', gamma=0.4, probability=True)
    pu_estimator = ElkanotoPuClassifier(estimator=svc, hold_out_ratio=0.2)
    y = np.zeros(len(X))
    y[len(X_UTR):] = 1
    pu_estimator.fit(X, y)
  else:
    pu_estimator =load('./elkanoto_models/%s_%s.joblib'%(model_name, witheld_ligands[i]))


  X_t = np.vstack([ X_RS, ] + [x[0] for x in ligand_dfs[:i]] + [x[0] for x in ligand_dfs[i+1:]] )

  predicted_RS = pu_estimator.predict_proba(X_t)
  predicted_witheld = pu_estimator.predict_proba(X_witheld)
  predicted_UTR = pu_estimator.predict_proba(X_UTR)

  UTR_acc.append( np.sum((predicted_UTR[:,1] < .5))/len(X_UTR) )
  RS_acc.append( np.sum((predicted_RS[:,1] > .5))/len(X_t) )
  witheld_acc.append( np.sum((predicted_witheld[:,1] > .5))/len(X_witheld)  )

  predicted_RSs.append(predicted_RS)
  predicted_withelds.append(predicted_witheld)
  predicted_UTRs.append(predicted_UTR)
  if retrain:
    if save:
      dump(pu_estimator,'./elkanoto_models/%s_%s.joblib'%(model_name, witheld_ligands[i]))
  estimators.append(pu_estimator)


# Double drop out ligands

pairs = [('SAM', 'cobalamin'), ('TPP','glycine'),('SAM','TPP'),('glycine','cobalamin'),('TPP','cobalamin'),
  ('FMN','cobalamin'),('FMN','TPP'),('FMN','SAM'),('FMN','glycine')]


witheld_acc_2 = []
RS_acc_2 = []
UTR_acc_2 = []
predicted_RSs_2 = []
predicted_withelds_2 = []
predicted_UTRs_2 = []
estimators_2 = []


for i in tqdm(range(len(pairs))):
  witheld_1 = pairs[i][0]
  witheld_2 = pairs[i][1]

  ind_1 = witheld_ligands.index(witheld_1)
  ind_2 = witheld_ligands.index(witheld_2)



  witheld_ligands[:i]
  X = np.vstack([X_UTR, X_RS, ] + [ligand_dfs[i][0] for i in range(len(ligand_dfs)) if i not in [ind_1,ind_2]])

  X_witheld = np.vstack([ligand_dfs[ind_1][0], ligand_dfs[ind_2][0]])

  if retrain:
    svc = SVC(C=10, kernel='rbf', gamma=0.4, probability=True)
    pu_estimator = ElkanotoPuClassifier(estimator=svc, hold_out_ratio=0.2)
    y = np.zeros(len(X))
    y[len(X_UTR):] = 1
    pu_estimator.fit(X, y)
  else:
    pu_estimator =load('./elkanoto_models/%s_%s_%s.joblib'%(model_name,witheld_1, witheld_2))

  X_t = np.vstack([ X_RS, ] + [ligand_dfs[i][0] for i in range(len(ligand_dfs)) if i not in [ind_1,ind_2]])

  predicted_RS_2 = pu_estimator.predict_proba(X_t)
  predicted_witheld_2 = pu_estimator.predict_proba(X_witheld)
  predicted_UTR_2 = pu_estimator.predict_proba(X_UTR)

  UTR_acc_2.append( np.sum((predicted_UTR_2[:,1] < .5))/len(X_UTR) )
  RS_acc_2.append( np.sum((predicted_RS_2[:,1] > .5))/len(X_t) )
  witheld_acc_2.append( np.sum((predicted_witheld_2[:,1] > .5))/len(X_witheld)  )

  predicted_RSs_2.append(predicted_RS_2)
  predicted_withelds_2.append(predicted_witheld_2)
  predicted_UTRs_2.append(predicted_UTR_2)
  if retrain:
    if save:
      dump(pu_estimator,'./elkanoto_models/%s_%s_%s.joblib'%(model_name,witheld_1, witheld_2))
  estimators_2.append(pu_estimator)



#combine the ensemble classifiers into a list
ensemble = estimators + estimators_2 + estimators_other

predicted_RS_all = predicted_RSs + predicted_RSs_2 + predicted_RSs_other
predicted_witheld_all = predicted_withelds + predicted_withelds_2 + predicted_withelds_other
predicted_UTR_all = predicted_UTRs + predicted_UTRs_2 + predicted_UTRs_other
UTR_acc_all = UTR_acc + UTR_acc_2 + UTR_acc_other
RS_acc_all = RS_acc + RS_acc_2 + RS_acc_other
witheld_acc_all = witheld_acc + witheld_acc_2 + witheld_acc_other

#normalization vector for the outputs to the max of the training
#ensemble_norm = np.load('./%s'%model_norm)



###############################################################################
# LEARN SINGULAR FEATURE IMPORTANCES
###############################################################################

redo_importance = False
if redo_importance:
    #PERMUTATION IMPORTANCE
    from sklearn.inspection import permutation_importance
    
    rs = []
    X = np.vstack([X_RS_full[:5000], X_UTR[:5000]])
    y = np.zeros(len(X))
    y[len(X_UTR[:5000]):] = 1
    for i in range(len(ensemble)):
        print(i)
        rs.append(permutation_importance(ensemble[i], X, y,
                                   n_repeats=10,
                                   random_state=0))
else:
    importances = np.load('importances.npy')



#plt.boxplot(r.importances.T, vert=False, showfliers=False); plt.xlabel('accuracy loss'); plt.ylabel('feature')
plt.figure(figsize=(5,10), dpi=300); #plt.boxplot(r.importances.T, vert=False, showfliers=False); plt.xlabel('accuracy loss'); plt.ylabel('feature'); plt.yticks(rotation=90, fontsize=6); plt.grid(True)
import itertools
#importances = np.array([x.importances.T for x in rs])
plt.figure(figsize=(5,10),dpi=300);
from matplotlib import cm
for i in range(20):
    plt.boxplot(importances[i], vert=False, showfliers=False, boxprops={'alpha':.8, 'color':cm.viridis(i/20)},
                capprops={'alpha':.8, 'color':cm.viridis(i/40)},
                whiskerprops={'alpha':.8, 'color':cm.viridis(i/40)},
                medianprops={'alpha':.8, 'color':cm.viridis(i/40)},
                meanprops={'alpha':.8, 'color':cm.viridis(i/40)},
                ); 
plt.xlabel('accuracy loss'); plt.ylabel('feature'); plt.yticks(rotation=90, fontsize=6); plt.grid(True)


ls = ['',] + [''.join(x) for x in itertools.product(['a','c','u','g'], repeat=3)] + ['GC%', 'M=FE', 'UBS', 'BS', 'LL','RL','L','RB','LB','%UP']


ax = plt.gca()
ax.set_yticks([x for x in range(75)])
ax.set_yticklabels(ls)
ax.yaxis.set_tick_params(rotation=0)



plt.figure(figsize=(5,10),dpi=300);
from matplotlib import cm

i=0
plt.boxplot(importances.reshape(200,74), vert=False, showfliers=False, boxprops={'alpha':1, 'color':cm.viridis(i/20)},
            capprops={'alpha':1, 'color':cm.viridis(i/20)},
            whiskerprops={'alpha':1, 'color':cm.viridis(i/20)},
            medianprops={'alpha':1, 'color':cm.viridis(i/20)},
            meanprops={'alpha':1, 'color':cm.viridis(i/20)},
            ); 
plt.xlabel('accuracy loss'); plt.ylabel('feature'); plt.yticks(rotation=90, fontsize=6); plt.grid(True)


ls = ['',] + [''.join(x) for x in itertools.product(['a','c','u','g'], repeat=3)] + ['GC%', 'M=FE', 'UBS', 'BS', 'LL','RL','L','RB','LB','%UP']


ax = plt.gca()
ax.set_yticks([x for x in range(75)])
ax.set_yticklabels(ls)
ax.yaxis.set_tick_params(rotation=0)




plt.figure(figsize=(5,13),dpi=300);
from matplotlib import cm

i=0
plt.boxplot(importances.reshape(200,74), vert=False, showfliers=False, boxprops={'alpha':1, 'color':cm.viridis(i/20), 'label':'_nolegend_'},
            capprops={'alpha':1, 'color':cm.viridis(i/20), 'label':'_nolegend_'},
            whiskerprops={'alpha':1, 'color':cm.viridis(i/20), 'label':'_nolegend_'},
            medianprops={'alpha':1, 'color':cm.viridis(i/20), 'label':'_nolegend_'},
            meanprops={'alpha':1, 'color':cm.viridis(i/20), 'label':'_nolegend_'},
            ); 

colors = ['#ef476f', '#073b4c','#06d6a0','#7400b8','#073b4c', '#118ab2',]
for i in range(74):
    plt.scatter(np.mean(importances,axis=1)[:,i], [i+1]*20, marker='o', color=colors[0], s=4)
plt.xlabel('accuracy loss'); plt.ylabel('feature'); plt.yticks(rotation=90, fontsize=6); plt.grid(True)


ls = ['',] + [''.join(x) for x in itertools.product(['a','c','u','g'], repeat=3)] + ['GC%', 'MFE', 'UBS', 'BS', 'LL','RL','L','RB','LB','%UP']


ax = plt.gca()
ax.set_yticks([x for x in range(75)])
ax.set_yticklabels(ls, fontsize=9)
ax.yaxis.set_tick_params(rotation=0)
plt.title('Feature importance across the final ensemble')
plt.legend(['single classifier mean (n=10)'],  loc='lower left')
plt.plot([0,0],[1,74],'k-',lw=1)
plt.savefig('./SIfigure2.png')




###############################################################################
# PCA PLOT for SI
###############################################################################
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
#p = pca.fit(np.vstack([X_UTR, X_RS, X_RAND, X_EXONS]))

p = pca.fit(np.vstack([X_UTR, X_RS_full])) # X_RAND, X_EXONS]))
print(pca.explained_variance_ratio_)
x_utr_t = p.transform(X_UTR)
x_rs_t = p.transform(X_RS_full)


with open('./final_set.json', 'r') as f:
    UTR_hit_list = json.load(f)
    
    
x_hits = x_utr_t[[ids_UTR.index(x) for x in ids_UTR if x not in UTR_hit_list]]
thresh = .25
fx = lambda t: (np.sum((x_utr_t[:,0] > t) == 1) + np.sum((x_rs_t[:,0] <= t) == 1)) / (len(x_utr_t) + len(x_rs_t))
y = []
for t in np.linspace(-.25,5,1000):
    y.append(fx(t))


thresh = np.linspace(-.25,5,1000)[np.argmax(y)]

x,bins = np.histogram(x_utr_t[:,0], bins=30);
plt.stairs(x,edges=bins, lw=2, fill=False, color=colors[0]);
plt.stairs(x,edges=bins, lw=2, fill=True, alpha=.1, color=colors[0],label='_nolegend_');
x,_ = np.histogram(x_rs_t[:,0], bins=bins)
plt.stairs(x,edges=bins, lw=2, color=colors[1]);
x,_ = np.histogram(x_utr_t[[ids_UTR.index(x) for x in UTR_hit_list]][:,0], bins=bins)
plt.stairs(x,edges=bins, lw=2, alpha=1, color=colors[2]);
plt.stairs(x,edges=bins, lw=2, fill=True, alpha=.1,  color=colors[2], label='_nolegend_');

x,_ = np.histogram(x_hits[:,0], bins=bins)
#plt.stairs(x,edges=bins, lw=2, ls='--', color=colors[3]);
plt.plot([thresh, thresh], [0,10000000], 'k--')

plt.legend(['UTR', 'RS', '436'], bbox_to_anchor=(1.25, 1.05))
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.3f}%)')
ax = plt.gca()
ax.set_yscale('log')
plt.xlim([-.8, 1.4])
plt.ylim([1e0, 1e5])
plt.text(1, 10**4.5, 'UTR: ' + f'{np.sum((x_utr_t[:,0] > thresh) == 1)/len(x_utr_t)*100:.3f}'+ '%', color='k', size=7  )
plt.text(-.75, 10**4.5, 'UTR: ' +  f'{np.sum((x_utr_t[:,0] > thresh) == 0)/len(x_utr_t)*100:.3f}' +'%' , color='k',size=7  )

plt.text(1, 10**4.2, 'RS: ' +  f'{np.sum((x_rs_t[:,0] > thresh) == 1)/len(x_rs_t)*100:.3f}'+ '%', color='k',size=7  )
plt.text(-.75, 10**4.2, 'RS: ' + f'{np.sum((x_rs_t[:,0] > thresh) == 0)/len(x_rs_t)*100:.3f}' +'%' , color='k',size=7  )


plt.text(-.4, 10**.5, f'{436/len(x_utr_t)*100:.3f}'+ '% of UTRs', color=colors[2],size=7  )
plt.text(-.4, 10**2.3, f'{(1-436/len(x_utr_t))*100:.3f}' +'% of UTRs' , color=colors[0],size=7  )
plt.title('PCA separation of UTR and RS datasets')
plt.savefig('pca_2.svg')

###############################################################################
# Circle plot of the ensemble for the paper
###############################################################################

all_utr_predicitions = np.array(predicted_UTRs + predicted_UTRs_2 + predicted_UTRs_other).T


id_set = []

for i in range(len(predicted_UTRs)):
  subset = []
  matches = predicted_UTRs[i][:,1] > .95
  for j in range(len(matches)):
    if matches[j]:
      subset.append(ids_UTR[j])
  id_set.append(subset)
  if i == 0:
    total_set =set(subset)
  else:
    total_set = total_set.intersection(set(subset))

id_set = []

for i in range(len(predicted_UTRs_2)):
  subset = []
  matches = predicted_UTRs_2[i][:,1] > .95
  for j in range(len(matches)):
    if matches[j]:
      subset.append(ids_UTR[j])

  id_set.append(subset)
  if i == 0:

    total_set_2 =set(subset)
  else:
    total_set_2 = total_set_2.intersection(set(subset))

id_set = []

for i in range(len(predicted_UTRs_other)):
  subset = []
  matches = predicted_UTRs_other[i][:,1] > .95
  for j in range(len(matches)):
    if matches[j]:
      subset.append(ids_UTR[j])

  id_set.append(subset)
  if i == 0:

    total_set_3 =set(subset)
  else:
    total_set_3 = total_set_3.intersection(set(subset))



from matplotlib import colors as col
nn = 20
def truncate_colormap(cmapIn='jet', minval=0.0, maxval=1.0, n=100):
    '''truncate_colormap(cmapIn='jet', minval=0.0, maxval=1.0, n=100)'''
    cmapIn = plt.get_cmap(cmapIn)

    new_cmap = col.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmapIn.name, a=minval, b=maxval),
        cmapIn(np.linspace(minval, maxval, n)))
    return new_cmap

total_set = set(list(total_set) + list(total_set_2) + list(total_set_3))
fig,ax = plt.subplots(1,1,dpi=600)
fig_x, fig_y = 1,1
data = np.round(np.array([RS_acc +  RS_acc_2]),2)[:,::-1]

names = ['Other'] + witheld_ligands +  [x[1] + '+' + x[0] for x in [('SAM', 'cobalamin'), ('TPP','glycine'),('SAM','TPP'),('glycine','cobalamin'),('TPP','cobalamin'),
  ('FMN','cobalamin'),('FMN','TPP'),('FMN','SAM'),('FMN','glycine')]]
import matplotlib.colors as col
norm = col.Normalize(vmin=.7, vmax=1)
colors = cm.summer_r(norm(data[0]) )

_,f,t = ax.pie([1,]*nn, labels = names, explode = nn*[.1,], autopct='',
        shadow=False, startangle=90, colors=colors, labeldistance =1.05, pctdistance=.53, textprops={'fontsize': 7})

centre_circle = plt.Circle((0,0),0.70,fc='white', zorder=1)
#ax.add_artist(centre_circle)

data = np.round(np.array([witheld_acc +  witheld_acc_2 + witheld_acc_other]),2)[:,::-1]
#data = np.array([  [np.sum(predicted_UTRs[i] > .95) for i in range(len(predicted_UTRs))] + [np.sum(predicted_UTRs_2[i] > .95) for i in range(len(predicted_UTRs_2))]  ])

names = ['',]*nn
colors = cm.summer_r(norm(data[0]))

_,f,t = ax.pie([1,]*nn, labels = names, explode = nn*[.1,], autopct='', radius=.8,
        shadow=False, startangle=90, colors=colors, labeldistance =1.05, pctdistance=.53, textprops={'fontsize': 7})

centre_circle = plt.Circle((0,0),0.50,fc='white',)
#ax.add_artist(centre_circle)

names = ['',]*nn
data = np.array([  [np.sum(predicted_UTRs[i][:,1] > .95) for i in range(len(predicted_UTRs))] + [np.sum(predicted_UTRs_2[i][:,1] > .95) for i in range(len(predicted_UTRs_2))] + [np.sum(predicted_UTRs_other[i][:,1] > .95) for i in range(len(predicted_UTRs_other))] ])[:,::-1]
colors = ['#eeeeee']*nn
_,f,t = ax.pie([1,]*nn, labels = names, explode = nn*[.1,], autopct='', radius=.6,
        shadow=False, startangle=90, colors=colors, labeldistance =1.05, pctdistance=.53, textprops={'fontsize': 7}, )

centre_circle = plt.Circle((0,0),0.5,fc='white')
ax.add_artist(centre_circle)
print(data)
match_indexes = [np.where(all_utr_predicitions[:,i] >.95)[0].tolist() for i in range(20)]
hit_overlap = set.intersection(*map(set,match_indexes))
ax.text(-.45,0, '%0.0f / %0.0f of UTRs'%(len(hit_overlap),len(X_UTR)), fontsize=7)

x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()

r_acc  = np.round(np.array([RS_acc +  RS_acc_2 + RS_acc_other]),2)
val_acc  = np.round(np.array([witheld_acc +  witheld_acc_2 + witheld_acc_other]),2)
thresh = .95
UTR_pred = np.array([  [np.sum(predicted_UTRs[i] > thresh) for i in range(len(predicted_UTRs))] + [np.sum(predicted_UTRs_2[i] > thresh) for i in range(len(predicted_UTRs_2))]  + [np.sum(predicted_UTRs_other[i] > thresh) for i in range(len(predicted_UTRs_other))] ] )

n = nn
r = 1.1
dr = 360/nn
angle = dr/2
for i in range(nn):

  flippers = [0,1,2,3,4, 10,11,12,13]
  x,y = r * np.sin(np.radians(angle)), r* np.cos(np.radians(angle))
  #plt.plot([0,x],[0,y],'r-', zorder=1)

  s = .1
  x1,y1 = (r-.1) * np.sin(np.radians(angle)), (r-.1)* np.cos(np.radians(angle))
  x2,y2 = (r-.3) * np.sin(np.radians(angle)), (r-.3)* np.cos(np.radians(angle))
  x3,y3 = (r-.5) * np.sin(np.radians(angle)), (r-.5)* np.cos(np.radians(angle))

  #plt.scatter([x1,x2,x3],[y1,y2,y3], zorder=3)
  angle +=dr

  fs = 5

  # insane matplotlib rotation fix I hate matplotlib with the burning passion of
  # 10000 firey suns.
  # https://stackoverflow.com/questions/51028431/calculating-matplotlib-text-rotation

  dx = x2-x1
  dy = y2-y1
  Dx = dx * fig_x / (x_max - x_min)
  Dy = dy * fig_y / (y_max - y_min)


  if i not in flippers:
    aa = 90+ (180/np.pi)*np.arctan( Dy / Dx)
  else:
    aa = 270+ (180/np.pi)*np.arctan( Dy / Dx)



  plt.text(*(x1,y1),s='%0.2f'%r_acc[0][i], ha='center', va='center', fontsize=fs, rotation=aa, rotation_mode='anchor'  )
  plt.text(*(x2,y2),s='%0.2f'%val_acc[0][i], ha='center', va='center',fontsize=fs, rotation=aa, rotation_mode='anchor'  )
  plt.text(*(x3,y3),s='%0.0f'%UTR_pred[0][i], ha='center', va='center',fontsize=fs, rotation =aa, rotation_mode='anchor'  )

1/0
###############################################################################
# Generate X_RAND for FPR analysis
###############################################################################

reload_X_RAND = True

if reload_X_RAND:
    X_RAND = np.load('./X_RAND.npy')
else:
    import random
    random.seed(42)
    def generate_key(): # function to make random RNA sequences
        STR_KEY_GEN = 'augc'
        return ''.join(random.choice(STR_KEY_GEN) for _ in range(600))
    
    rand_30 = [generate_key() for i in range(60000)] #generate 60000, 600nt random sequences
    
    # cut the random sequences into lengths that match the RS data set
    h,xx = np.histogram(RS_lens, bins = np.max(RS_lens), density=True);
    cdf = np.cumsum(h*np.diff(xx)) 
    rand_lens = []
    for i in range(60000):
        rand_lens.append(np.where(np.random.rand() > cdf)[0][-1] + 25)
    plt.hist(lens,bins = 100, density=True, alpha=.4)
    plt.hist(RS_lens, bins=100, density=True, alpha=.4)
    
    # Convert the sequences to feature sets
    RAND_RS_features = []
    for i in range(len(rand_30)):
      mfe,hr = get_mfe_nupack(rand_30[i][:rand_lens[i]])
      RAND_RS_features.append([rand_30[i][:rand_lens[i]], str(mfe[0][0]), mfe[0][1]])
    
    X_RAND =np.zeros([len(rand_30), 66+8])
    k = 0
    for i in range(len(rand_30)):
      seq = clean_seq(rand_30[i])
      if len(seq) > 25:
    
        #seq = clean_seq(RS_df['SEQ'].iloc[i])
        kmerf = kmer_freq(seq)
        X_RAND[k,:64] = kmerf/np.sum(kmerf)
        X_RAND[k,64] = RAND_RS_features[i][-1]/max_mfe
        X_RAND[k,65] = get_gc(seq)
        #ds_RS.append(RS_df['ID'].iloc[i])
        #dot_RS.append(RS_df['NUPACK_DOT'].iloc[i])
        X_RAND[k,-8:] = be.annoated_feature_vector(RAND_RS_features[i][-2])
    
        X_RAND[k,66] = X_RAND[k,66]/max_ubs
        X_RAND[k,67] = X_RAND[k,67]/max_bs
        X_RAND[k,68] = X_RAND[k,68]/max_ill
        X_RAND[k,69] = X_RAND[k,69]/max_ilr
        X_RAND[k,70] = X_RAND[k,70]/max_lp
        X_RAND[k,71] = X_RAND[k,71]/max_lb
        X_RAND[k,72] = X_RAND[k,72]/max_rb
    
        k+=1

# use the original ensemble on X_RAND
predicted_rand = np.zeros([len(X_RAND), 20])
ensemble = estimators + estimators_2 + estimators_other
for j in range(20):
  predicted_rand[:,j] = ensemble[j].predict_proba(X_RAND)[:,1]
plt.plot(np.sum(predicted_rand > .5, axis =1)/20,'o')



###############################################################################
# Substring testing
###############################################################################

reload_X_SUB = True
reload_prediction_sub = True
save_substrings = True
if reload_X_SUB:
    X_SUB = np.load('./X_SUB.npy')
    n_substrings = 20
else:
    n_substrings = 20
    X_SUB =np.zeros([X_UTR.shape[0],n_substrings+1, 66+8])
    print('WARNING THIS WILL TAKE A LONG TIME TO GENERATE!!')
    print('generating sub strings...')
    k = 0
    for i in tqdm(range(len(UTR_db))):
      if not pd.isna(UTR_db[ccds_length].iloc[i]):
        seq = clean_seq(UTR_db[ccds_length].iloc[i])
        if len(seq) > 25:
          if np.sum(X_SUB[k]) == 0:
            print(k)
    
            subs = np.linspace(30,len(seq),n_substrings+1).astype(int)
            substrs = [seq[-x:] for x in subs]  + [seq,]
            for j in range(n_substrings+1):
              mfe,hr = get_mfe_nupack(substrs[j])
              kmerf = kmer_freq(substrs[j])
              X_SUB[k,j,:64] = kmerf/np.sum(kmerf)
    
              X_SUB[k,j,64] = mfe[0][1]/max_mfe
    
              X_SUB[k,j,65] = get_gc(seq)
    
              X_SUB[k,j,-8:] = be.annoated_feature_vector(str(mfe[0][0]), encode_stems_per_bp=True)
    
              X_SUB[k,j,66] = X_SUB[k,j,66]/max_ubs
              X_SUB[k,j,67] = X_SUB[k,j,67]/max_bs
              X_SUB[k,j,68] = X_SUB[k,j,68]/max_ill
              X_SUB[k,j,69] = X_SUB[k,j,69]/max_ilr
              X_SUB[k,j,70] = X_SUB[k,j,70]/max_lp
              X_SUB[k,j,71] = X_SUB[k,j,71]/max_lb
              X_SUB[k,j,72] = X_SUB[k,j,72]/max_rb
    
          k+=1
          if save_substrings:
              if k%1000 == 0:
                np.save('./X_SUB.npy', X_SUB)
               
if reload_prediction_sub:
    predicted_sub = np.load('./predicted_sub.npy')
else:
    predicted_sub = np.zeros([len(X_SUB),2,20])
    for i in tqdm(range(0,len(X_SUB))):
       print('WARNING THIS WILL TAKE A LONG TIME TO GENERATE!!')
      #if np.sum(predicted_sub[i,:,j]) == 0:
      for j in range(20):
          predicted_sub[i,:,j] = ensemble[j].predict_proba(X_SUB[i])


fullseqs_UTR = np.sum(predicted_sub[list(full_seqs_found_ids),:,:],axis=-1).T/20
subseqs_UTR = np.sum(predicted_sub[list(subseqs_found_ids - full_seqs_found_ids),:,:],axis=-1).T/20
subseqs_all = np.sum(predicted_sub,axis=-1).T/20

for i in [0,1,2]:
    plt.figure(dpi=global_dpi)
    b = fullseqs_UTR.T.flatten()
    if i == 0:
        a = np.vstack([np.linspace(0,20,21)]*fullseqs_UTR.shape[1]).flatten()
        t = 'All UTR found by full seqeunces'
    if i == 1:
        a = np.vstack([np.linspace(0,20,21)]*subseqs_UTR.shape[1]).flatten()
        t = 'subsequences UTR'
    if i == 2:
        a = np.vstack([np.linspace(0,20,21)]*subseqs_all.shape[1]).flatten()
        t = 'subsequences all'
    if i == 3: 
        a = np.vstack([np.linspace(0,20,21)]*subseqs_UTR.shape[1]).flatten()
        t = 'UTRs found not by full UTR'
    plt.hist2d(a, b, (np.arange(0,22)-.5, 20), cmap=plt.cm.RdBu, density=True, vmax=.7)
    plt.colorbar()
    plt.xlabel('subsequence id')
    plt.ylabel('ensemble probability')
    plt.title(t)

###############################################################################
# Testing on synthethic riboswitches
###############################################################################

theophylline_RSes = [ '''UAUGUUGAUACU
UAAUUUAAAGAU
UAAACAAAAGAU
GAUACCAGCCGA
AAGGCCCUUGGC
AGCUCUCGUGGA
GUGGAUGAAGUG''',

'''
CCCGGUACCGGU
GAUACCAGCAUC
GUCUUGAUGCCC
UUGGCAGCACCU
AUAAAGACAACA
AGAUGUGCGAAC
UCG
''',

'''
GGUGAUACCAGC
AUCGUCUUGAUG
CCCUUGGCAGCA
CCCCGCUGCAGG
ACAACAAGAUG
''',


'''
GGUGAUACCAGC
AUCGUCUUGAUG
CCCUUGGCAGCA
CCCUGCUAAGGU
AACAACAAGAUG
''',

'''
GGUACCGGUGAU
ACCAGCAUCGUC
UUGAUGCCCUUG
GCAGCACCCUGA
GAAGGGGCAACA
AGAUG''',

'''
GGUACCGGUGAU
ACCAGCAUCGUC
UUGAUGCCCUUG
GCAGCACCCGCU
GCGCAGGGGGUA
UCAACAAGAUG
''',

'''
GGUACCUGAUAA
GAUAGGGGUGAU
ACCAGCAUCGUC
UUGAUGCCCUUG
GCAGCACCAAGA
CAACAAGAUG
''',

'''
GGUACCGGUGAU
ACCAGCAUCGUC
UUGAUGCCCUUG
GCAGCACCCUGC
UAAGGUAACAAC
AAGAUG''',

'''
GGUACCGGUGAU
ACCAGCAUCGUC
UUGAUGCCCUUG
GCAGCACCCUGC
UAAGGAGGUAAC
AACAAGAUG''',

'''
GGUACCGGUGAU
ACCAGCAUCGUC
UUGAUGCCCUUG
GCAGCACCCUGC
UAAGGAGGCAAC
AAGAUG''',

'''
AUACGACUCACU
AUAGGUGAUACC
AGCAUCGUCUUG
AUGCCCUUGGCA
GCACCCUGCUAA
AGGAGGUAACAA
CAAGAUG''',

'''
AACGGGACUCAC
UAUAGGUACCGG
UGAUACCAGCAU
CGUCUUGAUGCC
CUUGGCAGCACC
CUGCGGGCCGGG
CAACAAGAUG
''',

'''
CACUGUUCGUCA
AGAAAGCAUCAU
UGUGACUGUGUA
GAUUGCUAUUAC
AAGAAGAUCAGG
AGCAAACUAUG''',

'''GGUGAUACCAGC
AUCGUCUUGAUG
CCCUUGGCAGCA
CCCUGCUAAGGA
GGUAACAACAUG''',

'''
GGUGAUACCAGC
AUCGUCUUGAUG
CCCUUGGCAGCA
CCCUGCUAAGGA
GGUAACUUAAUG
''',

'''
GGUGAUACCAGC
AUCGUCUUGAUG
CCCUUGGCAGCA
CCCUGCUAAGGA
GGUGUGUUAAUG''',

'''
GGUGAUACCAGC
AUCGUCUUGAUG
CCCUUGGCAGCA
CCCUGCUAAGGA
GGUCAACAAGAU
G''',

'''
CAGGUGAUACCA
GCAUCGUCUUGA
UGCCCUUGGCAG
CACCTATATAAGA
AGAAGGGUACCU
UAAACCCCUUCU
UCUUAUGAAGAA
GGGGUUUUUAUU
UUGGAGGAAUUU
UCCAUG''',

'''
AAGUGAUACCAG
CAUCGUCUUGAU
GCCCUUGGCAGC
ACUUCAGAAAUC
UCUGAAGUGCUG
UUUUUUUUAGGA
GGUUAAUGAUG''',

'''
AAUUAAAUAGCU
AUUAUCACGAUU
UUAUACCAGCUU
CGAAAGAAGCCC
UUGGCAGAAAAU
CCUGAUUACAAA
AUUUGUUUAUGA
CAUUUUUUGUAA
UCAGGAUUUUUU
UUGGAGGAATTTT
CCAUG''',

'''
GGGAGACCACAA
CGGUUUCCCUAU
CACCUUUUUGUA
GGUUGCCCGAAA
GGGCGACCCUGA
UGAGCCUGGAUA
CCAGCCGAAAGG
CCCUUGGCAGUU
AGACGAAACAAG
AAGGAGAUAUAC
CAAUG''',

'''AUAGGUACCUAA
UGCAACCUGAUA
CCAGCAUCGUCU
UGAUGCCCUUGG
CAGCAGGCAACA
AG
''',

'''
AAUUUCAUAGUU
AGAUCGUGUUAU
AUGGUGAAGAUA
AUACCAGCUUCG
AAAGAAGCCCUU
GGCAGUAUCUCG
UUGUUCAUAAUC
AUUUAUGAUGAU
UAAUUGAUAAGC
AAUGAGAGUAUU
CCUCUCAUUGCU
UUUUUUAUUGUG
GACAAAGCGCUC
UUUCUCCUCACC
CGCACGAACCAA
AAUGUAAAGGGU
GGUAAUACAUG
''',

'''
CUUCCUGACACGA
AAAUUUCAUAUCC
GUUCUUAAUACCA
GCUUCGAAAGAAG
CCCUUGGCAGUAA
GAAGAGACAAAAU
CACUGACAAAGUC
UUCUUCUUAAGAG
GACUUUUUUUAUU
UCUCUUUUUUCCU
UGCUGAUGUGAAU
AAAGGAGGCAGAC
AAUG
''',

'''
CAAAAAAUUAAUA
ACAUUUUCUCUUA
UACCAGCUUCGAA
AGAAGCCCUUGGC
AGGAGAGAGGCAG
UGUUUUACGUAGA
AAAGCCUCUUUCU
CUCAUGGGAAAGA
GGCUUUUUGUUGU
GAGAAAACCUCUU
AGCAGCCUGUAUC
CGCGGGUGAAAGA
GAGUGUUUUACAU
AUAAAGGAGGAGA
AACAAUG
'''

]



theophylline_RS = [x.replace('\n','').lower() for x in theophylline_RSes] + [ 'TCTAGAGACCGCTAAAGGAGATACCAGCATCGTCTTGATGCCCTTGGCAGCTCCGGTTCAGCGCCGAGGAATATAGGAGGTAATCCC'.replace('T','U').lower()]

Thyroxine_RS = ['GACGTCCTTAACGCGGGATAACATAGTCACGGTTTGTGGGAGGCTGTGGAGGCGAGACCGTGACCCCGGCAGCACCCAGGAGGAATACT'.replace('T','U').lower()]
TMR_RS = ['GCAGGCTCCCACGGATCGCGACTGGCGAGAGCCAGGTAACGAATCGATCCAGTACCCACGATTCGTTAAGGAGGTAATCC'.replace('T','U').lower()]
Dopamine_RS = ['GACGTCCTACCGCATTTCGGACATAGGGAATTCCGCGTGTGCGCCGCGGAAGACGTTGGAAGGATAGATACCTACAACGGGGAATATAGAGGCCAGCACATAGTGAGGCCCTCCTCCCCCCGGACGTACACGGGAGGCAGTTT'.replace('T','U').lower(),
               'GACGTCGCGACGCAACAGCGACCCCGTCTCTGTGTGCGCCAGAGACACTGGGGCAGATATGGGCCAGCACAGAATGAGGCCCCCTAGTAACATTACGGAGGGCGCCC'.replace('T','U').lower()]
Synthetic_Fluoride_RS = ['TCTAGACACCCTAATTGGAGATGGCATTCCTCCATTAACAAACCGCTGCGCCCGTAGCAGCTGATGATGCCTACAGAACACATAAGGAGGGTAGTCAT'.replace('T','U').lower()]
DNT_RS = ['TCTAGAGTCGAAGATAAGGCCGCTTTCCAGCTCGGTACCATAACACAAGTGGTAGACTATTCTCTGGTACGTGCGCCCCCGGCCGTATTACGGGAGCACGCCGGCTAAGGGAATAAGCGCACCGAGGAGGTCAAAT'.replace('T','U').lower()]

synthetic_apt = ['GGATCGCGACTGGCGAGAGCCAGGTAACGAATCGATCC'.replace('T','U').lower(),
                 'TCCAGCTCGGTACCATAACACAAGTGGTAGACTATTCTCTGGTACGTGCGCCCCCGGCCGTATTACGGGAGCACGCCGGCTAAGGG'.replace('T','U').lower(),
                 'GGAGATACCAGCATCGTCTTGATGCCCTTGGCAGCTCC'.replace('T','U').lower(),
                 'ATTGGAGATGGCATTCCTCCATTAACAAACCGCTGCGCCCGTAGCAGCTGATGATGCCTACAGA'.replace('T','U').lower(),
                 'GGATCGCGACTGGCGAGAGCCAGGTAACGAATCGATCC'.replace('T','U').lower(),]


syn_RS_features = []
for i in range(len(syn_RS)):
  mfe,hr = get_mfe_nupack(syn_RS[i])
  syn_RS_features.append([syn_RS[i], str(mfe[0][0]), mfe[0][1]])

X_SYN =np.zeros([len(syn_RS), 66+8])
k = 0
for i in range(len(syn_RS)):
  seq = clean_seq(syn_RS[i])
  if len(seq) > 25:

    #seq = clean_seq(RS_df['SEQ'].iloc[i])
    kmerf = kmer_freq(seq)
    X_SYN[k,:64] = kmerf/np.sum(kmerf)
    X_SYN[k,64] = syn_RS_features[i][-1]/max_mfe
    X_SYN[k,65] = get_gc(seq)
    #ds_RS.append(RS_df['ID'].iloc[i])
    #dot_RS.append(RS_df['NUPACK_DOT'].iloc[i])
    X_SYN[k,-8:] = be.annoated_feature_vector(syn_RS_features[i][-2])

    X_SYN[k,66] = X_SYN[k,66]/max_ubs
    X_SYN[k,67] = X_SYN[k,67]/max_bs
    X_SYN[k,68] = X_SYN[k,68]/max_ill
    X_SYN[k,69] = X_SYN[k,69]/max_ilr
    X_SYN[k,70] = X_SYN[k,70]/max_lp
    X_SYN[k,71] = X_SYN[k,71]/max_lb
    X_SYN[k,72] = X_SYN[k,72]/max_rb

    k+=1


predicted_syn = np.zeros([len(syn_RS), 20])
ensemble = estimators + estimators_2 + estimators_other
for j in range(20):
  predicted_syn[:,j] = ensemble[j].predict_proba(X_SYN)
  
x = np.linspace(0.01, .99, 100)
plt.figure(dpi=300)
p_r = [np.sum(np.sum(predicted_rand,axis=1)/20 > xx)/300 for xx in x]
p_t = [np.sum(((np.sum(predicted_syn > .5, axis =1)/20)[:25]) > xx)/25 for xx in x]
plt.plot(x, p_r)
plt.plot(x, p_t)
plt.legend(['Random','Theophylline RSs'])
plt.ylim([0,1])
plt.ylabel('% of sequences identified as RS')
plt.xlabel('selection threshold (P(Seq) > value)')
plt.plot([x[50], x[-5], x[-1]], [p_t[50], p_t[-5], p_t[-1]],'o', color=colors[1])
plt.text(x[50]-.03, p_t[50]-.1+.05, '***')
plt.text(x[-5]-.02, p_t[-5]-.1+.05, '**')
plt.text(x[-1]-.01, p_t[-1]-.1+.05, '**')

plt.figure(dpi=300).set_figheight(2)

p_values = [stats.binomtest(int(p_t[i]*25), n=25, p=p_r[i], alternative='two-sided').pvalue for i in range(len(x))]
plt.semilogy(x, p_values, color=colors[2])
plt.semilogy([0,1],[.05,.05],'--', color=colors[3])
plt.ylim([10**-7, 10**-.5])
plt.ylabel('P-value (binomal test 2 sided)')
plt.xlabel('selection threshold (P(Seq) > value)')
plt.text(.9, .08,'0.05', fontsize=8, color=colors[3])

###############################################################################
# GO analyses
###############################################################################

#Print list of genes to use for GO analysis
gene_hits = UTR_db[UTR_db['ID'].isin(UTR_hit_list)]['GENE'].values.tolist()


#@title barchart setup for GO Process
overall_labels = []
go_ids = []
genes = []
folds = []
pvals = []

sublabels = []
subgenes = []
sub_gos = []
sub_fold = []
sub_pvals = []
levels = []
for i in range(len(process['overrepresentation']['group'])):

  if i != 15:
    try:
      overall_labels.append(process['overrepresentation']['group'][i]['result'][0]['term']['label'])
      go_ids.append([process['overrepresentation']['group'][i]['result'][x]['term']['id'] for x in range(len(process['overrepresentation']['group'][i]['result']))][0])
      folds.append(process['overrepresentation']['group'][i]['result'][0]['input_list']['fold_enrichment'])
      pvals.append(process['overrepresentation']['group'][i]['result'][0]['input_list']['pValue'])
      genes.append(process['overrepresentation']['group'][i]['result'][0]['input_list']['mapped_id_list']['mapped_id'])


      sublabels.append([process['overrepresentation']['group'][i]['result'][x]['term']['label'] for x in range(len(process['overrepresentation']['group'][i]['result']))][0:])
      sub_gos.append([process['overrepresentation']['group'][i]['result'][x]['term']['id'] for x in range(len(process['overrepresentation']['group'][i]['result']))][0:])
      subgenes.append([process['overrepresentation']['group'][i]['result'][x]['input_list']['mapped_id_list']['mapped_id'] for x in range(len(process['overrepresentation']['group'][i]['result']))][0:])
      sub_fold.append([process['overrepresentation']['group'][i]['result'][x]['input_list']['fold_enrichment'] for x in range(len(process['overrepresentation']['group'][i]['result']))][0:])
      sub_pvals.append([process['overrepresentation']['group'][i]['result'][x]['input_list']['pValue'] for x in range(len(process['overrepresentation']['group'][i]['result']))][0:])
      levels.append([process['overrepresentation']['group'][i]['result'][x]['term']['level'] for x in range(len(process['overrepresentation']['group'][i]['result']))][0:])
    except:
      overall_labels.append(process['overrepresentation']['group'][i]['result']['term']['label'])
      go_ids.append(process['overrepresentation']['group'][i]['result']['term']['id'])
      folds.append(process['overrepresentation']['group'][i]['result']['input_list']['fold_enrichment'])
      pvals.append(process['overrepresentation']['group'][i]['result']['input_list']['pValue'])
      if i<16:
        genes.append(process['overrepresentation']['group'][i]['result']['input_list']['mapped_id_list']['mapped_id'])
        subgenes.append([process['overrepresentation']['group'][i]['result']['input_list']['mapped_id_list']['mapped_id']])
      else:
        genes.append([])
        subgenes.append(['None', ])


      sublabels.append([process['overrepresentation']['group'][i]['result']['term']['label']])
      sub_gos.append([process['overrepresentation']['group'][i]['result']['term']['id']])
      levels.append([0,])
      sub_fold.append([process['overrepresentation']['group'][i]['result']['input_list']['fold_enrichment']])
      sub_pvals.append([process['overrepresentation']['group'][i]['result']['input_list']['pValue']])

sublabels = [item for sublist in sublabels for item in sublist][::-1]
levels = [item for sublist in levels for item in sublist][::-1]
sub_gos = [item for sublist in sub_gos for item in sublist][::-1]
sub_fold = [item for sublist in sub_fold for item in sublist][::-1]
sub_pvals = [item for sublist in sub_pvals for item in sublist][::-1]
subgenes =  [item for sublist in subgenes for item in sublist][::-1]
ask = [    ''.join(    [['','*'][p<.05] ,  ['','*'][p<.01] , ['','*'][p<.001] , ['','*'][p<.0001] ])  for p in sub_pvals]


gc_total = [526,560,1216,1423,1270,444,368,247,243,607,451,243,181,2883,179,5727,2633,449,303,254,536,3050,2999,3292,2825,2276,1635,868,413,223,127,252,508,106,7228,3920,1487,2518,773,5941,2314,2534,1333,2603,2464,1588,
            527,3573,802,7697,6710,5013,537,408,379,124,67,77,59,86,103,60,1943,3542,3122,3174,2058,4102,5872,5647,5709,4067,3752,4163,3938,50,91,91,119,113,15044,8131,6606,159,86,46,25,111,28]

gc = [len(x) for x in subgenes]
gc[0] = 0
gc[1] = 0


gc_per = np.array(gc)/np.array(456)
ss = [sub_gos[i] + ':      '+ ''.join(['->',]*levels[i]) +' '+ sublabels[i] + ''.join([' ',]) for i in range(len(sublabels))]

#@title barchart for GO Process
colors = ['#ef476f', '#073b4c','#06d6a0','#7400b8','#073b4c', '#118ab2',]
fig,axes = plt.subplots(1,2,figsize=(2,8),dpi=400)
bars = axes[0].barh(ss, sub_fold,)
axes[0].plot([1,1], [-1, len(sublabels)+1],'b-',lw=.5)

axes[0].tick_params(axis='y', which='major', labelsize=3)
axes[0].tick_params(axis='y', which='minor', labelsize=2)

axes[0].tick_params(axis='x', which='major', labelsize=5)
axes[0].tick_params(axis='x', which='minor', labelsize=5)

axes[1].tick_params(axis='x', which='major', labelsize=5)
axes[1].tick_params(axis='x', which='minor', labelsize=5)

axes[0].set_xlabel('Fold enrichment', fontsize=6)
r = axes[0].set_yticklabels(ss, ha = 'left')


yax = axes[1].get_yaxis()
yax.set_visible(False)
b2 = axes[1].barh(ss, np.log10(np.array(sub_pvals)))
axes[1].set_xlabel('Log10 (p-value)', fontsize=6)

crosslines = np.where(np.array(levels[1:]) - np.array(levels[:-1]) <= 0)[0] + 1

axes[1].plot([-101,-2],[0-.5,0-.5],'k-',lw=.5, clip_on=False, zorder=100)
k = 0
c = colors[k]
for i in range(len(levels)):
  bars[i].set_color(colors[k])
  b2[i].set_color(colors[k])
  if levels[i] == 0:
    a = axes[1].plot([-101,-2],[i+.5,i+.5],'k-',lw=.5, clip_on=False, zorder=100)
    k+=1
    k = k%len(colors)

for i in range(len(ask)):
  axes[0].text(sub_fold[i]+1  ,i,s=ask[i], fontsize=3, ha='center',va='center')

axes[0].set_ylim([-1, len(levels)+1])
axes[1].set_ylim([-1, len(levels)+1])
axes[0].set_xlim([0,15])
axes[1].set_xlim([-18,-2])
plt.draw()
yax = axes[0].get_yaxis()
pad = max(T.label.get_window_extent().width for T in yax.majorTicks)
axes[0].yaxis.set_tick_params(pad=pad/5)

axes[1].plot([-5,-5], [-1, len(sublabels)+1],'k-',lw=.2)
axes[1].plot([-10,-10], [-1, len(sublabels)+1],'k-',lw=.2)
axes[1].plot([-15,-15], [-1, len(sublabels)+1],'k-',lw=.2)

plt.savefig('process_go_plot.svg')

#@title GO function bar chart setup
with open('/content/drive/MyDrive/function.json', 'r') as f:
    function = json.load(f)
overall_labels = []
go_ids = []
genes = []
folds = []
pvals = []

sublabels = []
subgenes = []
sub_gos = []
sub_fold = []
sub_pvals = []
levels = []
for i in range(len(function['overrepresentation']['group'])):

  if i != 5: #unclassified is 5
    try:
      overall_labels.append(function['overrepresentation']['group'][i]['result'][0]['term']['label'])
      go_ids.append([function['overrepresentation']['group'][i]['result'][x]['term']['id'] for x in range(len(function['overrepresentation']['group'][i]['result']))][0])
      folds.append(function['overrepresentation']['group'][i]['result'][0]['input_list']['fold_enrichment'])
      pvals.append(function['overrepresentation']['group'][i]['result'][0]['input_list']['pValue'])
      genes.append(function['overrepresentation']['group'][i]['result'][0]['input_list']['mapped_id_list']['mapped_id'])


      sublabels.append([function['overrepresentation']['group'][i]['result'][x]['term']['label'] for x in range(len(function['overrepresentation']['group'][i]['result']))][0:])
      sub_gos.append([function['overrepresentation']['group'][i]['result'][x]['term']['id'] for x in range(len(function['overrepresentation']['group'][i]['result']))][0:])
      subgenes.append([function['overrepresentation']['group'][i]['result'][x]['input_list']['mapped_id_list']['mapped_id'] for x in range(len(function['overrepresentation']['group'][i]['result']))][0:])
      sub_fold.append([function['overrepresentation']['group'][i]['result'][x]['input_list']['fold_enrichment'] for x in range(len(function['overrepresentation']['group'][i]['result']))][0:])
      sub_pvals.append([function['overrepresentation']['group'][i]['result'][x]['input_list']['pValue'] for x in range(len(function['overrepresentation']['group'][i]['result']))][0:])
      levels.append([function['overrepresentation']['group'][i]['result'][x]['term']['level'] for x in range(len(function['overrepresentation']['group'][i]['result']))][0:])
    except:
      overall_labels.append(function['overrepresentation']['group'][i]['result']['term']['label'])
      go_ids.append(function['overrepresentation']['group'][i]['result']['term']['id'])
      folds.append(function['overrepresentation']['group'][i]['result']['input_list']['fold_enrichment'])
      pvals.append(function['overrepresentation']['group'][i]['result']['input_list']['pValue'])
      if i<16:
        genes.append(function['overrepresentation']['group'][i]['result']['input_list']['mapped_id_list']['mapped_id'])
        subgenes.append([function['overrepresentation']['group'][i]['result']['input_list']['mapped_id_list']['mapped_id']])
      else:
        genes.append([])
        subgenes.append(['None', ])


      sublabels.append([function['overrepresentation']['group'][i]['result']['term']['label']])
      sub_gos.append([function['overrepresentation']['group'][i]['result']['term']['id']])
      levels.append([0,])
      sub_fold.append([function['overrepresentation']['group'][i]['result']['input_list']['fold_enrichment']])
      sub_pvals.append([function['overrepresentation']['group'][i]['result']['input_list']['pValue']])

sublabels = [item for sublist in sublabels for item in sublist][::-1]
levels = [item for sublist in levels for item in sublist][::-1]
sub_gos = [item for sublist in sub_gos for item in sublist][::-1]
sub_fold = [item for sublist in sub_fold for item in sublist][::-1]
sub_pvals = [item for sublist in sub_pvals for item in sublist][::-1]
subgenes =  [item for sublist in subgenes for item in sublist][::-1]
ss = [sub_gos[i] + ':      '+ ''.join(['->',]*levels[i]) +' '+ sublabels[i] + ''.join([' ',]) for i in range(len(sublabels))]
ask = [    ''.join(    [['','*'][p<.05] ,  ['','*'][p<.01] , ['','*'][p<.001] , ['','*'][p<.0001] ])  for p in sub_pvals]

###############################################################################
# 20 fold k cross validation without structural holdouts
###############################################################################

#Here is an optional ensemble without structural cross validation, instead it uses
# a 20 fold cross validation with all structures scrambled. Asked as a reviewer question
# for the original submission

import sklearn

do_20kcv = True
save = True
retrain = False
model_name = 'EKmodel_witheld_20kfcv_2'

print('Training an ensemble with no structural cross validation, using random 20 fold cross validation...')
print('save the model files? %i'%save)
print('retraining? %i'%retrain)
print('using model name: %s'%model_name)
print('__________________________')
if do_20kcv:
    X = np.vstack([X_UTR, X_RS_full,])
    y = np.zeros(len(X))
    y[len(X_UTR):] = 1
    # generate the 20 fold cross validation shuffles
    kf = sklearn.model_selection.KFold(n_splits=20, shuffle=True, random_state=42)
    
    X_RAND = np.load('./X_RAND.npy')
    X_EXONS = np.load('./X_EXONS.npy')
    
    ensemble_2 = [] # preallocate a list for the 20 classifiers
    witheld_RS_acc_2 = []
    for i, (train_index, test_index) in enumerate(kf.split(X,y)): # for every cross fold train
        print('training cross fold %i'%i)
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        if retrain:
            svc = SVC(C=15, kernel='rbf', gamma=0.2, probability=True)
            pu_estimator = ElkanotoPuClassifier(estimator=svc, hold_out_ratio=0.2)
            pu_estimator.fit(X_train, y_train)
            if save:
              dump(pu_estimator,'./elkanoto_models/%s_%s.joblib'%(model_name,i))
        else:
            svc = SVC(C=10, kernel='rbf', gamma=0.4, probability=True)
            pu_estimator = ElkanotoPuClassifier(estimator=svc, hold_out_ratio=0.2)
            pu_estimator =load('./elkanoto_models/%s_%s.joblib'%(model_name,str(i)))
            
        predicted_RS = pu_estimator.predict_proba(X_test[y_test==1])
        witheld_RS_acc_2.append(np.sum(predicted_RS[:,1] > .5 )/ (len(X_test[y_test==1])))
        print('Test accuracy:')
        print(np.sum(predicted_RS[:,1] > .5 )/ (len(X_test[y_test==1])))
        print('__________________________')
        ensemble_2.append(pu_estimator)
    
    kf = sklearn.model_selection.KFold(n_splits=20, shuffle=True, random_state=42)
    train_ens2_acc = []
    test_ens2_acc = []
    found_UTR_ens2 = []
    random_ens2_acc = []
    structured_ens2_acc = []
    
    rands_2 = []
    exons_2 = []
    
    print('Running ensemble on all data to construct plot....')
    # for each cross fold get its performance on the data
    for i, (train_index, test_index) in enumerate(kf.split(X,y)):
        print('k fold %i'%i)
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        
        p = ensemble_2[i].predict_proba((X_test[y_test==1]))
        t = ensemble_2[i].predict_proba((X_train[y_train==1]))
        u = ensemble_2[i].predict_proba(X_UTR)
        r =   ensemble_2[i].predict_proba(X_RAND)       
        ex =   ensemble_2[i].predict_proba(X_EXONS)                  
    
        rands_2.append(r)
        exons_2.append(ex)
        test_ens2_acc.append(np.sum(p[:,1] > .5 )/ (len(X_test[y_test==1])))
        train_ens2_acc.append(np.sum(t[:,1] > .5 )/ (len(X_train[y_train==1])))
        found_UTR_ens2.append(np.sum(u[:,1] > .5 )/ (len(X_UTR)))
        random_ens2_acc.append(np.sum(r[:,1] > .5 )/ (len(X_RAND)))
        structured_ens2_acc.append(np.sum(ex[:,1] > .5 )/ (len(X_EXONS)))
        
        
    train_ens_acc = []
    test_ens_acc = []
    found_UTR_ens = []
    
    random_ens_acc = []
    structured_ens_acc = []
    for i in range(len(ensemble)):
        r =   ensemble[i].predict_proba(X_RAND)       
        ex =   ensemble[i].predict_proba(X_EXONS)      
        random_ens_acc.append(np.sum(r[:,1] > .5 )/ (len(X_RAND)))
        structured_ens_acc.append(np.sum(ex[:,1] > .5 )/ (len(X_EXONS)))
    
    
    data = np.array([RS_acc + RS_acc_2 + RS_acc_other,
                          witheld_acc + witheld_acc_2 + witheld_acc_other, 
                          UTR_acc + UTR_acc_2 + UTR_acc_other,
                          random_ens_acc,
                          structured_ens_acc])
    
    # Ensemble with structural cross validation
    data[2,:] = 1 - data[2,:]
    plt.matshow(data, cmap='coolwarm_r', vmax=1, vmin=0)
    ax = plt.gca()
    for (i, j), z in np.ndenumerate(data):
        ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
    ax.set_xticks([x for x in range(20)])
    ax.set_xticklabels([x for x in range(20)])
    ax.set_yticklabels(['','Train', 'Test', 'UTR', 'RAND', 'EXON'])
    ax.set_xticklabels( witheld_ligands + ['-'.join(x) for x in pairs] + ['other'], rotation=90)
    ax.set_xlabel('Ensemble w/ structural cross validation training and testing results on all groups')
    
    data = np.array([train_ens2_acc,
                          test_ens2_acc, 
                          found_UTR_ens2,
                          random_ens2_acc,
                          structured_ens2_acc])
    
    # Ensemble 2 training outputs
    plt.figure()
    plt.matshow(data, cmap='coolwarm_r', vmax=1, vmin=0)
    
    ax = plt.gca()
    for (i, j), z in np.ndenumerate(data):
        ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
    ax.set_xticks([x for x in range(20)])
    ax.set_xticklabels([x for x in range(20)])
    ax.set_yticklabels(['', 'Train', 'Test', 'UTR', 'RAND', 'EXON'])
    ax.set_xlabel('Ensemble w/ random 20 fold cross validation training and testing results on all groups')
    
    #ax.set_xticklabels( witheld_ligands + ['-'.join(x) for x in pairs] + ['other'], rotation=90)
    
    

#@title GO function bar chart
colors = ['#ef476f', '#073b4c','#06d6a0','#7400b8','#073b4c', '#118ab2',]
fig,axes = plt.subplots(1,2,figsize=(2,2),dpi=400)
bars = axes[0].barh(ss, sub_fold)
axes[0].plot([1,1], [-1, len(sublabels)+1],'b-',lw=.5)

axes[0].tick_params(axis='y', which='major', labelsize=3)
axes[0].tick_params(axis='y', which='minor', labelsize=2)

axes[0].tick_params(axis='x', which='major', labelsize=5)
axes[0].tick_params(axis='x', which='minor', labelsize=5)

axes[1].tick_params(axis='x', which='major', labelsize=5)
axes[1].tick_params(axis='x', which='minor', labelsize=5)

axes[0].set_xlabel('Fold enrichment', fontsize=6)
r = axes[0].set_yticklabels(ss, ha = 'left')


yax = axes[1].get_yaxis()
yax.set_visible(False)
b2 = axes[1].barh(ss, np.log10(np.array(sub_pvals)))
axes[1].set_xlabel('Log10(p-value)', fontsize=6)

crosslines = np.where(np.array(levels[1:]) - np.array(levels[:-1]) <= 0)[0] + 1

axes[1].plot([-75,-2],[0-.5,0-.5],'k-',lw=.5, clip_on=False, zorder=100)
k = 0
c = colors[k]
for i in range(len(levels)):
  bars[i].set_color(colors[k])
  b2[i].set_color(colors[k])
  if levels[i] == 0:
    a = axes[1].plot([-75,-2],[i+.5,i+.5],'k-',lw=.5, clip_on=False, zorder=100)
    k+=1
    k = k%len(colors)

for i in range(len(ask)):
  axes[0].text(sub_fold[i]+1  ,i,s=ask[i], fontsize=3, ha='center',va='center')

axes[0].set_ylim([-1, len(levels)+1])
axes[1].set_ylim([-1, len(levels)+1])
axes[0].set_xlim([0,15])
axes[1].set_xlim([-18,-2])
plt.draw()
yax = axes[0].get_yaxis()
pad = max(T.label.get_window_extent().width for T in yax.majorTicks)
axes[0].yaxis.set_tick_params(pad=pad/5)

axes[1].plot([-5,-5], [-1, len(sublabels)+1],'k-',lw=.2)
axes[1].plot([-10,-10], [-1, len(sublabels)+1],'k-',lw=.2)
axes[1].plot([-15,-15], [-1, len(sublabels)+1],'k-',lw=.2)

plt.savefig('function_go_plot.svg')


###############################################################################
# RETRAIN ENSEMBLE WITH RANDOM AND EXONS
###############################################################################
X_RAND = np.load('./X_RAND.npy')
X_EXONS = np.load('./X_EXONS.npy')

witheld_ligands = ['cobalamin', 'guanidine', 'TPP','SAM','glycine','FMN','purine','lysine','fluoride','zmp-ztp',]

pairs = [(x,) for x in witheld_ligands] + [('SAM', 'cobalamin'), ('TPP','glycine'),('SAM','TPP'),('glycine','cobalamin'),('TPP','cobalamin'),
  ('FMN','cobalamin'),('FMN','TPP'),('FMN','SAM'),('FMN','glycine'), ('other',)]


#@title single drop out of <2% ligands
witheld_acc = []
RS_acc = []
RS_all_acc = []
UTR_acc = []
Exon_acc = []
Rand_acc = []


predicted_RSs = []
predicted_withelds = []
predicted_UTRs = []
predicted_Exons = []
predicted_rands = []
predicted_RSs_all = []

estimators= []

retrain  = False #@param {type:"boolean"}
save  = False #@param {type:"boolean"}
model_name = "EKmodel_witheld_w_exon_rand" #@param {type:"string"}


for i in tqdm(range(20)):

  if pairs[i][0] != 'other':
      if len(pairs[i]) == 1:
          X = np.vstack([X_UTR, X_RAND, X_EXONS, X_RS, ] + [x[0] for x in ligand_dfs[:i]] + [x[0] for x in ligand_dfs[i+1:]] )
          X_witheld = ligand_dfs[i][0]
          X_t = np.vstack([ X_RS, ] + [x[0] for x in ligand_dfs[:i]] + [x[0] for x in ligand_dfs[i+1:]] )
          name =  pairs[i][0]
      else:
          witheld_1 = pairs[i][0]
          witheld_2 = pairs[i][1]

          ind_1 = witheld_ligands.index(witheld_1)
          ind_2 = witheld_ligands.index(witheld_2)

          X = np.vstack([X_UTR, X_RAND, X_EXONS, X_RS,] + [ligand_dfs[i][0] for i in range(len(ligand_dfs)) if i not in [ind_1,ind_2]])
          X_witheld = np.vstack([ligand_dfs[ind_1][0], ligand_dfs[ind_2][0]])
          X_t = np.vstack([ X_RS, ] + [ligand_dfs[i][0] for i in range(len(ligand_dfs)) if i not in [ind_1,ind_2]])
          name = witheld_1 + '_' + witheld_2
  else:
      X = np.vstack([X_UTR, X_RAND, X_EXONS,] + [x[0] for x in ligand_dfs]  )
      X_witheld = X_RS
      X_t = np.vstack([ X_RS, ] + [x[0] for x in ligand_dfs]  )
      name = 'other'
        
  print('___________________________')
  print('Training:')
  print(pairs[i])
  print('X shape: ')
  print(X.shape)
  print('X_witheld:')
  print(X_witheld.shape)

    

  if retrain:
    svc = SVC(C=10, kernel='rbf', gamma=0.4, probability=True)
    pu_estimator = ElkanotoPuClassifier(estimator=svc, hold_out_ratio=0.2)
    y = np.zeros(len(X))
    y[len(X_UTR)+len(X_RAND)+len(X_EXONS):] = 1
    pu_estimator.fit(X, y)
  else:
    pu_estimator =load('./elkanoto_models/%s_%s.joblib'%(model_name, name))


  predicted_RS = pu_estimator.predict_proba(X_t)
  predicted_witheld= pu_estimator.predict_proba(X_witheld)
  predicted_UTR = pu_estimator.predict_proba(X_UTR)
  predicted_RAND = pu_estimator.predict_proba(X_RAND)
  predicted_EXON = pu_estimator.predict_proba(X_EXONS)
  predicted_RS_all = pu_estimator.predict_proba(X_RS_full)

  UTR_acc.append( np.sum((predicted_UTR[:,1] < .5))/len(X_UTR) )
  RS_acc.append( np.sum((predicted_RS[:,1] > .5))/len(X_t) )
  witheld_acc.append( np.sum((predicted_witheld[:,1] > .5))/len(X_witheld)  )
  Rand_acc.append(np.sum((predicted_RAND[:,1] > .5))/len(X_RAND)  )
  Exon_acc.append(np.sum((predicted_EXON[:,1] > .5))/len(X_EXONS)  )
  RS_all_acc.append(np.sum((predicted_RS_all[:,1] > .5))/len(predicted_RS_all)  )
  
  print('UTR accuracy:')
  print( np.sum((predicted_UTR[:,1] < .5))/len(X_UTR) )
  print('RS accuracy:')
  print(np.sum((predicted_RS[:,1] > .5))/len(X_t) )
  print('Witheld accuracy:')
  print( np.sum((predicted_witheld[:,1] > .5))/len(X_witheld))
  print('Rand accuracy:')
  print(np.sum((predicted_RAND[:,1] > .5))/len(X_RAND) )
  print('Exon accuracy:')
  print(np.sum((predicted_EXON[:,1] > .5))/len(X_EXONS) )
  
  predicted_RSs.append(predicted_RS)
  predicted_withelds.append(predicted_witheld)
  predicted_UTRs.append(predicted_UTR)
  predicted_Exons.append(predicted_EXON)
  predicted_rands.append(predicted_RAND)
  predicted_RSs_all.append(predicted_RS_all)
  if retrain:
    if save:
      dump(pu_estimator,'./elkanoto_models/%s_%s.joblib'%(model_name, name))
  estimators.append(pu_estimator)


witheld_acc = []
RS_acc = []
UTR_acc = []
Exon_acc = []
Rand_acc = []

t = .5
for i in range(20):
    UTR_acc.append( np.sum((predicted_UTRs[i][:,1] < t))/len(X_UTR) )
    RS_acc.append( np.sum((predicted_RSs[i][:,1] > t))/len(predicted_RSs[i][:,1]) )
    witheld_acc.append( np.sum((predicted_withelds[i][:,1] > t))/len(predicted_withelds[i][:,1])  )
    Rand_acc.append(np.sum((predicted_rands[i][:,1] > t))/len(X_RAND)  )
    Exon_acc.append(np.sum((predicted_Exons[i][:,1] > t))/len(X_EXONS)  )
    

mat = np.array([RS_acc, witheld_acc, UTR_acc, Rand_acc, Exon_acc])
mat[2, :] = 1 -  mat[2,:]


fig, ax = plt.subplots()
# Using matshow here just because it sets the ticks up nicely. imshow is faster.
ax.matshow(mat, cmap='summer', vmax=1)
ax.set_yticklabels(['', 'RS training', 'RS witheld', 'UTR', 'RAND', 'EXON'], size=5)
for (i, j), z in np.ndenumerate(mat):
    ax.text(j, i, '{:0.1f}'.format(z*100), ha='center', va='center', size=5)
ax.set_xticks([x for x in range(20)])
ax.set_xticklabels(['-'.join(x) for x in pairs] , size=5, rotation=90, )
plt.savefig('ensemble2_5.svg')





witheld_acc = []
RS_acc = []
UTR_acc = []
Exon_acc = []
Rand_acc = []
RS_all_acc = []

t = .5
for i in range(20):
    UTR_acc.append( np.sum((predicted_UTRs[i][:,1] < t))/len(X_UTR) )
    RS_acc.append( np.sum((predicted_RSs[i][:,1] > t))/len(predicted_RSs[i][:,1]) )
    witheld_acc.append( np.sum((predicted_withelds[i][:,1] > t))/len(predicted_withelds[i][:,1])  )
    Rand_acc.append(np.sum((predicted_rands[i][:,1] > t))/len(X_RAND)  )
    Exon_acc.append(np.sum((predicted_Exons[i][:,1] > t))/len(X_EXONS)  )
    RS_all_acc.append(np.sum((predicted_RS_all[:,1] > t))/len(predicted_RS_all)  )

witheld_acc_2 = []
RS_acc_2 = []
UTR_acc_2 = []
Exon_acc_2 = []
Rand_acc_2 = []
RS_all_acc_2 = []
t = .95
for i in range(20):
    UTR_acc_2.append( np.sum((predicted_UTRs[i][:,1] < t))/len(X_UTR) )
    RS_acc_2.append( np.sum((predicted_RSs[i][:,1] > t))/len(predicted_RSs[i][:,1]) )
    witheld_acc_2.append( np.sum((predicted_withelds[i][:,1] > t))/len(predicted_withelds[i][:,1])  )
    Rand_acc_2.append(np.sum((predicted_rands[i][:,1] > t))/len(X_RAND)  )
    Exon_acc_2.append(np.sum((predicted_Exons[i][:,1] > t))/len(X_EXONS)  )
    
    RS_all_acc_2.append(np.sum((predicted_RS_all[:,1] > t))/len(predicted_RS_all)  )
    
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('blank', [(1,1,1), (1,1,1)], N=2)

fig, ax = plt.subplots()
mat = np.array([[len(X_RS_full), len(X_UTR), len(X_EXONS), len(X_RAND)],
               [np.mean(RS_all_acc), 1-np.mean(UTR_acc), np.mean(Exon_acc), np.mean(Rand_acc)],
               [int(x) for x in [np.mean(RS_all_acc)*len(X_RS_full), (1-np.mean(UTR_acc))*len(X_UTR), len(X_EXONS)*np.mean(Exon_acc), len(X_RAND)*np.mean(Rand_acc)]],
               [np.mean(RS_all_acc_2), 1-np.mean(UTR_acc_2), np.mean(Exon_acc_2), np.mean(Rand_acc_2)], 
               [int(x) for x in [np.mean(RS_all_acc_2)*len(X_RS_full), (1-np.mean(UTR_acc_2))*len(X_UTR), len(X_EXONS)*np.mean(Exon_acc_2), len(X_RAND)*np.mean(Rand_acc_2)]]])

ax.matshow(mat, cmap=cmap, vmax=1, aspect = .3)
ax.set_xticklabels(['', 'RS', 'UTR', 'EXON', 'RAND'], size=5)
for (i, j), z in np.ndenumerate(mat):
    if i == 0:
        ax.text(j, i, '{}'.format(int(z)), ha='center', va='center', size=10)
    if i in [1,3]:
        ax.text(j, i, '{:0.3f}'.format(z*100) + '%', ha='center', va='center', size=10)
    if i in [2,4]:
        ax.text(j, i, '{}'.format(int(z)), ha='center', va='center', size=10)
ax.set_xticks(np.arange(-.5,4,1), minor=True)
ax.set_yticks(np.arange(-.5,5,1), minor=True)
ax.grid(which='minor')

plt.savefig('fpr.svg')
        
        
        
plt.figure()
plt.boxplot(np.array([RS_acc, witheld_acc, [1-x for x in UTR_acc], Exon_acc, Rand_acc]).T  , vert=True, showfliers=True, 
            #boxprops={'alpha':.8, 'color':cm.viridis(i/20)},
            #capprops={'alpha':.8, 'color':cm.viridis(i/40)},
            #whiskerprops={'alpha':.8, 'color':cm.viridis(i/40)},
            #medianprops={'alpha':.8, 'color':cm.viridis(i/40)},
            #meanprops={'alpha':.8, 'color':cm.viridis(i/40)},
            ); 

ens1_utrs = [ids_UTR.index(x) for x in ids_UTR if x in UTR_hit_list]
ens2_utrs = []
for i in range(20):
   ens2_utrs.append((np.where(predicted_UTRs[i][:,1] > t)[0]).tolist())
ens2_utrs = set.intersection(*[set(x) for x in ens2_utrs])
ax = plt.gca()
ax.set_xticklabels(['train RS','test RS','UTR','Exon','Random'])
plt.title('Box plot across classifiers for FPR analyses (.95 threshold)')



###############################################################################
# Generate X_Eukaryotic for testing the ensembles
###############################################################################


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
a = RS_db['DESC']
b = [x for x in a if True in [y.lower()  in x.lower() for y in eukaryote_list ]]
c = [x for x in a if True in [y.lower()  in x.lower() for y in remove_list ]]
d = [x for x in a if True not in [y.lower()  in x.lower() for y in remove_list ]]
e = [x for x in d if True not in [y.lower() in x.lower() for y in eukaryote_list ]]

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
        g.append(f[i])
    else:
        h.append(f[i])
        
# manually add some
        
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

#CHECK THAT ALL DESCRIPTIONS ARE LABELED
print(len(set(g + c)) / len(set(a)))

eukaryotic = []
for i in range(len(a)):
    if a[i] in g:
        eukaryotic.append(1)
    if a[i] in c:
        eukaryotic.append(0)
RS_db['EUKARYOTIC'] = eukaryotic


lower_desc = [y.lower() for y in a]
eukaryotic_TPP = [x.lower() for x in g if 'TPP'  in x]
eukaryotic_junk = [x for x in g if 'TPP' not  in x]

X_eukaryotic = np.zeros([len(eukaryotic_TPP), 66+8])
k = 0
missing = []
for i in tqdm(range(len(RS_db))):
  if RS_db['EUKARYOTIC'].iloc[i]:
      if'tpp' in RS_db['DESC'].iloc[i].lower():
          seq = clean_seq(RS_db['SEQ'].iloc[i])
          if len(seq) <=25:
              print(len(seq))
          if len(seq) > 25:
            seq = clean_seq(RS_db['SEQ'].iloc[i])
            kmerf = kmer_freq(seq)
            X_eukaryotic[k,:64] = kmerf/np.sum(kmerf)
            X_eukaryotic[k,64] = RS_db['NUPACK_MFE'].iloc[i]
            X_eukaryotic[k,65] = get_gc(seq)
            X_eukaryotic[k,-8:] = be.annoated_feature_vector(RS_db['NUPACK_DOT'].iloc[i], encode_stems_per_bp=True)
            k+=1
            
X_eukaryotic[:,66] = X_eukaryotic[:,66]/max_ubs
X_eukaryotic[:,67] = X_eukaryotic[:,67]/max_bs
X_eukaryotic[:,68] = X_eukaryotic[:,68]/max_ill
X_eukaryotic[:,69] = X_eukaryotic[:,69]/max_ilr
X_eukaryotic[:,70] = X_eukaryotic[:,70]/max_lp
X_eukaryotic[:,71] = X_eukaryotic[:,71]/max_lb
X_eukaryotic[:,72] = X_eukaryotic[:,72]/max_rb
X_eukaryotic[:,64] = X_eukaryotic[:,64]/max_mfe
X_eukaryotic = X_eukaryotic[:k]



euk_acc = []
euk_proba = []
t = .5
for i in range(20):
    predicted_euk = ensemble[i].predict_proba(X_eukaryotic)
    euk_proba.append(predicted_euk)
    euk_acc.append( np.sum((predicted_euk[:,1] > t))/len(predicted_euk) )



from sklearn.decomposition import PCA
pca = PCA(n_components=2)
p = pca.fit(np.vstack([X_UTR, X_RS]))
print(pca.explained_variance_ratio_)
p = pca.fit(np.vstack([X_UTR, X_RS_full]))
x_utr_t = p.transform(X_UTR)
x_rs_t = p.transform(X_RS_full)
x_euk_t = p.transform(X_eukaryotic)
plt.figure()
plt.scatter(x_utr_t[:,0], x_utr_t[:,1], s=5,alpha=.2)
plt.scatter(x_rs_t[:,0], x_rs_t[:,1],s=5, alpha=.2)
plt.scatter(x_euk_t[:,0], x_euk_t[:,1],s=5, alpha=.2)
plt.legend(['UTR','RS'])
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.3f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.3f}%)')


# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 13:41:30 2023

@author: willi
"""

import joblib
import numpy as np

# Load previously trained ensemble

model_name = "EKmodel_witheld_w_struct_features_9_26"
model_norm = 'utr_proba_norm.npy'
pairs = [('SAM', 'cobalamin'), ('TPP','glycine'),('SAM','TPP'),('glycine','cobalamin'),('TPP','cobalamin'),
  ('FMN','cobalamin'),('FMN','TPP'),('FMN','SAM'),('FMN','glycine')]

estimators_other = []
for i in tqdm(range(1)):
  pu_estimator =joblib.load('./%s_%s.joblib'%(model_name, 'other'))
  estimators_other.append(pu_estimator)

estimators=[]
for i in tqdm(range(len(witheld_ligands))):
  pu_estimator =joblib.load('./%s_%s.joblib'%(model_name, witheld_ligands[i]))
  estimators.append(pu_estimator)

estimators_2 = []
for i in tqdm(range(len(pairs))):
  witheld_1 = pairs[i][0]
  witheld_2 = pairs[i][1]
  ind_1 = witheld_ligands.index(witheld_1)
  ind_2 = witheld_ligands.index(witheld_2)
  witheld_ligands[:i]
  pu_estimator =joblib.load('./%s_%s_%s.joblib'%(model_name,witheld_1, witheld_2))
  estimators_2.append(pu_estimator)

# make the ensemble list
ensemble = estimators + estimators_2 + estimators_other
#normalization vector for the outputs to the max of the training
ensemble_norm = np.load('./%s'%model_norm)

# TO USE THE ENSEMBLE where X is a feature vector
    #  predicted_values = []
    #  for j in range(20):
    #     predicted_values.append(ensemble[j].predict_proba(X)/ensemble_norm[j]

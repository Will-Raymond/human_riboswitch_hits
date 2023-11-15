  
# Predictions

* ```all_utr_predictions.npy``` - ensemble output for all UTRs  (48031 UTRs x 20 classifiers)
* ```final_set_436.json``` - UTRdb 1.0 IDs for the 436 hits from the overlap of all 20 classifiers (436 UTRs x 1 )
* ```final_set_1533.json``` - UTRdb 1.0 IDs for the 1533 hits (1533 UTRs x 1 )
* ```utr_dot_hits_1533.json``` - dot structures for the 1533 hits (1533 UTRs x 1 )
* ```utr_proba_1533.npy``` - ensemble probability of the 1533 found RS hits (20 classifiers x 1533 UTRs)
* ```utr_proba_norm.npy``` - normalization vector for ensemble output (20 classifiers x 1 )

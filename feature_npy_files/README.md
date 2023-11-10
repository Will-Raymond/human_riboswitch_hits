
# feature vector arrays

Computed N x 74 feature vector Numpy arrays
* ```X_RS_full.npy``` - Riboswitch data (67683 RS x 74)
* ```X_UTR.npy``` - 5'UTR data (48301 UTRs x 74)
* ```X_SUB.npy``` - Subsequences of the feature vector for UTR (48301 UTRs x 21 [20 subsequences + full sequence) x 74) 
* ```ids_UTR.npy``` - list of 5'UTR ids that match the feature vector array (48301 UTRs x 1)
* ```ids_RS_full.npy``` - list of RS ids that match the feature vector array (67683 RS x 1
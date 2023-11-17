
# Identification of potential riboswitch elements in *Homo Sapiens* mRNA 5'UTR sequences using Positive-Unlabeled machine learning
  

---

  

### William S. Raymond<sup>1</sup>, Jacob DeRoo<sup>1</sup>, Brian Munsky<sup>1,2</sup>

  

  

#####  <sup><sup>1</sup> School of Biomedical Engineering, Colorado State University Fort Collins, CO 80523, USA</sup>

  

  

#####  <sup><sup>2</sup> Chemical and Biological Engineering, Colorado State University Fort Collins, CO 80523, USA</sup>

  

  

![](./Figures/Abstract.png?raw=true)


This repository contains all the final data files used for the analysis in the above manuscript.

```

├─  alns/ - final alignment images for the website
│		├─ aln_%%ID%%_%%ENS_NORM%%.png
├─  data_files/ - raw/processed data files used to make the feature sets
│		├─ CCDS_nucleotide.current_11.28.2021.fa
│		├─ riboswitch_RNAcentral_8.19.21.json
│		├─ rs_dot.json
│		├─ RSid_to_ligand.json
│		├─ 5primeUTR_final_db_2.csv
│		├─ 5primeUTR_newutrdb_ML2.csv
│		├─ RS_final_db.csv
│		├─ RS_id_to_ligand.csv
│		├─ check_new_utr.py
│		├─ data_processor.py
│		├─ process_rna_central_rs_json.py
│		├─ rna_central_dot_structure_scraper.py
│		├─ species_in_RS_set.txt
├─  elkanoto_models/ - PUlearn models
│		├─ EKmodel_witheld_w_struct_features_9_26_%%LIGAND%%_%%LIGAND%%.joblib
│		├─ load_ensemble.py
│		├─ utr_proba_norm.npy
├─  ensemble_predictions/ - predictions from the ensemble classifier
│		├─ all_utr_predictions.npy
│		├─ final_set_436.json
│		├─ final_set_1533.json
│		├─ utr_dot_hits_1533.json
│		├─ utr_proba_norm.npy
│		├─ utr_proba_1533.npy
├─  feature_npy_files/ - extracted feature data arrays
│		├─ X_RS_full.npy
│		├─ X_UTR.npy
│		├─ ids_UTR.npy
│		├─ ids_RS.npy
├─  Figures/  ─  figure files for the paper
├─  GO/ ─  GO analysis output files 
│		├─ component_436.txt
│		├─ component_1533.txt
│		├─ function_436.txt
│		├─ function_1533.txt
│		├─ process_436.txt
│		├─ process_1533.txt
│		├─ gene_list_436.txt
│		├─ gene_list_1533.txt
```

Read the manuscript here:
![](./Identification%20of%20potential%20riboswitch%20elements%20in%20Homo%20Sapiens%20mRNA%205'UTR%20sequences%20using%20Positive-Unlabeled%20machine%20learning%20-%20biorxv.pdf)

Rerun the analysis here: [![Rerun the analysis here (Colab)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17zmKJh8iHAC2tImNNSyBrwUpU0uYKefx?usp=sharing)

Contact info: wsraymon@rams.colostate.edu

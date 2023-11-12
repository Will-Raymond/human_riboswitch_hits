
# Positive-Unlabeled identification of potential riboswitch elements in *Homo Sapiens* mRNA 5'UTR sequences

  

---

  
  

### William S. Raymond<sup>1</sup>, Jacob DeRoo<sup>1</sup>, Brian Munsky<sup>1,2</sup>

  

  

#####  <sup><sup>1</sup> School of Biomedical Engineering, Colorado State University Fort Collins, CO 80523, USA</sup>

  

  

#####  <sup><sup>2</sup> Chemical and Biological Engineering, Colorado State University Fort Collins, CO 80523, USA</sup>

  

  

![](./Figures/Abstract.png?raw=true)

  
  

This repository contains all the final data files used for the analysis in the above manuscript.

```
├─  alignment_matrices/
│		├─ levdist_array_1533.npy
│		├─ utr_hits_ldiff_mse_1533.npy
│		├─ utr_hits_mse_1533.npy
├─  alns/
│		├─ aln_%%ID%%_%%ENS_NORM%%.png
├─  data_files/
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
├─  elkanoto_models/
│		├─ EKmodel_witheld_w_struct_features_9_26_%%LIGAND%%_%%LIGAND%%.joblib
│		├─ load_ensemble.py
│		├─ utr_proba_norm.npy
├─  ensemble_predictions/
│		├─ all_utr_predictions.npy
│		├─ final_set_436.json
│		├─ final_set_1533.json
│		├─ utr_dot_hits_1533.json
│		├─ utr_proba_norm.npy
├─  feature_npy_files/
│		├─ X_RS_full.npy
│		├─ X_UTR.npy
│		├─ X_SUB.npy
│		├─ ids_UTR.npy
│		├─ ids_RS.npy
├─  Figures/  ─  figure files for the paper
├─  GO/
│		├─ 
```


  

[Read the manuscript here]()

  
  

Rerun the analysis here: [![Rerun the analysis here (Colab)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ThbS0ayh1q0u_45qELKpc0z-MZoVbYXp?usp=sharing)

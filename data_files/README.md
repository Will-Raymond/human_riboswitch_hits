
# Data Files

Computed N x 74 feature vector Numpy arrays
* ```CCDS_nucleotide.current_11.28.2021.fa``` - CCDS release from November 28, 2021 used to reconstruct 5'UTR + 25 nucleotide sequences
* ```riboswitch_RNAcentral_8.19.21.json``` - raw data file pulled from RNAcentral of all sequences with the tag "Riboswitch"
* ```rs_dot.json``` - Riboswitch dot structures
* ```RSid_to_ligand.json``` - RNAcentral IDs to ligand pairs for riboswitches
* ```5primeUTR_final_db_2.csv``` -  5'UTR database made from UTRdb 1.0 (2010) used for machine learning
* ```5primeUTR_newutrdb_ML2.csv``` -  remade 5'UTR database with UTRdb 2.0 (2023)
* ```RS_final_db.csv``` - Riboswitch database used for machine learning made from RNAcentral data
* ```RS_id_to_ligand.csv``` - RNAcentral IDs to ligand pairs for riboswitches

## python scripts

* ```check_new_utr.py``` - build the data file from the new UTRdb download
* ```data_processor.py``` - generalized data processor
* ```process_rna_central_rs_json.py``` - script that matches RS data to their ligand or reported RFAM ligand
* ```rna_central_dot_structure_scraper.py``` - RNAcentral scraper used to pull dot structures from their webpages (dot structures are not included within their download)

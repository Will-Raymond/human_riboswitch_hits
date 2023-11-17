
  

# Data Files

 

*  ```CCDS_nucleotide.current_11.28.2021.fa``` - CCDS release from November 28, 2021 used to reconstruct 5'UTR + 25 nucleotide sequences
*  ```riboswitch_RNAcentral_8.19.21.json``` - raw data file pulled from RNAcentral of all sequences with the tag "Riboswitch"
*  ```rs_dot.json``` - Riboswitch dot structures
*  ```RSid_to_ligand.json``` - RNAcentral IDs to ligand pairs for riboswitches
*  ```5primeUTR_final_db_2.csv``` - 5'UTR database made from UTRdb 1.0 (2010) used for machine learning
*  ```5primeUTR_newutrdb_ML2.csv``` - remade 5'UTR database with UTRdb 2.0 (2023)
*  ```RS_final_db.csv``` - Riboswitch database used for machine learning made from RNAcentral data
*  ```RS_id_to_ligand.csv``` - RNAcentral IDs to ligand pairs for riboswitches
* ```UTR_ID_to_gene.json``` - UTRdb (2010) id to gene id dictionary
* ```5UTRaspic.Hum.fasta```  - UTRdb (2010) 5'UTR download fasta
* ```/ccds_2018/``` - CCDS files used for reconstructing sequences 
  
## python scripts

 

*  ```check_new_utr.py``` - build the data file from the new UTRdb download
*  ```data_processor.py``` - generalized data processor
*  ```process_rna_central_rs_json.py``` - script that matches RS data to their ligand or reported RFAM ligand
*  ```rna_central_dot_structure_scraper.py``` - RNAcentral scraper used to pull dot structures from their webpages (dot structures are not included within their download)
* ```make_initial_UTR_csv.py``` - example of making the original data csv from the UTRdb fasta file.
* ```add_nupack_dots_to_db.ipynb``` - example notebook of making the data csvs (adding ccds ids, matching genes, making dot structures)
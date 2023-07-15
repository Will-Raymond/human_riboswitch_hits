# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 11:50:59 2022

@author: willi
"""
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



## All riboswitches were pulled from RNAcentral on 8.19.22, any entry with the tag "riboswitch"
## data was filtered for duplicates leaving (n = 73119)
## all ligands were extracted and added to a database (RSid_to_ligand.json)

# this code parses them all to the same molecule names and then plots them.

# riboswitches considered speculative and thus labeled unknown: nhA-I motif, duf1646, raiA, synthetic, sul1,
#

RS_data = json.load(open('./riboswitch_RNAcentral_8.19.21.json.json'))

ligand_data = json.load(open('./RSid_to_ligand.json'))

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

'''
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
    


RS_df = pd.DataFrame(list(zip(ids,descs,ligands,seqs)), columns=columns)

'''

count_list = list(clean_ligand_data.values())
ligand_names = list(set(count_list))
counts = np.array([count_list.count(x) for x in ligand_names])
idx = np.argsort(counts)[::-1]

sorted_counts = counts[idx]
sorted_names = [ligand_names[i] for i in idx.tolist()]
explode = [0.02]*len(sorted_names)

sub1 = sorted_counts/np.sum(sorted_counts) < .01

colors = cm.Spectral_r(np.linspace(.05,.95,len(sorted_names)))

plt.figure(dpi=300)
_,f,t = plt.pie(sorted_counts, labels = sorted_names, explode = explode, autopct='%1.1f%%',
        shadow=False, startangle=90, colors=colors, labeldistance =1.05, pctdistance=.53, textprops={'fontsize': 7})

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

plt.text(0.17,.4,'18.0%',color='r', size=7)
plt.plot([.64,.93], [.64/2.2222222222222,.41850000418500005],'r',alpha=.5)
plt.plot([0,0], [.71,1],'r',alpha=.5)

plt.text(.6,1,'Other (34 types)',color='r')
plt.text(.7,.85,'<2% each',color='r')

plt.text(1.1,-1.1,'N = %0.0f'%len(count_list))
plt.title('RS ligand representation')


species = [' '.join(rs['description'].split(' ')[:2]) for rs in RS_data]
print(len(species))
print(len(set(species)))
unique_species = list(set(species))

species_remove = ['metagenome', 'marine','human gut','unclassified',
                  'uncultured','wallaby','unidentified','synthetic',
                  'freshwater','domain','compost','blood','Chains','-MER','Riboswitch','gut','pfI','thiM',' GC',
                  'glyQ','domain']



'''
parsed_out = []
species_bact = []
for i in range(len(unique_species)):
    if any(st in unique_species[i] for st in species_remove):
        parsed_out.append(unique_species[i])
    else:
        species_bact.append(unique_species[i])

families = list(set([x.split(' ')[0] for x in species_bact]))


count_list = [x.split(' ')[0] for x in species_bact]
ligand_names = list(set(count_list))
counts = np.array([count_list.count(x) for x in ligand_names])
idx = np.argsort(counts)

sorted_counts = counts[idx]
sorted_names = [ligand_names[i] for i in idx.tolist()]
explode = [0.02]*len(sorted_names)

sub1 = sorted_counts/np.sum(sorted_counts) < .01

colors = cm.Spectral(np.linspace(.05,.95,len(sorted_names)))

plt.figure(dpi=300)
_,f,t = plt.pie(sorted_counts, labels = sorted_names, explode = explode, autopct='%1.1f%%',
        shadow=False, startangle=90, colors=colors, labeldistance =1.15, pctdistance=.53)

'''
'''
ligands = list(set(ligands))
descs_not_caught = []
ids = []

for entry in RS_data:
    desc = entry['description'].lower()
    if any(ligand in desc for ligand in ligands):
        n_caught+=1
    else:
        descs_not_caught.append(desc)
        ids.append(entry['id'])

id_to_ligand = {}
for entry in RS_data:
    desc = entry['description'].lower()
    if any(ligand in desc for ligand in ligands):
        id_to_ligand[entry['id']] = [ligand for ligand in ligands if ligand in desc ][0]


manual = {'URS000197CF52_1462':'glyq',
          'URS0000CBFF2B_2021':'guanidine-iii',
          'URS0000E60BA7_679936':'ppgpp',
          'URS00005F7350_985006':'fmn',
          'URS00002E2531_1262464':'tpp',
          'URS000080DEF0_717961':'thf',
          'URS0000273C6A_985002':'purine',
          'URS00005F7350_663951':'fmn',
          'URS000024555B_272563':'sam',
          'URS00000452A5_218494':'glycine',
          'URS000054DF16_272563':'tpp',
          'URS0002214063_1597':'fluoride',
          'URS0001E4420F_1898207':'zmp',
          'URS0000CF85F4_2043167':'fluoride',
          'URS0000C426A4_1404':'fluoride',
          'URS0000BEBEA3_317013':'fluoride',
          'URS000014D39E_1353533':'lysine',
          'URS0000211D9E_985002':'lysine',
          'URS00007C918C_1035':'fluoride',
          'URS000025F6D9_1262462':'fmn',
          'URS0000A09F21_1703966':'fmn',
          'URS0000C12306_95486':'zmp',
          'URS0000B4A76E_1402135':'sam',
          'URS0000223C9B_869213':'tpp',
          'URS000017353C_28450':'fmn',
          'URS000060F9E7_224308':'trna',
          'URS0000C46DE7_185949':'fluoride',
          'URS0000E52C98_2173034':'zmp',
          'URS00004DF65B_1226680':'tpp',
          'URS0000B55ED6_526949':'fmn',
          'URS0000BB1AEE_1402135':'glycine',
          'URS000080E047_119072':'sam',
          'URS00011BB5D8_637390':'fluoride',
          'URS000010910E_114615':'glycine',
          'URS00019B0309_2493108':'fluoride',
          'URS0000D92743_2772432':'zmp',
          'URS0000D020D0_1446794':'zmp',
          'URS00000C5777_985002':'glcyine',
          'URS00019B0309_2493107':'fluoride',
          'URS00003876AA_585035':'fmn',
          'URS0000DD3D87_2026785':'tpp',
          'URS00004CBB4B_272563':'fmn',
          'URS0000B7EF36_562':'tpp',
          'URS000080DE72_851':'fmn',
          'URS00019E2404_1748965':'fluoride',
          'URS000226EDF6_2773266':'zmp',
          'URS0000E09F9E_2249425':'fluoride',
          'URS0000C3DBE2_2766784':'zmp',
          'URS0000BAC32A_1420917':'glycine',
          'URS00005EF701_663951':'tpp',
          'URS0002300AA7_137838':'zmp',
          'URS00001B4DE8_483908':'fluoride',
          'URS00007672DD_1432049':'fmn',
          'URS00001A42A3_1262463':'cobalamin',
          'URS0000B95CA8_616991':'tpp',
          'URS00019D4100_2038397':'fluoride',
          'URS00005F1A09_1262463':'fmn',
          'URS0000379E31_1226680':'tpp',
          'URS0000355D88_291112':'tpp',
          'URS0000CDB5A2_1528773':'zmp',
          'URS000012A390_1394889':'glna',
          'URS0000D2B623_2026799':'fluoride',
          'URS00001855A3_168807':'lysine',
          'URS00003B92BC_1328306':'fmn',
          'URS0000BF3905_2493107':'fluoride',
          'URS0000C230D6_95486':'zmp',
          'URS00021E381D_553981':'fluoride',
          'URS000225E693_2773264':'fluoride',
          
          }

unconfirmed = {'URS0000E5FE41_1156935':'sul1',
               'URS0000E60464_1938756':'sul1',
               'URS0000D68ADA_1262914':'raiA',
               'URS0000D6765D_316067':'obsolete cofactor?',
               'URS0000D6B79E_338966':'obsolete cofactor?',
               'URS0000D6644D_218491':'obsolete covaftor?'}
'''
'''
print((n_caught + len(manual) + len(unconfirmed))/len(RS_data))
print(len(ids))
ids = list(set(ids) - set(list(manual.keys()))  - set(list(unconfirmed.keys()))   )
print(len(ids))
print(ids[-1])
cmd='echo '+ids[-1].strip()+'|clip'
subprocess.check_call(cmd, shell=True)
'''


# webscrape the remaining ligands:
'''    
    
from selenium import webdriver  
from selenium.common.exceptions import NoSuchElementException  
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.common.keys import Keys  

from bs4 import BeautifulSoup
import re
import os
import psutil
import tqdm as tqdm

os.environ["LANG"] = "en_US.UTF-8"
options = webdriver.FirefoxOptions()
options.add_argument('--headless')
browser = webdriver.Firefox(options=options,executable_path=GeckoDriverManager().install() )  


summaries = []
#or i in tqdm(range(len(ids))):
for i in tqdm.tqdm(range(len(ids))):
    
    missing_id = ids[i]
    browser.get(('https://rnacentral.org/rna/' + missing_id ))  
    time.sleep(3)
    html_source = browser.page_source         
    soup = BeautifulSoup(html_source,'html.parser')  
    
    try:
        summary = soup.body.findAll(text=re.compile('. Matches'))
    except:
        summary = ''
    
    summaries.append(summary)
    
    
ligands = []
for i in range(len(summaries)):
    if len(summaries[i]) > 0:
        ligands.append(summaries[i][0].split(' Matches ')[1].split('(')[1].split(',')[0].lower())
    else:
        ligands.append('')
    
extra_entries = dict(zip(ids, ligands))

'''


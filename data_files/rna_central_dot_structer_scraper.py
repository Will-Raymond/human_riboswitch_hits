# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 18:58:16 2020

@author: willi
"""

from selenium import webdriver  
from selenium.common.exceptions import NoSuchElementException  
from selenium.webdriver.common.keys import Keys  

from bs4 import BeautifulSoup
import re
import os
import psutil

import multiprocessing as mp

from bs4 import BeautifulSoup
import re
import os
import psutil

import numpy as np

import time
import pandas as pd

col=['index',	'Asc',	'Seq',	'Dot']
dotdf_final = pd.DataFrame(columns=['index',	'Asc',	'Seq',	'Dot'])

rs_pd = pd.read_csv('../RS_db.csv')

indexes = rs_pd['Unnamed: 0']
accessions = rs_pd['ID']
sequences = rs_pd['SEQ']
os.environ["LANG"] = "en_US.UTF-8"

options = webdriver.FirefoxOptions()
options.add_argument('--headless')
#options.add_argument('window-size=1920x1080')
#options.add_argument("disable-gpu")

st = time.time()
#9


#big seg 4
# to do: 35600-40000 TO RUN
# 48750 -end   running
# 17800-20000 running 
#27500-30000 running
\
seg = 1
browser = webdriver.Firefox(options=options)  



c = pd.read_csv('./dotstruct_db/combined_dotstructures.csv')
accessions_dot = c['Asc']


inds = np.linspace(0,49427,49428).astype(int)
missing_asc = []
missing_inds = []
missing_seqs = []
db_acs = list(accessions)
dot_acs = list(accessions_dot)
for i in range(0,len(db_acs)):
    if db_acs[i] not in dot_acs:
        missing_asc.append(db_acs[i])
        missing_inds.append([i])
        missing_seqs.append(sequences[i])



#for i in range(10000*seg,10000*seg + 10000):
for i in range(len(missing_asc)):
    
    if i%50 ==0:
        print(i)
                
        dotdf_final.to_csv(( 'RS_dot_structures_' +str(seg)+ '_missing.csv')) 
        #browser.close()
        #browser.quit()
        #browser = webdriver.Chrome(options=options)  
        process = psutil.Process(os.getpid())
        print(process.memory_info().rss/1e6)  # in bytes 
        print(time.time()-st)




    browser.get(('https://rnacentral.org/rna/' + missing_asc[i] ))  
    time.sleep(3)
    html_source = browser.page_source         
    soup = BeautifulSoup(html_source,'html.parser')  
    
    try:
        dot_header = soup.body.findAll(text=re.compile('Dot'))[0]
        dot_notation = dot_header.find_next().text
    except:
        dot_notation = ''
        
    newrow = [[missing_inds[i],missing_asc[i], missing_seqs[i], dot_notation ]]
    
    dotdf_final= dotdf_final.append(      pd.DataFrame(newrow,columns = col))
    
        
dotdf_final.to_csv(( 'RS_dot_structures_' +str(seg)+ '_missing.csv')) 
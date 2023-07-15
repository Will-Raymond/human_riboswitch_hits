# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:17:17 2020

@author: wsraymon
"""

from Bio import SeqIO  #BIOPYTHON
from Bio import pairwise2

import numpy as np
import itertools as it
import re
import os

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import itertools
import warnings
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, f1_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

from tensorflow import keras

import joblib 
import pickle

Sequential =  keras.models.Sequential  
Dense = keras.layers.Dense
LSTM = keras.layers.LSTM 
Conv2D = keras.layers.Conv2D
Embedding =  keras.layers.Embedding 
sequence =  keras.preprocessing.sequence
Flatten = keras.layers.Flatten
MaxPooling2D = keras.layers.MaxPooling2D
LSTM = keras.layers.LSTM 
Embedding =  keras.layers.Embedding 
sequence =  keras.preprocessing.sequence


class KNN():
    def __init__(self, n_neighbors, weights):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights =weights )
    def fit(self,kmer_training, kmer_test, labels_training, labels_test ):
        self.model.fit(kmer_training, labels_training)
       
        
        self.acc = 1- np.sum ( np.abs(labels_test - self.model.predict(kmer_test)) )/len(labels_test)

    def load_model(self, filename):
        self.model = pickle.load(open(filename,'rb'))
    


class RF():
    def __init__(self, n_trees, seed, max_features, criterion):
        
        self.model = RandomForestClassifier(n_estimators=n_trees, 
                               bootstrap = True,
                               max_features = max_features,
                               random_state = seed,
                               criterion = criterion,
                               verbose=1)
        self.seed = seed
        
    def fit(self,kmer_training, kmer_test, labels_training, labels_test):
        self.model.fit(kmer_training, labels_training)
        
        self.acc = 1- np.sum ( np.abs(labels_test - self.model.predict(kmer_test)) )/len(labels_test)

    def load_model(self, filename):
        self.model = None
        self.model = joblib.load(filename)
        


class LSTMNN():
    
    def __init__(self, input_length, n_neurons, activation,vectors_per_char, kmer ):

        #Defaults
            
        # input_length = 300

        # n_neruons = 100
        # activation = 'sigmoid'
        # vectors_per_char = 32
    
        
        self.model = Sequential()
        self.model.add(Embedding(1024, vectors_per_char, input_length=input_length)) 
        self.model.add(LSTM(n_neurons))   
     
        self.model.add(Dense(1, activation = activation ))            
            
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.kmer = kmer
        
    
    def fit(self,kmer_training, kmer_test, labels_training, labels_test,   n_epochs = 20, batch_size = 200, normalize = False):
        print('------------------------------------------------------')
        print('Running LSTM training with the following conditions: ')
        print('n epochs: %i    batches: %i '%(n_epochs,batch_size) )
        print('total testing data: %i    total training: %i '%(int(len(labels_training)), int(len(labels_test))  ))
        print('------------------------------------------------------')
        print(self.model.summary())
        if normalize == True:
            kmer_training = kmer_training/max([kmer_training.max(), kmer_test.max()])
            kmer_test = kmer_test/max([kmer_training.max(), kmer_test.max()])

        
        self.model.fit(kmer_training, labels_training, epochs=n_epochs, batch_size=batch_size, verbose=1,validation_data=(kmer_test, labels_test)) 
        self.scores = self.model.evaluate(kmer_test, labels_test, verbose=0 )
        
    def load_model(self,file):
        self.model.load_weights(file)       
        




class CNN():
    
    def __init__(self, input_shape, conv_layers, conv_sizes, conv_kernels,  dense_layers, neurons, activations ):
        #Defaults
            
        # input_dim = 8x8x1

        # conv layers = 4
        # conv_sizes = [50,46,30,16]
        # conv_kernels = [(2,2),(2,2),(2,2),(2,2) ]
        
        
        # dense_layers = 4
        # neurons = [256,128,64,32,2]
        # activations = ['relu','relu','relu','relu','relu','relu','relu','softmax']
        
        
        self.model = Sequential()  #convolutions
        for i in range(conv_layers):
            if i == 0:
                self.model.add(Conv2D (conv_sizes[i],conv_kernels[i],  input_shape=input_shape, activation=activations[i]))
                #self.model.add(MaxPooling2D((2, 2)))
            else:
                
                self.model.add(Conv2D( conv_sizes[i],conv_kernels[i] , activation=activations[i]))
                #self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Flatten())  #flatten 
        
        for i in range(dense_layers):
                self.model.add(Dense(neurons[i], activations[i]))
                
        self.model.add(Dense(neurons[-1], activation = activations[-1] ))            
            
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
    
    def fit(self,kmer_training, kmer_test, labels_training, labels_test,   n_epochs = 20, batch_size = 200, normalize = False):
        print('------------------------------------------------------')
        print('Running CNN training with the following conditions: ')
        print('n epochs: %i    batches: %i    normalized to 0-1:  %i'%(n_epochs,batch_size,normalize) )
        print('total testing data: %i    total training: %i '%(int(len(labels_training)), int(len(labels_test))  ))
        print('------------------------------------------------------')
        print(self.model.summary())
        if normalize == True:
            kmer_training = kmer_training/max([kmer_training.max(), kmer_test.max()])
            kmer_test = kmer_test/max([kmer_training.max(), kmer_test.max()])

        
        self.model.fit(kmer_training, labels_training, epochs=n_epochs, batch_size=batch_size, verbose=1,validation_data=(kmer_test, labels_test)) 
        self.scores = self.model.evaluate(kmer_test, labels_test, verbose=0 )
        
    def load_model(self,file):
        self.model.load_weights(file)       
        
    




class ffNN():
    
    def __init__(self, input_dim=64, layers = 6, neurons = [100,50,25,10,4,1] , activations =['relu', 'relu', 'relu', 'relu', 'relu', 'sigmoid'] ):
        #Defaults
            
        # input_dim = 64
        # layers = 6
        # neurons = [100,50,25,10,4,1]
        # activations = ['relu','relu','relu','relu','relu','sigmoid']
            
        
        self.model = Sequential()
        for i in range(layers-1):
            if i == 0:
                self.model.add(Dense(neurons[i], input_dim=input_dim, activation=activations[i]))
            else:
                self.model.add(Dense(neurons[i], activations[i]))
        self.model.add(Dense(neurons[-1], activation = activations[-1] ))
        
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
    
    def fit(self,kmer_training, kmer_test, labels_training, labels_test,   n_epochs = 20, batch_size = 200, normalize = True):
        print('------------------------------------------------------')
        print('Running FFNN training with the following conditions: ')
        print('n epochs: %i    batches: %i    normalized to 0-1:  %i'%(n_epochs,batch_size,normalize) )
        print('total testing data: %i    total training: %i '%(int(len(labels_training)), int(len(labels_test))  ))
        print('------------------------------------------------------')
        print(self.model.summary())
        if normalize == True:
            kmer_training = kmer_training/max([kmer_training.max(), kmer_test.max()])
            kmer_test = kmer_test/max([kmer_training.max(), kmer_test.max()])

        
        self.model.fit(kmer_training, labels_training, epochs=n_epochs, batch_size=batch_size, verbose=1,validation_data=(kmer_test, labels_test)) 
        self.scores = self.model.evaluate(kmer_test, labels_test, verbose=0 )
        
        
        
    def load_model(self,file):
        self.model.load_weights(file)
        
        
        

class RS_ML:
    def __init__(self):
        self.training_test_processed = False 
        self.training_positive = None  #dataframe
        self.training_negative = None
        self.training_array = None
        
        self.testing_set = None
        
        self.model = None
        
        
        self.default_model_args = {}
        self.default_model_args['ffnn'] = {'input_dim': 64,
                                           'layers': 6,
                                           'neurons': [100, 50, 25, 10, 4, 1],
                                           'activations': ['relu', 'relu', 'relu', 'relu', 'relu', 'sigmoid']}
        
        self.default_model_args['cnn'] = {'input_shape': (8,8,1),
                                          'conv_layers':4,
                                          'conv_kernels':  [(3,3),(3,3),(2,2),(2,2) ],
                                          'conv_sizes': [128,64,32,16],
                                          'dense_layers': 4,
                                          'neurons': [512,256,128,64,2],
                                          'activations': ['relu', 'relu', 'relu', 'relu', 'relu','relu','relu', 'softmax']}        
        
        
        self.default_model_args['lstm'] = {'input_length': 300,
                                           'n_neurons': 200,
                                           'vectors_per_char':32,
                                           'activation': 'sigmoid',
                                           'kmer':3}
        
    
        self.default_model_args['rf'] = {'n_trees':100, 'seed':10, 'criterion':'gini','max_features':'log2'}
        self.default_model_args['knn'] = {'n_neighbors':2}
        
        self.default_data_args = {'witheld_percentage':.2,'seed':10 }
        
        self.default_training_args = {}
        self.default_training_args['ffnn'] = {'n_epochs': 20, 'batch_size': 200, 'normalize': False}
        self.default_training_args['cnn'] = {'n_epochs': 20, 'batch_size': 500, 'normalize': False}
        self.default_training_args['lstm'] = {'n_epochs': 5, 'batch_size': 500, 'normalize': False}
        self.default_training_args['knn'] = {}
        
        self.default_training_args['rf'] = {}
    
        
    def compare_training_set_length_distb(self):
        
        ncrna_kmer_sum = np.sum(self.training_negative.iloc[0:,3:].values,axis=1)
        rs_kmer_sum = np.sum(self.training_positive.iloc[0:,3:].values,axis=1)
        
        #print(np.mean(rs_kmer_sum),np.var(rs_kmer_sum))
        #print(np.mean(ncrna_kmer_sum),np.var(ncrna_kmer_sum))
        
        
        #Check that the two databases have similar length sizes
        plt.hist(ncrna_kmer_sum,100,density=True,alpha=.5)
        plt.hist(rs_kmer_sum,100,density=True,alpha = .5)
        
        
    def load_training_data_norm(self,training_positive, training_negative, max_neg = None, lengthlimit = 20):
        self.training_positive = pd.read_csv(training_positive) 
        self.training_negative = pd.read_csv(training_negative)
        if np.sum(self.training_negative.columns[3:] != self.training_positive.columns[3:]) > 0:
            print('These files headers do not match, double check they are the same data format' )
            self.training_positive = None
            self.training_negative = None       
            
        if max_neg == None:
            max_neg = -1
        
        
        print('total positive samples before length limit: %i'%self.training_positive.shape[0] )
        print('total negative samples before length limit: %i'%self.training_negative.shape[0] )
        positive_seq_lens = [ len(self.training_positive['SEQ'][x]) for x in range(len(self.training_positive['SEQ'])) ]
        self.training_positive['LEN'] = positive_seq_lens
        
        self.training_positive =self.training_positive[self.training_positive['LEN'] > lengthlimit]
        
        negative_seq_lens = [ len(self.training_negative['SEQ'][x]) for x in range(len(self.training_negative['SEQ'])) ]
        self.training_negative['LEN'] = negative_seq_lens        
        
        
        self.training_negative =self.training_negative[self.training_negative['LEN'] > lengthlimit]
        
        training_array = self.training_positive.iloc[0:,3:-1].values
        labels = np.ones(training_array.shape[0])
        
        self.training_array = np.vstack((training_array,self.training_negative.iloc[0:max_neg,3:-1].values))
        
        
        
        self.labels = np.hstack((labels,np.zeros(self.training_negative.iloc[0:max_neg,3:-1].values.shape[0] ) ))
        self.labels_2d = np.vstack((self.labels,np.abs(self.labels-1))).T
        self.positive_samples = self.training_positive.iloc[0:,3:].values.shape[0]
        self.negative_samples = self.training_negative.iloc[0:max_neg,3:].values.shape[0]
        print('total positive samples after length limit: %i'%self.training_positive.shape[0] )
        print('total negative samples after length limit: %i'%self.training_negative.shape[0] )        
        
                
        
        

    def load_training_data(self,training_positive, training_negative, max_neg = None):
        self.training_positive = pd.read_csv(training_positive) 
        self.training_negative = pd.read_csv(training_negative)
        if np.sum(self.training_negative.columns[3:] != self.training_positive.columns[3:]) > 0:
            print('These files headers do not match, double check they are the same data format' )
            self.training_positive = None
            self.training_negative = None       
            
        if max_neg == None:
            max_neg = -1
        
        training_array = self.training_positive.iloc[0:,3:].values
        labels = np.ones(training_array.shape[0])
        
        self.training_array = np.vstack((training_array,self.training_negative.iloc[0:max_neg,3:].values))
        
        
        
        self.labels = np.hstack((labels,np.zeros(self.training_negative.iloc[0:max_neg,3:].values.shape[0] ) ))
        self.labels_2d = np.vstack((self.labels,np.abs(self.labels-1))).T
        self.positive_samples = self.training_positive.iloc[0:,3:].values.shape[0]
        self.negative_samples = self.training_negative.iloc[0:max_neg,3:].values.shape[0]
        
        
        
        #self.training_set = self.training_positive.columns[3:], self.training_negative.columns 
        
        
    def format_training_data(self,data,labels, witheld_percentage =.2, seed = 10 ):
        kmer_train, kmer_test, label_train, label_test = train_test_split(data, labels, test_size=witheld_percentage,random_state=seed)
        return kmer_train, kmer_test, label_train, label_test

    def convert_to_kmer_ids(self, seq, kmer_list, maxlen=300):
        inds = [0,4,16,64,4**4,4**5]
        n = inds.index(len(kmer_list))
        
        seq = seq[:maxlen]
        if len(seq) < n:
            return np.zeros(maxlen)
            
            
        kmer_inds = np.zeros(len(seq) - n)
    
        for i in range(0,len(seq)-n):   
            kmer_inds[i] = int(kmer_list.index(seq[i:i+n]))+1 
        kmer_inds = sequence.pad_sequences([kmer_inds], maxlen=maxlen)
        return kmer_inds 
    


    def generate_lstm_training_data(self, maxlen=300, kmer=3, max_neg = -1):
        p_seq = self.training_positive['SEQ']
        n_seq = self.training_negative['SEQ'][0:max_neg]
        seqs = list(p_seq.append(n_seq))
        kmer_str = [ ''.join(x) for x in  list(itertools.product(['a','c','u','g'], repeat = kmer)  )]

        self.training_array_lstm = np.zeros((len(p_seq)+len(n_seq), maxlen  ))
        
        for i in range(len(seqs)):
           self.training_array_lstm[i,:] = self.convert_to_kmer_ids(seqs[i],kmer_str,maxlen=maxlen)
        
        

               
        
        
        


    def resize_for_cnn(self,data, x=8,y=8):
        return np.reshape(data, (data.shape[0],x,y,1))
        
    
    def save_model(self,filename):
        if self.model_type in ['ffnn','cnn', 'lstm']:
            self.model.model.save(filename)
        if self.model_type == 'rf':
            joblib.dump(self.model.model, filename) 
        if self.model_type == 'knn':
            with open(filename, 'wb') as f:
                pickle.dump(self.model.model, f)   
      
        
            
    
    
    def load_model(self,model_type, model_file, modelargs = None):
        if modelargs == None:
            modelargs = self.default_model_args[model_type.lower()]
 
        if model_type.lower() == 'ffnn':
            self.model_type = 'ffnn'
            self.model = ffNN( **modelargs)
            self.model.load_model(model_file)
            
        if model_type.lower() == 'cnn':
            self.model_type = 'cnn'
            self.model = CNN( **modelargs)
            self.model.load_model(model_file)         
            
        if model_type.lower() == 'lstm':
            self.model_type = 'lstm'
            self.model = LSTMNN( **modelargs)
            self.model.load_model(model_file) 
            
        if model_type.lower() == 'knn':
            self.model_type = 'knn'
            self.model = KNN( **modelargs)
            self.model.load_model(model_file) 
            
        if model_type.lower() == 'rf':
            self.model_type = 'rf'
            self.model = RF( **modelargs)
            self.model.load_model(model_file) 
            
        self.model_trained = True
        
        
    
    def make_model_ffnn(self, 
                   activations = ['relu', 'relu', 'relu', 'relu', 'relu', 'sigmoid'],
                   layers = 6,
                   neurons=[100,50,25,10,4,1],
                   input_dim = 64):

        return ffNN(input_dim, layers,neurons,activations)
            
        
        
        
        
    

    def train_model(self, model_type, modelargs = None, dataargs = None, trainingargs = None):
        if modelargs == None:
            modelargs = self.default_model_args[model_type.lower()]
 
        if dataargs == None:
            dataargs = self.default_data_args
            
        if trainingargs == None:
            trainingargs = self.default_training_args[model_type.lower()]
            
        if model_type.lower() in ['ffnn','knn','rf']:
            self.model_type = model_type.lower()
            if model_type.lower() == 'ffnn':
                self.model = ffNN( **modelargs)
            if model_type.lower() == 'knn':
                self.model = KNN( **modelargs)
            if model_type.lower() == 'rf':
                self.model = RF( **modelargs)            
            self.model.fit( *self.format_training_data(self.training_array,self.labels,**dataargs),**trainingargs  )
            
        if model_type.lower() == 'cnn':
            self.model_type = 'cnn'
      
            self.model = CNN( **modelargs)
            
      
            kmer_train, kmer_test, label_train, label_test = self.format_training_data(self.training_array,self.labels_2d)
            kmer_train = self.resize_for_cnn(kmer_train)
            kmer_test = self.resize_for_cnn(kmer_test)         
                      
            self.model.fit(kmer_train, kmer_test, label_train, label_test,**trainingargs  )    
            
        if model_type.lower() in ['lstm']:
            self.model_type = model_type.lower()
          
            self.model = LSTMNN( **modelargs)
            
            self.generate_lstm_training_data( maxlen = modelargs['input_length'], kmer = modelargs['kmer'] )
            
            self.model.fit( *self.format_training_data(self.training_array_lstm,self.labels,**dataargs),**trainingargs  )
                
            
        
        self.model_trained = True



    def model_eval_lstm(self, kmer_csv_file, kmer=3, sliding_window = 5, verbose= True, maxlen= 300,probability_limit= .9,length_limit= 300, save = True):
        if not self.model_trained:
            return 
        utr_data = pd.read_csv(kmer_csv_file) 
        
        kmer_str = [ ''.join(x) for x in  list(itertools.product(['a','c','u','g'], repeat = kmer)  )]
        
        st = time.time()
        
        kmers_array = utr_data[kmer_str].values

        
        potential_hits = []
        
        
        
        for i in range(kmers_array.shape[0]):
            
            
            
            if verbose:
                if i %1000 == 0 and i != 0:
                    
                    print('processed %i / %i sequences...' % (i,kmers_array.shape[0])  )
                    print('time finished: %f minutes' % ( (kmers_array.shape[0] / i)*((time.time()-st)/60)  - ((time.time()-st)/60)  )   )
                    if save:
                        if i %1000 == 0 and i != 0:
                            potentialhits_df = pd.DataFrame(potential_hits)
                            potentialhits_df.columns = ['index','Asc','Seq','pRS','prob != RS','length bp', 'best subseq', 'length subseq','subseq pRS']
                            potentialhits_df = potentialhits_df.sort_values(by ='subseq pRS',ascending=False)
                            
                            rshp = RS_HitsProcessor()
                            rshp.load_hits_df(potentialhits_df)
                            rshp.prune_hits(probability_limit,length_limit )     
                            rshp.pruned_dataframe.to_csv('tmp.csv')         
            
            seq = utr_data['SEQ'][i].lower()
            
            if len(seq) < kmer:
                continue
            
            if len(seq) > maxlen:
                tmp_seqs = [seq[i:i+maxlen] for i in range(0, len(seq), maxlen)]
                maxprs = 0
                for j in range(len(tmp_seqs)):
                
                    kmer_inds = self.convert_to_kmer_ids(tmp_seqs[j], kmer_str,maxlen=maxlen )
                    pRS = self.model.model.predict( np.atleast_2d(kmer_inds))
                    if maxprs  < pRS:
                        maxprs = maxprs
                        bestseq = tmp_seqs[j]
            else:
                kmer_inds = self.convert_to_kmer_ids(seq, kmer_str,maxlen = maxlen )
                
                pRS = self.model.model.predict( np.atleast_2d(kmer_inds))       
                bestseq = seq
                maxprs = pRS
                
                
            asc = utr_data['ID'][i]
            
            index = utr_data['Unnamed: 0'][i]
            length = len(seq)
            
            
            if maxprs > .5:  #potentially a riboswitch run subsequence
                
                windows = self.sliding_window(bestseq, sliding_window)
                subseqs = self.generate_subseqs(bestseq, windows)
                
        
                flat_subseqs = [item for sublist in subseqs for item in sublist]
                subseq_prs_array = np.zeros( (len(flat_subseqs),maxlen) ).astype(int)
                for j in range(len(flat_subseqs)):
                   
                    subseq_prs_array[j,:] = self.convert_to_kmer_ids(flat_subseqs[j], kmer_str,maxlen=maxlen )
                    
                
                sub_pRS = self.model.model.predict( subseq_prs_array)
                
                best_subseq = flat_subseqs[np.argmax(sub_pRS)]
                len_subseq = len(best_subseq)
                
                sub_prs = sub_pRS[np.argmax(sub_pRS)][0]
                
            else:
                best_subseq = bestseq
                len_subseq = length
                sub_prs = pRS[0][0]
                
            pRS = pRS[0][0]
                

            
            potential_hits.append([index,asc,seq,pRS,1-pRS,length,best_subseq,len_subseq,sub_prs])
        
        potentialhits_df = pd.DataFrame(potential_hits)
        potentialhits_df.columns = ['index','Asc','Seq','pRS','prob != RS','length bp', 'best subseq', 'length subseq','subseq pRS']
        potentialhits_df = potentialhits_df.sort_values(by ='subseq pRS',ascending=False)
        
        rshp = RS_HitsProcessor()
        rshp.load_hits_df(potentialhits_df)
        rshp.prune_hits(probability_limit,length_limit )
        
        return rshp.pruned_dataframe


    def model_eval_rf(self, kmer_csv_file, kmer=3, sliding_window = 5, verbose= True,probability_limit= .9,length_limit= 300, save= True ):
        if not self.model_trained:
            return 
        utr_data = pd.read_csv(kmer_csv_file) 
        
        kmer_str = [ ''.join(x) for x in  list(itertools.product(['a','c','u','g'], repeat = kmer)  )]
        
        st = time.time()
        
        kmers_array = utr_data[kmer_str].values
        
        if self.model_type == 'cnn':
            kmers_array = np.reshape(kmers_array, (kmers_array.shape[0],8,8,1)  )      
        
        potential_hits = []
        for i in range(kmers_array.shape[0]):
            
            if verbose:
                if i %1000 == 0 and i != 0:
                    
                    print('processed %i / %i sequences...' % (i,kmers_array.shape[0])  )
                    print('time finished: %f minutes' % ( (kmers_array.shape[0] / i)*((time.time()-st)/60)  - ((time.time()-st)/60)  )   )
                    
                    if save:
                        if i %1000 == 0 and i != 0:
                            potentialhits_df = pd.DataFrame(potential_hits)
                            potentialhits_df.columns = ['index','Asc','Seq','pRS','prob != RS','length bp', 'best subseq', 'length subseq','subseq pRS']
                            potentialhits_df = potentialhits_df.sort_values(by ='subseq pRS',ascending=False)
                            
                            rshp = RS_HitsProcessor()
                            rshp.load_hits_df(potentialhits_df)
                            rshp.prune_hits(probability_limit,length_limit )     
                            rshp.pruned_dataframe.to_csv('tmp.csv')
                        
            
            if self.model_type == 'cnn':
                pRS = self.model.model.predict_proba( np.array([np.atleast_2d( kmers_array[i])]     ))
                pRS = pRS.T
            else:
                pRS = self.model.model.predict_proba( np.atleast_2d( kmers_array[i]     ))
                
            
            
            asc = utr_data['ID'][i]
            seq = utr_data['SEQ'][i].lower()
            index = utr_data['Unnamed: 0'][i]
            length = len(seq)
            
            #print(pRS)
            

            
            if len(pRS) == 1:
                
                if pRS[0][0] > .5:  #potentially a riboswitch run subsequence
                    
                    windows = self.sliding_window(seq, sliding_window)
                    subseqs = self.generate_subseqs(seq, windows)
                    
            
                    flat_subseqs = [item for sublist in subseqs for item in sublist]
                    subseq_prs_array = np.zeros( (len(flat_subseqs),len(kmer_str)) ).astype(int)
                    for j in range(len(flat_subseqs)):
                       
                        subseq_prs_array[j,:] = self.kmer_freq(flat_subseqs[j], kmer_str)
                    
                    sub_pRS = self.model.model.predict_proba( subseq_prs_array)[0,:]
                    
                    best_subseq = flat_subseqs[np.argmax(sub_pRS)]
                    len_subseq = len(best_subseq)
                    
                    sub_prs = sub_pRS[np.argmax(sub_pRS)]
                    
                else:
                    best_subseq = ''
                    len_subseq = 0
                    sub_prs = 0
                
                pRS = sub_prs
              
                    
            else:
                pRS = pRS.flatten()[0]
                
                if pRS > .5:  #potentially a riboswitch run subsequence
                    
                    windows = self.sliding_window(seq, sliding_window)
                    subseqs = self.generate_subseqs(seq, windows)
                    
            
                    flat_subseqs = [item for sublist in subseqs for item in sublist]
                    subseq_prs_array = np.zeros( (len(flat_subseqs),len(kmer_str)) ).astype(int)
                    for j in range(len(flat_subseqs)):
                       
                        subseq_prs_array[j,:] = self.kmer_freq(flat_subseqs[j], kmer_str)
                    
                    subseq_prs_array = self.resize_for_cnn(subseq_prs_array)
                    sub_pRS = self.model.model.predict_proba( subseq_prs_array)[:,0]
                    
                    best_subseq = flat_subseqs[np.argmax(sub_pRS)]
                    len_subseq = len(best_subseq)
                    
                    sub_prs = sub_pRS[np.argmax(sub_pRS)]
                    
                else:
                    best_subseq = ''
                    len_subseq = 0
                    sub_prs = 0
                    
                pRS = sub_prs
            
            #print([index,asc,seq,pRS,1-pRS,length,best_subseq,len_subseq,sub_prs])
            potential_hits.append([index,asc,seq,pRS,1-pRS,length,best_subseq,len_subseq,sub_prs])
        
        print([index,asc,seq,pRS,1-pRS,length,best_subseq,len_subseq,sub_prs])
        
        potentialhits_df = pd.DataFrame(potential_hits)
        potentialhits_df.columns = ['index','Asc','Seq','pRS','prob != RS','length bp', 'best subseq', 'length subseq','subseq pRS']
        potentialhits_df = potentialhits_df.sort_values(by ='subseq pRS',ascending=False)
        
        rshp = RS_HitsProcessor()
        rshp.load_hits_df(potentialhits_df)
        rshp.prune_hits(probability_limit,length_limit )
        
        
        return rshp.pruned_dataframe

    def model_eval(self, kmer_csv_file, kmer=3, sliding_window = 5, verbose= True,probability_limit= .9,length_limit= 300, save= True ):
        if not self.model_trained:
            return 
        utr_data = pd.read_csv(kmer_csv_file) 
        
        kmer_str = [ ''.join(x) for x in  list(itertools.product(['a','c','u','g'], repeat = kmer)  )]
        
        st = time.time()
        
        kmers_array = utr_data[kmer_str].values
        
        if self.model_type == 'cnn':
            kmers_array = np.reshape(kmers_array, (kmers_array.shape[0],8,8,1)  )      
        
        potential_hits = []
        for i in range(kmers_array.shape[0]):
            
            if verbose:
                if i %1000 == 0 and i != 0:
                    
                    print('processed %i / %i sequences...' % (i,kmers_array.shape[0])  )
                    print('time finished: %f minutes' % ( (kmers_array.shape[0] / i)*((time.time()-st)/60)  - ((time.time()-st)/60)  )   )
                    
                    if save:
                        if i %1000 == 0 and i != 0:
                            potentialhits_df = pd.DataFrame(potential_hits)
                            potentialhits_df.columns = ['index','Asc','Seq','pRS','prob != RS','length bp', 'best subseq', 'length subseq','subseq pRS']
                            potentialhits_df = potentialhits_df.sort_values(by ='subseq pRS',ascending=False)
                            
                            rshp = RS_HitsProcessor()
                            rshp.load_hits_df(potentialhits_df)
                            rshp.prune_hits(probability_limit,length_limit )     
                            rshp.pruned_dataframe.to_csv('tmp.csv')
                        
            
            if self.model_type == 'cnn':
                pRS = self.model.model.predict( np.array([np.atleast_2d( kmers_array[i])]     ))
                pRS = pRS.T
            else:
                pRS = self.model.model.predict( np.atleast_2d( kmers_array[i]     ))
                
            
            
            asc = utr_data['ID'][i]
            seq = utr_data['SEQ'][i].lower()
            index = utr_data['Unnamed: 0'][i]
            length = len(seq)
            
            #print(pRS)
            

            
            if len(pRS) == 1:
                
                if pRS[0] > .5:  #potentially a riboswitch run subsequence
                    
                    windows = self.sliding_window(seq, sliding_window)
                    subseqs = self.generate_subseqs(seq, windows)
                    
            
                    flat_subseqs = [item for sublist in subseqs for item in sublist]
                    subseq_prs_array = np.zeros( (len(flat_subseqs),len(kmer_str)) ).astype(int)
                    for j in range(len(flat_subseqs)):
                       
                        subseq_prs_array[j,:] = self.kmer_freq(flat_subseqs[j], kmer_str)
                    
                    sub_pRS = self.model.model.predict( subseq_prs_array)
                    
                    best_subseq = flat_subseqs[np.argmax(sub_pRS)]
                    len_subseq = len(best_subseq)
                    
                    sub_prs = sub_pRS[np.argmax(sub_pRS)]
                    
                else:
                    best_subseq = ''
                    len_subseq = 0
                    sub_prs = 0
                
                pRS = sub_prs
              
                    
            else:
                pRS = pRS.flatten()[0]
                
                if pRS > .5:  #potentially a riboswitch run subsequence
                    
                    windows = self.sliding_window(seq, sliding_window)
                    subseqs = self.generate_subseqs(seq, windows)
                    
            
                    flat_subseqs = [item for sublist in subseqs for item in sublist]
                    subseq_prs_array = np.zeros( (len(flat_subseqs),len(kmer_str)) ).astype(int)
                    for j in range(len(flat_subseqs)):
                       
                        subseq_prs_array[j,:] = self.kmer_freq(flat_subseqs[j], kmer_str)
                    
                    subseq_prs_array = self.resize_for_cnn(subseq_prs_array)
                    sub_pRS = self.model.model.predict( subseq_prs_array)[:,0]
                    
                    best_subseq = flat_subseqs[np.argmax(sub_pRS)]
                    len_subseq = len(best_subseq)
                    
                    sub_prs = sub_pRS[np.argmax(sub_pRS)]
                    
                else:
                    best_subseq = ''
                    len_subseq = 0
                    sub_prs = 0
                    
                pRS = sub_prs
            
            #print([index,asc,seq,pRS,1-pRS,length,best_subseq,len_subseq,sub_prs])
            potential_hits.append([index,asc,seq,pRS,1-pRS,length,best_subseq,len_subseq,sub_prs])
        
        
        
        potentialhits_df = pd.DataFrame(potential_hits)
        potentialhits_df.columns = ['index','Asc','Seq','pRS','prob != RS','length bp', 'best subseq', 'length subseq','subseq pRS']
        potentialhits_df = potentialhits_df.sort_values(by ='subseq pRS',ascending=False)
        
        rshp = RS_HitsProcessor()
        rshp.load_hits_df(potentialhits_df)
        rshp.prune_hits(probability_limit,length_limit )
        
        
        return rshp.pruned_dataframe
        
        
    def sliding_window(self,sequence, nsteps):
        sequence_length = len(sequence)
        segments = np.geomspace( np.floor(.25*sequence_length)+1, sequence_length, nsteps ).astype(int)
        if segments[-1] != len(sequence):
            segments[-1] = len(sequence)
    
        return segments
    
    def generate_subseqs(self,sequence, windows):
        total_str = np.sum(len(sequence) - np.array(windows))
        
        if total_str > 100:
            warnings.warn("This generates %i substrings"%total_str, ResourceWarning)
            
        subseqs = []
        seqs = []
        for window in windows:
            seqs = [sequence[i:i+window] for i in range(len(sequence)+1-window)]
            subseqs.append(seqs)
        return subseqs        
    
    
    def kmer_freq(self,seq,kmer_ind ):
        '''
        calculate the kmer frequences of k size for seq
        '''
        k = len(kmer_ind[0])
        kmer_freq_vec = np.zeros(len(kmer_ind)).astype(int)
        for i in range(len(seq)-k):
            kmer_freq_vec[kmer_ind.index(seq[i:i+k])] += 1
            
        return kmer_freq_vec        
    
    
    
    
    def __train_all(self, model_dir, model_names = None):
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # print('Warning this will take a LONG time')
        # print('Starting at:')
        # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) )
        # print('_____________________________________________________')
        
        # print('training FFNN.....')
        # ffnn_args = self.default_model_args['ffnn']
        # ffnn_args['neurons'] = [400,600,300,100,25,1]
        # self.train_model('ffnn', modelargs = ffnn_args)
        # if model_names != None:
        #     self.save_model(  ('./' + model_dir + '/' + model_names[0]+ '.h5') )
        # else:
        #     self.save_model(  ('./' + model_dir + '/' +'ffnn.h5') )
        # print('completed at:')
        # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) )
        # print('_____________________________________________________')
        
        
        # print('training CNN.....')
        # self.train_model('cnn')
        # if model_names != None:
        #     self.save_model(  ('./' + model_dir + '/' + model_names[1]+ '.h5') )
        # else:
        #     self.save_model(  ('./' + model_dir + '/' +'cnn.h5') )        
        # print('completed at:')
        # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) )
        # print('_____________________________________________________')
        
        
        
        # print('training KNN.....')
        # self.train_model('knn')
        # if model_names != None:
        #     self.save_model(  ('./' + model_dir + '/' + model_names[2]+ '.p') )
        # else:
        #     self.save_model(  ('./' + model_dir + '/' +'knn.p') )   
        # print('completed at:')
        # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) )
        # print('_____________________________________________________')
        
        
        # print('training RF.....')
        # self.train_model('rf')
        # if model_names != None:
        #     self.save_model(  ('./' + model_dir + '/' + model_names[3]+ '.sav') )
        # else:
        #     self.save_model(  ('./' + model_dir + '/' +'rf.sav') )    
        # print('completed at:')
        # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) )
        # print('_____________________________________________________')
        
        
        print('training LSTM.....')
        
        self.train_model('lstm')
        if model_names != None:
            self.save_model(  ('./' + model_dir + '/' + model_names[3]+ '.h5') )
        else:
            self.save_model(  ('./' + model_dir + '/' +'lstm.h5') )  
        print('completed at:')
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) )
        print('_____________________________________________________')          
            

    def __test_model_folder(self, model_dir, utr_file ):
        
        print('Warning this will take a LONG time')
        print('Starting at:')
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) )
        print('_____________________________________________________')
               
        
        # self.load_model('ffnn', ('./' + model_dir +   '/ffnn.h5')  )
        # ffnn_hits = self.model_eval(utr_file, kmer=3, sliding_window = 5, verbose= True, save = False)
        
        # ffnn_hits.to_csv( ('./' + model_dir +   '/ffnn_hits.csv')  )

        self.load_model('knn', ('./' + model_dir +   '/knn.p')  )
        knn_hits = self.model_eval(utr_file, kmer=3, sliding_window = 5, verbose= True)
        
        knn_hits.to_csv( ('./' + model_dir +   '/knn_hits.csv')  )


        # self.load_model('lstm', ('./' + model_dir +   '/lstm.h5')  )
        # lstm_hits = self.model_eval_lstm(utr_file, kmer=3, sliding_window = 5, verbose= True, save = False)        
        # lstm_hits.to_csv( ('./' + model_dir +   '/lstm_hits.csv')  )

        # self.load_model('cnn', ('./' + model_dir +   '/cnn.h5')  )
        # cnn_hits = self.model_eval(utr_file, kmer=3, sliding_window = 5, verbose= True, save = False)        
        # cnn_hits.to_csv( ('./' + model_dir +   '/cnn_hits.csv')  )


        # self.load_model('rf', ('./' + model_dir +   '/rf.sav')  )
        # self.model.model.verbose = False
        # rf_hits = self.model_eval_rf(utr_file, kmer=3, sliding_window = 5, verbose= True, save=False)
        # rf_hits.to_csv( ('./' + model_dir +   '/rf_hits.csv')  )

    def __eval_model_folder(self, model_dir):
        
        scores = {}
        dataargs = self.default_data_args
        x,xtest, y,ytest = self.format_training_data(self.training_array,self.labels,**dataargs)
        
        self.load_model('ffnn', ('./' + model_dir +   '/ffnn.h5')  )
        ypred = self.model.model.predict(xtest)
        yclass = np.array(ypred >.5).astype(int)
        acc = np.sum(np.abs((yclass.flatten() - ytest.flatten())))/len(yclass)
        model_fpr, model_tpr = self.__get_roc(ypred,ytest)
        f1 = f1_score(ytest,yclass)
        conf = confusion_matrix(ytest,yclass)
        
        scores['ffnn'] = {'acc':1-acc, 'fpr':model_fpr,'tpr':model_tpr,'f1':f1, 'conf':conf }
        
     
        
        self.load_model('knn',('./' + model_dir +   '/knn.p') )
        ypred = self.model.model.predict(xtest)
        yclass = np.array(ypred >.5).astype(int)
        acc = np.sum(np.abs((yclass.flatten() - ytest.flatten())))/len(yclass)
        model_fpr, model_tpr = self.__get_roc(ypred,ytest)
        f1 = f1_score(ytest,yclass)
        conf = confusion_matrix(ytest,yclass)
        scores['knn'] = {'acc':1-acc, 'fpr':model_fpr,'tpr':model_tpr,'f1':f1 ,'conf':conf}      
        
        
        
        self.model = None
        self.load_model('rf',('./' + model_dir +   '/rf.sav') )
        ypred = self.model.model.predict(xtest)
        yclass = np.array(ypred >.5).astype(int)
        acc = np.sum(np.abs((yclass.flatten() - ytest.flatten())))/len(yclass)
        model_fpr, model_tpr = self.__get_roc(ypred,ytest)
        f1 = f1_score(ytest,yclass)
        conf = confusion_matrix(ytest,yclass)
        scores['rf'] = {'acc':1-acc, 'fpr':model_fpr,'tpr':model_tpr,'f1':f1,'conf':conf }    

        self.load_model('cnn',('./' + model_dir +   '/cnn.h5') )
        ypred = self.model.model.predict(self.resize_for_cnn( xtest))
        yclass = np.array(ypred >.5).astype(int)[:,0]
        
       
        acc = np.sum(np.abs((yclass.flatten() - ytest.flatten())))/len(yclass)
        model_fpr, model_tpr = self.__get_roc(ypred[:,0],ytest)
        f1 = f1_score(ytest,yclass)
        conf = confusion_matrix(ytest,yclass)
        scores['cnn'] = {'acc':1-acc, 'fpr':model_fpr,'tpr':model_tpr,'f1':f1,'conf':conf }
        
     
        
        modelargs = self.default_model_args['lstm']
        # self.generate_lstm_training_data( maxlen = modelargs['input_length'], kmer = modelargs['kmer'] )

        x,xtest, y,ytest = self.format_training_data(self.training_array_lstm,self.labels,**dataargs) 
        
        
        
        self.load_model('lstm',('./' + model_dir +   '/lstm.h5') )
        ypred = self.model.model.predict(xtest)
        yclass = np.array(ypred >.5).astype(int)
        acc = np.sum(np.abs((yclass.flatten() - ytest.flatten())))/len(yclass)
        model_fpr, model_tpr = self.__get_roc(ypred,ytest)
        f1 = f1_score(ytest,yclass)
        conf = confusion_matrix(ytest,yclass)
        scores['lstm'] = {'acc':1-acc, 'fpr':model_fpr,'tpr':model_tpr,'f1':f1,'conf':conf }    
       
        return scores

    def __get_roc(self,probs, test_labels):
        """Compare machine learning model to baseline performance.
        Computes statistics and shows ROC curve."""
        if type(probs) != list:
            probs = list(probs)
    
        if type(test_labels) != list:
            test_labels = list(test_labels)
        
        # Calculate false positive rates and true positive rates
        base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
        model_fpr, model_tpr, _ = roc_curve(test_labels, probs)
        return model_fpr, model_tpr


            

class kmer_DataProcessor():
    
    def __init__(self):
        self.unique_hits = set()
        self.data_file = ''
        self.all_fastas = []
        self.letter_dict = ['a','c','u','g']
        #preallocate small kmers
        self.kmer_4 = self.kmer_list(4)
        self.kmer_2 = self.kmer_list(2)
        self.kmer_3 = self.kmer_list(3)
        self.kmer_5 = self.kmer_list(5)
        self.kmer_1 = self.kmer_list(1)
        
        self.ns_dict = {'m':'a','w':'a','r':'g','y':'t','k':'g','s':'g','w':'a','h':'a','n':'a','x':'a'}
        
        self.test_seq = 'aucuguacguacguaucgaucguguacuggcaaaacguaguagcugagcaucaucuaugh'
        
    
    
    
    
    
    def create_database(self,path, u_thresh = 1, disp = False):
        '''
        Pull all sequences and throw out everything with a given threshold, if 
        u_thresh is 100% match it is fast
        otherwise this is a very slow function that needs alignment each iteration
        '''
        
        self.data_file = path
        self.all_fastas = list(SeqIO.parse(path,'fasta'))
        print('processing sequences')
        n = 0
        m = len(self.all_fastas)
        for f in self.all_fastas:
            n+= 1
            if n% 100 == 0:
                if disp:
                    print('processed: %d out of %d'%(n,m))
            if u_thresh == 1:
                newstr = self.clean_seq(str(f.seq))
                
                if newstr not in self.unique_hits:
                    self.unique_hits.add((f.id + '==='+newstr))
                    
            else:
                newstr = self.clean_seq(str(f.seq))
                best_match = self.check_percentage_alignments(newstr)
                
                if best_match < u_thresh:
                    
                    self.unique_hits.add((f.id + '==='+newstr))
        
        self.unique_hits = list(self.unique_hits)
        self.unique_ids = []
        for entry in self.unique_hits:
            self.unique_ids.append(entry.split('===')[0])
            
        self.get_unique_seqs()
        
        
    def reset_db(self,  u_thresh = 1,disp=False):
        print('resetting sequences')
        self.unique_hits = set()
        n = 0
        m = len(self.all_fastas)
        for f in self.all_fastas:
            n+= 1
            if n% 100 == 0:
                if disp:
                    print('processed: %d out of %d'%(n,m))
            if u_thresh == 1:
                newstr = self.clean_seq(str(f.seq))
                
                if newstr not in self.unique_hits:
                    self.unique_hits.add((f.id + '==='+newstr))
                    
            else:
                newstr = self.clean_seq(str(f.seq))
                best_match = self.check_percentage_alignments(newstr)
                
                if best_match < u_thresh:
                    
                    self.unique_hits.add((f.id + '==='+newstr))
                    
        self.unique_hits = list(self.unique_hits)
        self.unique_ids = []
        for entry in self.unique_hits:
            self.unique_ids.append(entry.split('===')[0])
            
        self.get_unique_seqs()        
        
                   
    def check_debruin_degeneracy():
        x=1
        
    def clean_seq(self,seq):
        '''
        clean the sequences to lowercase only a, u, g, c
        '''
        seq = seq.lower()
        
        for key in self.ns_dict.keys():            
            seq = seq.replace(key,self.ns_dict[key])
            
        seq = seq.replace('t','u')
        return seq
        
    
    def get_all_kmers(self,k):
        '''
        build the full array of kmers for the current database
        '''
        self.kmer_database = []
        for entry in self.unique_hits:
            kf = self.kmer_freq(entry.split('===')[1],k)
            self.kmer_database.append(kf)
            
        self.kmer_array = np.array(self.kmer_database)


    def get_all_kmers_length_dist(self,k,density,bins):
        '''
        build the full array of kmers for the current database
        '''
        self.kmer_database = []
        
        cum_density = np.cumsum(density)
        
        self.normalized_sizes = []
        self.normalized_seqs = []
        for entry in self.unique_hits:
            
            
            r = np.random.uniform()
            
            while len(np.where(r < cum_density)[0]) ==0:
                r = np.random.uniform()
                
            index = np.where(r < cum_density)[0][0]
                    
            length = int(bins[index])
            seq = entry.split('===')[1]
            
            if len(seq) > length:
                offset = int(np.random.uniform(0,len(seq)- length-1 ))
            else: 
                length = len(seq)
                offset = 0
                
            #print(len(seq[offset:(offset+ length)]))
            kf = self.kmer_freq(seq[offset:(offset+ length)],k)
            self.kmer_database.append(kf)
            self.normalized_sizes.append( len( seq[offset:(offset+ length)] ) )
            self.normalized_seqs.append(seq[offset:(offset+ length)])
            
            
            
        self.kmer_array = np.array(self.kmer_database)
        

    def get_all_kmers_norm(self,k):
        '''
        build the full array of kmers for the current database
        '''
        self.kmer_database_norm = []
        for entry in self.unique_hits:
            kf = self.kmer_freq(entry.split('===')[1],k)
            seq_length = len(entry.split('===')[1])
            self.kmer_database_norm.append(kf/seq_length)
            
        self.kmer_array_norm = np.array(self.kmer_database_norm)
                

    def remove_entries(self,id_list):
        to_remove_indexes = []
        new_all_fastas = []
    
        for i in range(len(self.all_fastas)):
            if self.all_fastas[i].id not in id_list:
                new_all_fastas.append(self.all_fastas[i])
        
        self.all_fastas = new_all_fastas
        
            

    def export_to_csv(self,filename,normalized=False):
        
        if normalized:
            k_arr = self.kmer_array_norm
            k_db = self.kmer_database_norm
        else:
            k_arr = self.kmer_array
            k_db = self.kmer_database
        
        k = k_arr.shape[1]
        n=1
        while k !=4:
            n+=1
            k= k/4
        k=n
        print(k)
        if k > 5:
            kmer_ind = self.kmer_list(k)
        else:
            if k == 1:
                kmer_ind = self.kmer_1
            if k == 2:
                kmer_ind = self.kmer_2
            if k == 3:
                kmer_ind = self.kmer_3            
            if k == 4:
                kmer_ind = self.kmer_4
            if k == 5:
                kmer_ind = self.kmer_5
                
        df = pd.DataFrame(k_db, columns = kmer_ind ) 
        df.insert(0,"ID",self.unique_ids,True)
        
        unique_seqs = []
        for entry in self.unique_hits:
            unique_seqs.append(entry.split('===')[1])
        
        
        df.insert(1,"SEQ",unique_seqs,True)
        df.to_csv(('../'+filename + '.csv'))
        
    def get_unique_seqs(self):
        unique_seqs = []
        for entry in self.unique_hits:
            unique_seqs.append(entry.split('===')[1])
        self.unique_seqs = unique_seqs
        self.get_sizes()
        
    def get_sizes(self):
        sizes = []
        for seq in self.unique_seqs:
            sizes.append(len(seq))
        self.all_sizes = sizes
        
    def check_percentage_alignments(self,seq):
        '''
        use global alignment to get percentage match to database 
        
        EXTREMELY SLOW
        '''
        max_align = 0
        lenseq = len(seq)
        for seq2 in self.unique_hits: 
            aligns = pairwise2.align.globalxx(seq, seq2)
            best_match = 0
            for align in aligns:
                if align[2] > best_match:
                    best_match = align[2]
                    
            percentage_match = best_match/lenseq
            if percentage_match > max_align:
                max_align = percentage_match
                
        return max_align
        

    def check_perecentage_via_freq(self,freq,thresh):
        '''
        check percentage of similarities of given kmer freq to the current database
        
        fast but inaccurate if the sequences have different sizes 
        
        '''
        similarities = []
        for kmer in self.kmer_database:
            similarities.append(np.mean( freq != kmer ))
            
        match_inds = np.where(np.array(similarities) < thresh)[0]
        return similarities, match_inds
        
        
    def kmer_list(self,k):        
        combos =[x for x in it.product(self.letter_dict, repeat=k)]       
        kmer = [''.join(y) for y in combos]
        return kmer
    
    def kmer_freq(self,seq,k):
        '''
        calculate the kmer frequences of k size for seq
        ''' 
        
        if k > 5:
            kmer_ind = self.kmer_list(k)
        else:
            if k == 1:
                kmer_ind = self.kmer_1
            if k == 2:
                kmer_ind = self.kmer_2
            if k == 3:
                kmer_ind = self.kmer_3            
            if k == 4:
                kmer_ind = self.kmer_4
            if k == 5:
                kmer_ind = self.kmer_5
                
        kmer_freq_vec = np.zeros((4**k)).astype(int)
        for i in range(len(seq)-k):
            kmer_freq_vec[kmer_ind.index(seq[i:i+k])] += 1
            
        return kmer_freq_vec
 



class RS_HitsCompare:
    def __init__(self):
        self.hits_dfs = []
        self.names = []
        
    def load_hits_dir(self, model_dir):
        for file in os.listdir(model_dir):
            if file.split('.')[-1] == 'csv':
                print(model_dir + file)
                df = pd.read_csv(model_dir + file)
                self.hits_dfs.append(df)
                self.names.append(file.split('.')[0] )
                
        self.__get_matching_asc()
        self.get_combos()
                
    def __get_matching_asc(self):
        
        for i in range(len(self.hits_dfs) ):
            
            if i == 0:
                self.matching_asc = set(list(self.hits_dfs[i]['Asc'].values))
            else:
                self.matching_asc =self.matching_asc & set(list(self.hits_dfs[i]['Asc'].values))
        
            print(len(self.matching_asc))
            
            
    def get_combos(self):
        combos = []
        combo_ids = [] 
        ids = [x for x in range(len(self.names)) ]
        for i in range(0,len(self.names)+1):
            combos = combos + ['-'.join(list(x)) for x in list(itertools.combinations(self.names, i))]
            combo_ids = combo_ids + [x for x in list(itertools.combinations(ids, i))]
            
        combos.pop(0)
        combo_ids.pop(0)
        self.combos = combos
        self.combo_ids = combo_ids
        
        match_lens = []
        matches = []
        
        for combo in combos:
            
            compare = [self.names.index(x) for x in    combo.split('-')] 
            print(compare)
            k = 0
            for i in compare:
                
                if k == 0:
                    matching_asc = set(list(self.hits_dfs[i]['Asc'].values))
                else:
                    matching_asc =matching_asc & set(list(self.hits_dfs[i]['Asc'].values))  
                k += 1
                    
            matches.append(matching_asc)
            match_lens.append(len(matching_asc))
            
        self.combo_matches = matches
        self.combo_lens = match_lens



    def combine_matches_csv(self,threshold,filename,keydf):
        combo_key = [ x < threshold for x in self.combo_lens]
        combo_matches_prune = []
        for i in range(len(combo_key)):
            if combo_key[i]: 
                combo_matches_prune.append(self.combo_matches[i])
        final_matches = list(set.union(*combo_matches_prune))
        
        colnames = ['Asc','Seq','length bp','pRS_lstm','pRS_cnn','pRS_ffnn','pRS_rf','pRS_knn','subseq_lstm','subseq_cnn' ,'subseq_ffnn' ,'subseq_rf' ,'subseq_knn'   ]
        new_data_frame = pd.DataFrame(columns = ['index','Asc','Seq','length bp','pRS_lstm','pRS_cnn','pRS_ffnn','pRS_rf','pRS_knn','subseq_lstm','subseq_cnn' ,'subseq_ffnn' ,'subseq_rf' ,'subseq_knn'   ])
        k = 0
        
        new_dataset = {}
        for i in range(len(final_matches)):
            
            new_dataset[final_matches[i]] = {}
            new_dataset[final_matches[i]]['Asc'] = final_matches[i]
            
            for j in range(len(self.hits_dfs)):
                if 'lstm' in self.names[j]:
                    if sum(self.hits_dfs[j]['Asc'] == final_matches[i]) != 0:
                        
                        new_dataset[final_matches[i]]['Seq'] = (self.hits_dfs[j]['Seq'][self.hits_dfs[j]['Asc'] == final_matches[i]]).values[0]
                        new_dataset[final_matches[i]]['length bp'] = (self.hits_dfs[j]['length bp'][self.hits_dfs[j]['Asc'] == final_matches[i]]).values[0]
                        new_dataset[final_matches[i]]['subseq_lstm'] = (self.hits_dfs[j]['best subseq'][self.hits_dfs[j]['Asc'] == final_matches[i]]).values[0]
                        new_dataset[final_matches[i]]['pRS_lstm'] = (self.hits_dfs[j]['pRS'][self.hits_dfs[j]['Asc'] == final_matches[i]]).values[0]

                    else:
                        new_dataset[final_matches[i]]['pRS_lstm'] = 0
                        new_dataset[final_matches[i]]['subseq_lstm'] = '-'
                
                if 'cnn' in self.names[j]:
                    if sum(self.hits_dfs[j]['Asc'] == final_matches[i]) != 0:
                        
                        new_dataset[final_matches[i]]['Seq'] = (self.hits_dfs[j]['Seq'][self.hits_dfs[j]['Asc'] == final_matches[i]]).values[0]
                        new_dataset[final_matches[i]]['length bp'] = (self.hits_dfs[j]['length bp'][self.hits_dfs[j]['Asc'] == final_matches[i]]).values[0]
                        new_dataset[final_matches[i]]['subseq_cnn'] = (self.hits_dfs[j]['best subseq'][self.hits_dfs[j]['Asc'] == final_matches[i]]).values[0]
                        new_dataset[final_matches[i]]['pRS_cnn'] = (self.hits_dfs[j]['pRS'][self.hits_dfs[j]['Asc'] == final_matches[i]]).values[0]

                    else:
                        new_dataset[final_matches[i]]['pRS_cnn'] = 0
                        new_dataset[final_matches[i]]['subseq_cnn'] = '-'                
                
                if 'ffnn' in self.names[j]:
                    if sum(self.hits_dfs[j]['Asc'] == final_matches[i]) != 0:
                        
                        new_dataset[final_matches[i]]['Seq'] = (self.hits_dfs[j]['Seq'][self.hits_dfs[j]['Asc'] == final_matches[i]]).values[0]
                        new_dataset[final_matches[i]]['length bp'] = (self.hits_dfs[j]['length bp'][self.hits_dfs[j]['Asc'] == final_matches[i]]).values[0]
                        new_dataset[final_matches[i]]['subseq_ffnn'] = (self.hits_dfs[j]['best subseq'][self.hits_dfs[j]['Asc'] == final_matches[i]]).values[0]
                        new_dataset[final_matches[i]]['pRS_ffnn'] = (self.hits_dfs[j]['pRS'][self.hits_dfs[j]['Asc'] == final_matches[i]]).values[0]

                    else:
                        new_dataset[final_matches[i]]['pRS_ffnn'] = 0
                        new_dataset[final_matches[i]]['subseq_ffnn'] = '-'     
                        
                if 'rf' in self.names[j]:
                    if sum(self.hits_dfs[j]['Asc'] == final_matches[i]) != 0:
                        
                        new_dataset[final_matches[i]]['Seq'] = (self.hits_dfs[j]['Seq'][self.hits_dfs[j]['Asc'] == final_matches[i]]).values[0]
                        new_dataset[final_matches[i]]['length bp'] = (self.hits_dfs[j]['length bp'][self.hits_dfs[j]['Asc'] == final_matches[i]]).values[0]
                        new_dataset[final_matches[i]]['subseq_rf'] = (self.hits_dfs[j]['best subseq'][self.hits_dfs[j]['Asc'] == final_matches[i]]).values[0]
                        new_dataset[final_matches[i]]['pRS_rf'] = (self.hits_dfs[j]['pRS'][self.hits_dfs[j]['Asc'] == final_matches[i]]).values[0]

                    else:
                        new_dataset[final_matches[i]]['pRS_rf'] = 0
                        new_dataset[final_matches[i]]['subseq_rf'] = '-'    
                        
                if 'knn' in self.names[j]:
                    if sum(self.hits_dfs[j]['Asc'] == final_matches[i]) != 0:
                       
                        new_dataset[final_matches[i]]['Seq'] = (self.hits_dfs[j]['Seq'][self.hits_dfs[j]['Asc'] == final_matches[i]]).values[0]
                        new_dataset[final_matches[i]]['length bp'] = (self.hits_dfs[j]['length bp'][self.hits_dfs[j]['Asc'] == final_matches[i]]).values[0]
                        new_dataset[final_matches[i]]['subseq_knn'] =( self.hits_dfs[j]['best subseq'][self.hits_dfs[j]['Asc'] == final_matches[i]]).values[0]
                        new_dataset[final_matches[i]]['pRS_knn'] = (self.hits_dfs[j]['pRS'][self.hits_dfs[j]['Asc'] == final_matches[i]]).values[0]

                    else:
                        new_dataset[final_matches[i]]['pRS_knn'] = 0
                        new_dataset[final_matches[i]]['subseq_knn'] = '-'                            
                        
                        
            newcol = ['-',]  +    [new_dataset[final_matches[i]][x] for x in colnames]  
            
            new_data_frame = new_data_frame.append(new_dataset[final_matches[i]],ignore_index=True )
            
            
        self.thresh_csv = new_data_frame
        return new_dataset

class RS_HitsProcessor():
    
    def __init__(self):
        self.original_dataframe = None
        
    def load_hits(self,csv_file):
        phdf = pd.read_csv(csv_file)
        phdf = phdf.drop(['Unnamed: 0'],axis=1)
        
        self.original_dataframe = phdf
        
    def load_hits_df(self,df):
        phdf = df
        try:
            phdf = phdf.drop(['Unnamed: 0'],axis=1) 
        except:
            pass
        self.original_dataframe = phdf
        
        
        
    def prune_hits(self,probability_limit,length_limit,df = None):
        if df == None:
            phdf = self.original_dataframe
        else:
            phdf = df
       
        lens = np.array([len(l) for l in phdf['Seq']  ])
        
        phdf_strip_len =  phdf[np.array([lens <length_limit]).T]
        
        
        phdf_pruned = phdf_strip_len[np.array([phdf_strip_len['subseq pRS'] >probability_limit]).T]
        
        
        phdf_final = pd.DataFrame(columns=phdf_pruned.columns)
        
        unique_values = []
        last_i = 0
        for i in range(len(phdf_pruned)):
            if phdf_pruned['Seq'].values[i] not in unique_values:
                phdf_final = phdf_final.append(phdf_pruned.iloc[i,:])
                unique_values.append(phdf_pruned['Seq'].values[i] )
                
            else:
                phdf_final.iloc[-1,1] = phdf_final.iloc[-1,1] + ',' +  phdf_pruned.iloc[i,1]
        
        self.pruned_dataframe = phdf_final
        
    
                        
                
def check_total_ntdiff(file):
    ns_dict_count = {'m':0,
                 'w':0,
                 'r':0,
                 'y':0,
                 'k':0,
                 's':0,
                 'w':0,
                 'h':0,
                 'n':0,
                 'x':0}
    
    ns_dict = {'m':'a',
                 'w':'a',
                 'r':'g',
                 'y':'t',
                 'k':'g',
                 's':'g',
                 'w':'a',
                 'h':'a',
                 'n':'a',
                 'x':'a'}
    all_fastas = list(SeqIO.parse(file,'fasta'))
    for f in all_fastas:
        seq = str(f.seq).lower()
        for key in ns_dict.keys():
            ns_dict_count[key] += seq.count(key)
            
    return ns_dict_count
        
               
               
        
if __name__ == "__main__":
    
    a = 1
    
    
#    utr5_db = RS_DataProcessor()
#    utr5_db.create_database('riboswitches.fasta')
#    utr5_db.get_all_kmers(5)
#    utr5_db.export_to_csv('RS_5mers.csv')
#
#
#    utr5_db = RS_DataProcessor()
#    utr5_db.create_database('lncRNA_32to400nt.fasta')
#    utr5_db.get_all_kmers(5)
#    utr5_db.export_to_csv('ncRNA_5mers.csv')
#

# rs_db = RS_DataProcessor()
# rs_db.create_database('./fasta_files/riboswitch_NOT_bacteria.fasta')



# #     nc_db = RS_DataProcessor()
# #     nc_db.create_database('riboswitches.fasta')
# #     nc_db.get_all_kmers_norm(3)
    
# fastas_to_remove = []
# for fasta in rs_db.all_fastas:
#     keyword_list = ['unclassified','phage','PDB',\
#                     'Euryarchaeota','Acidianus','Aciduliprofundum',\
#                     'Acyrthosiphon','Aedes','Amazona','Anopheles','Archaeoglobus'\
#                     'Beta','Candidatus','Crenarchaeota','Hadesarchaea','Haladaptatus'\
#                         ,'Halarchaeum','Halococcus','Halogeometricum','Haloprofundus','Halorubrum'\
#                         , 'Metallosphaera','Methanobacterium','Methanobrevibacter','Methanocella','Methanocorpusculum'\
#                         ,'Methanomassiliicoccales','Methanomicrobiales','Methanoregula','Methanosarcina'\
#                         ,'Methanosphaera','Methanosphaerula','Methanospirillum','Methanothermobacter'\
#                         , 'Natrialba','Natrinema','Natronolimnobius','Palaeococcus','Picrophilus',\
#                         'Pyrococcus','gut','Sulfolobus','Thaumarchaeota','Thermococcales',\
#                         'Thermococcus','Thermoplasma','Thermoplasmatales','archaeon', ]
#     if  any(bacstring.lower() in fasta.description.lower() for bacstring in keyword_list):
#         fastas_to_remove.append(fasta.id)
# len(fastas_to_remove)
# print('removing entries with keywords:')
# print(keyword_list)
# print(len(rs_db.all_fastas))
# rs_db.remove_entries(fastas_to_remove)
# rs_db.reset_db()
# print(len(rs_db.all_fastas))
    
# rs_db.get_all_kmers(3)


# [ rs_db.all_fastas[i].description.split(' ')[1] for i in range(len(rs_db.all_fastas))]

# google_these = ['Aedes','Anopheles','Acyrthosiphon', 'Brugia','Caenorhabditis','Chelonia','Ixodes',\
#                 'Dendroctonus','Drosophila','Fukomys','Glossina','Glycine','Gorilla','Heterocephalus',\
#                 'Lasius','Loxodonta','Lucilia','Megaselia','Monodelphis','Myotis','Nasonia','Neotoma',\
#                  'Nipponia','Octopus','Patagioenas','Pelodiscus','Phaethon','Phalacrocorax','Rattus',\
#                      'Rhodnius','Serpula','Tauraco']

    
# rs_db2 = RS_DataProcessor()
# rs_db2.create_database('./fasta_files/riboswitch_AND_tax_stringeukaryota.fasta')
# aa = [ rs_db2.all_fastas[i].description.split(' ')[1] for i in range(len(rs_db2.all_fastas))]
    
# rs_db2.get_all_kmers(3)
# rs_db2.export_to_csv('rs_3mers_eukaryotic.csv')
# # #    
   

#     utr5_db = RS_DataProcessor()
#     utr5_db.create_database('C:/Users/willi/Documents/GitHub/RibosEuk/Dataset/5UTRaspic.Hum.fasta')
#     utr5_db.get_all_kmers(5)
#     utr5_db.export_to_csv('5primeUTR_human_5mers_fulllen.csv')    
    
#     x = plt.hist(rs_db.all_sizes,400,density=True)
#     density = x[0]
#     bins = x[1]

# #   
# #    
# #    ncrna_db = RS_DataProcessor()
# #    ncrna_db.create_database('lncRNA_32to400nt.fasta')
# #    ncrna_db.get_all_kmers_length_dist(3,density,bins)
# #    ncrna_db.get_sizes()
# #    ncrna_db.export_to_csv('ncrna_db_length_norm')
# ##    
# #    
#     utr5_db = RS_DataProcessor()
#     utr5_db.create_database('lncRNA_32to400nt.fasta')
#     utr5_db.get_all_kmers_length_dist(4,density,bins)
#     utr5_db.export_to_csv('ncRNA_4mers_lengthnorm.csv')


#    x = plt.hist(rs_db.all_sizes,400,density=True)
#    density = x[0]
#    
#    cum_density = np.cumsum(density)
#    
#    r = np.random.uniform(size= 10000)
#    transfomed_vec = np.zeros(10000)
#    i = 0
#    for randn in r:
#        while len(np.where(r[i] < cum_density)[0]) ==0:
#            r[i] = np.random.uniform()
#        transfomed_vec[i] = np.where(r[i] < cum_density)[0][0]
#
#        i+=1
#        
#    rr = x[1][transfomed_vec.astype(int).tolist()]
#    
#    plt.hist(rs_db.all_sizes,400,density=True)
#    plt.hist(ncrna_db.all_sizes,400,density=True)
#    plt.legend('RSdb','NCdb')    
#    plt.xlabel('Lengths')
#    plt.ylabel('Probability')

#
#    ncrna_db = RS_DataProcessor()
#    ncrna_db.create_database('lncRNA_32to400nt.fasta')
#    ncrna_db.get_all_kmers_length_dist(3,)
#    
#    
    
    

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:48:16 2019

@author: Zhou
"""

import torch
import pickle
from torch import nn

class Encoder(object):
    def __init__(self):
        pass
    
    def encode(self, series):
        """
        Take a series of inputs and output encoded series as a tensor
        """
        raise NotImplementedError
    
    def decode(self, res_tensor):
        """
        Take a series of results as tensor and output decoded series
        """
        raise NotImplementedError
    
    
class XEncoder(Encoder):
    def __init__(self, embedding_dim, embedding_filename):
        with open(embedding_filename,'rb') as infile:
            vocab, vectors = pickle.load(infile, encoding='latin1')
        self.word_embeddings = nn.Embedding(len(vocab), embedding_dim)
        self.word_id = {v:idx for idx, v in enumerate(vocab)}
        self.word_embeddings.weight.data.copy_(torch.from_numpy(vectors))
    
    def encode(self, series):
        idxs = [self.word_id[w] if w in self.word_id.keys() \
                else self.word_id["<UNK>"] for w in series]
        sentence = torch.tensor(idxs, dtype=torch.long)
        return self.word_embeddings(sentence)
        
    

class YEncoder(Encoder):
    def __init__(self):
        pass
    

class OneHotKeyEncoder(XEncoder):
    def __init__(self):
        pass
    

class PolyglotEncoder(XEncoder):
    def __init__(self):
        pass
    



# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:48:16 2019

@author: Zhou
"""

import torch
import pickle
from torch import nn
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

import config


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
    
    def get_dim(self):
        """
        Return the encoding dimension of encoder
        """
        raise NotImplementedError
    
    
class XEncoder(Encoder):
    def __init__(self, embedding_filename):
        with open(embedding_filename,'rb') as infile:
            vocab, vectors = pickle.load(infile, encoding='latin1')
        self.embedding_dim = vectors.shape[1]
        self.word_embeddings = nn.Embedding(len(vocab), vectors.shape[1])
        self.word_id = {v:idx for idx, v in enumerate(vocab)}
        self.word_embeddings.weight.data.copy_(torch.from_numpy(vectors))
    
    def encode(self, series):
        idxs = [self.word_id[w] if w in self.word_id.keys() \
                else self.word_id["<UNK>"] for w in series]
        sentence = torch.tensor(idxs, dtype=torch.long)
        return self.word_embeddings(sentence)
    
    def get_dim(self):
        return self.embedding_dim
    
    
class BertEncoder(Encoder):
    def __init__(self):
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    def encode(self, series):
        if not isinstance(series, list):
            raise TypeError
        text = series[::]
        for i, x in enumerate(text):
            if x == "<S>":         #<S>,</S> for polyglot
                text[i] = " [CLS] "    #the space is to avoid messy code
            elif x == "</S>":
                text[i] = " [SEP] "
        tokenized_text = self.tokenizer.tokenize(''.join(text))
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0 for t in tokenized_text]
        
        print(tokenized_text)

        # convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor, segments_tensors)
        res = encoded_layers[-1].reshape(-1, config.BERT_DIM)
        return res       # Use last layer as embedding
    # TODO:try other ways, e.g. avg of last 4 layers, etc.
    
    def get_dim(self):
        return config.BERT_DIM
        

class YEncoder(Encoder):
    def __init__(self, tag_list):
        self.tag_dict = dict()      # int -> tag
        self.tag_index_dict = dict()       # tag -> int
        for x,y in enumerate(tag_list):
            self.tag_dict[y] = x
            self.tag_index_dict[x] = y
    
    def get_tag_dim(self):
        return len(self.tag_dict)
    
    def encode(self, series):
        return torch.tensor([self.tag_dict[w] for w in series], dtype=torch.long)
    
    def decode(self, res_tensor):
        return [self.tag_index_dict[t.item()] for t in res_tensor]
    
    def get_dim(self):
        return self.get_tag_dim()       # TODO: unify
    

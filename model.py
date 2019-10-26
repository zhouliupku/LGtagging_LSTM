# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 17:45:36 2019

@author: Zhou
"""

import torch
from torch import nn

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, tag_dict,
                 bidirectional=False,
                 ):
        super(LSTMTagger, self).__init__()
        self.tag_dict = tag_dict
        self.tagix_to_tag = {v:k for k,v in tag_dict.items()}
       
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional)
        tagset_size = len(self.tag_dict)
        if bidirectional:
            self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)
        else:
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        lstm_out, _ = self.lstm(sentence.view(sentence.shape[0], 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(sentence.shape[0], -1))
        tag_scores = nn.functional.log_softmax(tag_space, dim=1)
        return tag_scores
    
    def prepare_targets(self, tags):
        return torch.tensor([self.tag_dict[w] for w in tags], dtype=torch.long)
    
    def convert_results(self, tagix):
        return [self.tagix_to_tag[t.item()] for t in tagix]


# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 17:45:36 2019

@author: Zhou
"""

from torch import nn

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, tag_dim,
                 bidirectional=False):
        super(LSTMTagger, self).__init__()
        #TODO: change to encode/decode of yencoder
#        self.tag_dict = tag_dict
#        self.tagix_to_tag = {v:k for k,v in tag_dict.items()}
       
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional)
        if bidirectional:
            self.hidden2tag = nn.Linear(hidden_dim*2, tag_dim)
        else:
            self.hidden2tag = nn.Linear(hidden_dim, tag_dim)

    def forward(self, sentence):
        lstm_out, _ = self.lstm(sentence.view(sentence.shape[0], 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(sentence.shape[0], -1))
        tag_scores = nn.functional.log_softmax(tag_space, dim=1)
        return tag_scores

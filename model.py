# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 17:45:36 2019

@author: Zhou
"""

import torch
from torch import nn
import pickle

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, tagset_size,
                 embedding_filename="polyglot-zh_char.pkl"):
        super(LSTMTagger, self).__init__()
       
        with open(embedding_filename,'rb') as infile:
            vocab, vectors = pickle.load(infile, encoding='latin1')
        self.word_embeddings = nn.Embedding(len(vocab), embedding_dim)
        self.word_id = {v:idx for idx, v in enumerate(vocab)}
        self.word_embeddings.weight.data.copy_(torch.from_numpy(vectors))
       
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = nn.functional.log_softmax(tag_space, dim=1)
        return tag_scores
   
    def prepare_sequence(self, seq):
        idxs = [self.word_id[w] if w in self.word_id.keys() else self.word_id["<UNK>"] for w in seq]
        return torch.tensor(idxs, dtype=torch.long)

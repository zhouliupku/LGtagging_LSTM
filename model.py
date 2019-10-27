# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 17:45:36 2019

@author: Zhou
"""

import torch
from torch import nn

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, tag_dim,
                 bidirectional=False):
        super(LSTMTagger, self).__init__()
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

    def train(self, training_data, optimizer, loss_type, n_epoch, n_check):
        if loss_type == "NLL":
            loss_function = nn.NLLLoss()
        else:
            raise ValueError("Unsupported loss type: {}".format(loss_type))
        for epoch in range(n_epoch):
            for sentence, targets in training_data:
                self.zero_grad()   # clear accumulated gradient before each instance
                tag_scores = self.forward(sentence)
                loss = loss_function(tag_scores, targets)
                loss.backward(retain_graph=True)
                optimizer.step()
            if epoch % n_check == 0:
                print("Epoch {}".format(epoch))
                print("Loss = {}".format(loss.item()))

    def evaluate(self, test_data, y_encoder):
        """
        Take model and test data (list of strings), return list of list of tags
        """
        result_list = []
        with torch.no_grad():
            for test_sent in test_data:
                if len(test_sent) == 0:
                    continue
                tag_scores = self.forward(test_sent)
                res = y_encoder.decode(tag_scores.max(dim=1).indices)
                result_list.append(res)
            return result_list
        

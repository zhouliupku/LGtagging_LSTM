# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 17:45:36 2019

@author: Zhou
"""

import os
import pickle
import torch
from torch import nn
from torchcrf import CRF
import matplotlib.pyplot as plt
import datetime
import math

PLOT_PATH = os.path.join(os.getcwd(), "plot")


class Tagger(nn.Module):

    def __init__(self, logger, args, x_encoder, y_encoder):
        super(Tagger, self).__init__()
        self.logger = logger
        self.hidden_dim = args.hidden_dim
        self.x_encoder = x_encoder
        self.y_encoder = y_encoder
        self.bidirectional = args.bidirectional
        self.model_setup()
        
    def calc_loss(self, tag_scores, targets, loss_func):
        return loss_func(tag_scores, targets)
    
    def save_model(self, save_path, epoch):
        torch.save(self.state_dict(), os.path.join(save_path, "epoch{}.pt".format(epoch)))
        pickle.dump(self.x_encoder, open(os.path.join(save_path, "x_encoder.p"), "wb"))
        pickle.dump(self.y_encoder, open(os.path.join(save_path, "y_encoder.p"), "wb"))

    def train_model(self, training_data, cv_data,
                    optimizer, args, save_path, need_plot=False):
        if args.loss_type == "NLL":
            loss_function = nn.NLLLoss()
        else:
            raise ValueError("Unsupported loss type")
            
        losses_train = []
        losses_cv = []
        if need_plot:
            plt.figure(figsize=[20, 10])
        
        for epoch in range(args.n_epoch):
            # Use training data to train
            sum_loss_train = 0
            for sentence, targets in training_data:
                self.zero_grad()   # clear accumulated gradient before each instance
                tag_scores = self.forward(sentence)
                loss = self.calc_loss(tag_scores, targets, loss_function)
                sum_loss_train += loss.item()
                loss.backward(retain_graph=True)
                optimizer.step()
            losses_train.append(sum_loss_train / len(training_data))
            self.logger.info("Epoch {}".format(epoch))
            self.logger.info("Training Loss = {}".format(loss.item()))

            # Use CV data to validate
            with torch.no_grad():
                sum_loss_cv = 0
                for sentence, targets in cv_data:
                    tag_scores = self.forward(sentence)
                    loss = loss_function(tag_scores, targets)
                    sum_loss_cv += loss.item()
                losses_cv.append(sum_loss_cv / len(cv_data))
            self.logger.info("CV Loss = {}".format(loss.item()))
                
            # Save model snapshot
            self.save_model(save_path, epoch)
         
        # Plot the loss function
        if need_plot:
            plt.plot(list(map(math.log10, losses_train))) 
            plt.plot(list(map(math.log10, losses_cv)))
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend(["Train", "CV"])
            curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = "plot"+curr_time+".png"
            plt.savefig(os.path.join(PLOT_PATH, filename))
            plt.close()
        
        # Save final model
        self.save_model(save_path, "_final")

    def evaluate_model(self, test_data, y_encoder):
        """
        Take model and test data (list of tensors), return list of list of tags
        """
        result_list = []
        with torch.no_grad():
            for test_sent in test_data:
                if len(test_sent) == 0:
                    continue
                tag_scores = self.forward(test_sent)
                tag_seq = self.transform(tag_scores)
                res = y_encoder.decode(tag_seq)
                result_list.append(res)
            return result_list
        
    def transform(self, tag_scores):
        return tag_scores.max(dim=1).indices
    
    
class LSTMTagger(Tagger):
        
    def model_setup(self):
        self.lstm = nn.LSTM(self.x_encoder.get_dim(), self.hidden_dim,
                            bidirectional=self.bidirectional)
        if self.bidirectional:
            self.hidden2tag = nn.Linear(self.hidden_dim*2, self.y_encoder.get_dim())
        else:
            self.hidden2tag = nn.Linear(self.hidden_dim, self.y_encoder.get_dim())

    def forward(self, sentence):
        lstm_out, _ = self.lstm(sentence.view(sentence.shape[0], 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(sentence.shape[0], -1))
        tag_scores = nn.functional.log_softmax(tag_space, dim=1)
        return tag_scores
       
        
class TwoLayerLSTMTagger(Tagger):
        
    def model_setup(self):
        self.lstm1 = nn.LSTM(self.x_encoder.get_dim(), self.hidden_dim,
                             bidirectional=self.bidirectional)
        self.lstm2 = nn.LSTM(self.hidden_dim, self.hidden_dim,
                             bidirectional=self.bidirectional)
        if self.bidirectional:
            self.lstm2 = nn.LSTM(self.hidden_dim*2, self.hidden_dim,
                                 bidirectional=self.bidirectional)
            self.hidden2tag = nn.Linear(self.hidden_dim*2, self.y_encoder.get_dim())
        else:
            self.lstm2 = nn.LSTM(self.hidden_dim, self.hidden_dim,
                                 bidirectional=self.bidirectional)
            self.hidden2tag = nn.Linear(self.hidden_dim, self.y_encoder.get_dim())

    def forward(self, sentence):
        lstm_out1, _ = self.lstm1(sentence.view(sentence.shape[0], 1, -1))
        lstm_out2, _ = self.lstm2(lstm_out1.view(sentence.shape[0], 1, -1))
        tag_space = self.hidden2tag(lstm_out2.view(sentence.shape[0], -1))
        tag_scores = nn.functional.log_softmax(tag_space, dim=1)
        return tag_scores
       
        
class LSTMCRFTagger(Tagger):
        
    def model_setup(self):
        self.lstm = nn.LSTM(self.x_encoder.get_dim(), self.hidden_dim,
                            bidirectional=self.bidirectional)
        if self.bidirectional:
            self.hidden2tag = nn.Linear(self.hidden_dim*2, self.y_encoder.get_dim())
        else:
            self.hidden2tag = nn.Linear(self.hidden_dim, self.y_encoder.get_dim())
            
        self.crf = CRF(self.y_encoder.get_dim())

    def forward(self, sentence):
        lstm_out, _ = self.lstm(sentence.view(sentence.shape[0], 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(sentence.shape[0], -1))
        tag_scores = nn.functional.log_softmax(tag_space, dim=1)
        return tag_scores
    
    def calc_loss(self, tag_scores, targets, loss_func):
        tag_scores = nn.functional.log_softmax(tag_scores, dim=1)
        return -self.crf.forward(tag_scores.view(tag_scores.shape[0], 1, -1),
                                targets.view(targets.shape[0], -1))
        
    def transform(self, tag_scores):
        return torch.tensor(self.crf.decode(tag_scores.view(tag_scores.shape[0],1,-1))[0],
                            dtype=torch.long)
        
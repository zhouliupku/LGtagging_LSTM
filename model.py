# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 17:45:36 2019

@author: Zhou
"""

import os
import pickle
import torch
from torch import nn, optim
from torchcrf import CRF
from torch.autograd import Variable
import matplotlib.pyplot as plt
import datetime
import math

import config


class Tagger(nn.Module):

    def __init__(self, logger, args, x_encoder, y_encoder):
        super(Tagger, self).__init__()
        self.logger = logger
        self.hidden_dim = args.hidden_dim
        self.x_encoder = x_encoder
        self.y_encoder = y_encoder
        self.bidirectional = args.bidirectional
        self.model_setup()
        self.save_path = None
        
    def calc_loss(self, tag_scores, targets, loss_func):
        return loss_func(tag_scores, targets)
    
    def get_optimizer(self, args):
        if args.optimizer == "SGD":
            return optim.SGD(self.parameters(),
                             lr=args.learning_rate)
        elif args.optimizer == "Adam":
            return optim.Adam(self.parameters(),
                              lr=args.learning_rate,
                              betas=(0.9, 0.999))
        else:
            raise ValueError
    
    def save_model(self, save_path, epoch):
        torch.save(self.state_dict(), os.path.join(save_path, "epoch{}.pt".format(epoch)))
        pickle.dump(self.x_encoder, open(os.path.join(save_path, "x_encoder.p"), "wb"))
        pickle.dump(self.y_encoder, open(os.path.join(save_path, "y_encoder.p"), "wb"))

    def train_model(self, training_data, cv_data, args, need_plot=False):
        optimizer = self.get_optimizer(args)
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
                if args.use_cuda and torch.cuda.is_available():
                    sentence = Variable(sentence).cuda()
                    targets = Variable(targets).cuda()
                tag_scores = self.forward(sentence)
                loss = self.calc_loss(tag_scores, targets, loss_function)
                sum_loss_train += loss.item()
                loss.backward(retain_graph=True)
                optimizer.step()
            losses_train.append(sum_loss_train / len(training_data))
            self.logger.info("Epoch {}".format(epoch))
            self.logger.info("Training Loss = {}".format(loss.item()))
            print("Epoch {}".format(epoch))
            print("Training Loss = {}".format(loss.item()))

            # Use CV data to validate
            with torch.no_grad():
                sum_loss_cv = 0
                for sentence, targets in cv_data:
                    if args.use_cuda and torch.cuda.is_available():
                        sentence = Variable(sentence).cuda()
                        targets = Variable(targets).cuda()
                    tag_scores = self.forward(sentence)
                    loss = loss_function(tag_scores, targets)
                    sum_loss_cv += loss.item()
                losses_cv.append(sum_loss_cv / len(cv_data))
            self.logger.info("CV Loss = {}".format(loss.item()))
            print("CV Loss = {}".format(loss.item()))
                
            # Save model snapshot
            self.save_model(self.save_path, epoch)
         
        # Plot the loss function
        if need_plot:
            plt.plot(list(map(math.log10, losses_train))) 
            plt.plot(list(map(math.log10, losses_cv)))
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend(["Train", "CV"])
            curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = "plot"+curr_time+".png"
            plt.savefig(os.path.join(config.PLOT_PATH, filename))
            plt.close()
        
        # Save final model
        self.save_model(self.save_path, "_final")

    def evaluate_model(self, test_data, y_encoder, args):
        """
        Take model and test data (list of tensors), return list of list of tags
        """
        result_list = []
        with torch.no_grad():
            for test_sent in test_data:
                if len(test_sent) == 0:
                    continue
                if args.use_cuda and torch.cuda.is_available():
                    test_sent = Variable(test_sent).cuda()
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
        
        
class ModelFactory(object):
    def __init__(self):
        self.model_root_path = os.path.join(os.getcwd(), "models")
        
    def get_new_model(self, logger, args, x_encoder, y_encoder):
        if args.model_type == "LSTM":
            model = LSTMTagger(logger, args, x_encoder, y_encoder)
        if args.model_type == "TwoLayerLSTM":
            model = TwoLayerLSTMTagger(logger, args, x_encoder, y_encoder)
        if args.model_type == "LSTMCRF":
            model = LSTMCRFTagger(logger, args, x_encoder, y_encoder)
        self.setup_saving(model, args)
        if args.use_cuda and torch.cuda.is_available():
            model.cuda()
        return model
        
    def get_trained_model(self, logger, args):
        model_path = self.format_path(args)
        if not os.path.isdir(model_path):
            raise FileNotFoundError("No such model!")
        x_encoder = pickle.load(open(os.path.join(model_path, "x_encoder.p"), "rb"))
        y_encoder = pickle.load(open(os.path.join(model_path, "y_encoder.p"), "rb"))
        
        if args.model_type == "LSTM":
            model = LSTMTagger(logger, args, x_encoder, y_encoder)
        if args.model_type == "TwoLayerLSTM":
            model = TwoLayerLSTMTagger(logger, args, x_encoder, y_encoder)
        if args.model_type == "LSTMCRF":
            model = LSTMCRFTagger(logger, args, x_encoder, y_encoder)
            
        if not (args.use_cuda and torch.cuda.is_available()):
            model.load_state_dict(torch.load(os.path.join(model_path, "epoch_final.pt"),
                                             map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(os.path.join(model_path, "epoch_final.pt")))
        model.eval()
        self.setup_saving(model, args)
        return model
    
    def setup_saving(self, model, args):
        save_path = self.format_path(args)
        model.save_path = save_path
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
            
    def format_path(self, args):
        return os.path.join(self.model_root_path,
                          "{}_model".format(args.task_type),
                          args.data_size,
                          args.model_type,
                          args.model_alias)

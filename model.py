# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 17:45:36 2019

@author: Zhou
"""

import os
import pickle
import torch
import numpy as np
from torch import nn, optim
from torchcrf import CRF
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn_utils
import matplotlib.pyplot as plt
import datetime
import math
from tqdm import tqdm

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
        self.optimizer = self.get_optimizer(args)
        
    def calc_loss(self, outputs, labels):
        """
        Calculate loss based on outputs and given targets. Base implementation
        is NLL loss
        """
        labels = labels.view(-1)
        mask = (labels < self.y_encoder.get_num_unmask_tag()).float()
        #the number of tokens is the sum of elements in mask
        num_tokens = int(torch.sum(mask).item())
        #pick the values corresponding to labels and multiply by mask
        outputs = outputs[range(outputs.shape[0]), labels]*mask
        #cross entropy loss for all non 'PAD' tokens
        return -torch.sum(outputs)/num_tokens
    
    
   
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
        torch.save({"epoch": epoch,
                    "model": self.state_dict(),
                    "optimizer": self.optimizer.state_dict()
                    },
                   os.path.join(save_path, "epoch{}.pt".format(epoch)))
        pickle.dump(self.x_encoder, open(os.path.join(save_path, "x_encoder.p"), "wb"))
        pickle.dump(self.y_encoder, open(os.path.join(save_path, "y_encoder.p"), "wb"))

    def make_padded_batch(self, raw_data, batch_size, contain_tag=True):
        '''
        raw_batch_data: list of (list of str, list of tag str)
            if contain_tag is false, then: list of list of str only
        return: list of (Variable(Tensor(x)) of size batch_size x sent_len x embed_dim,
                 Variable(Tensor(y)) of size batch_size x sent_len)
            if contain_tag is false, then: list of Variable(Tensor(x)) only
        '''
        if contain_tag:
            raw_data = sorted(raw_data, key=lambda d: len(d[0]))
        else:
            raw_data = sorted(raw_data, key=len)
        num_batch = len(raw_data) // batch_size        
        batches = []    # list of (Var(x), Var(y))
        for i_batch in range(num_batch):
            batch_end_idx = (i_batch + 1) * batch_size if i_batch < num_batch - 1 else len(raw_data)
            raw_batch_data = raw_data[(i_batch * batch_size) : batch_end_idx]
        
            batch_max_len = max([len(s[0]) if contain_tag else len(s) for s in raw_batch_data])
            padded_xs = []
            padded_ys = []
            if contain_tag:
                for x, y in raw_batch_data:
                    pad_num = batch_max_len - len(x)
                    padded_xs.append(self.x_encoder.encode(x + [config.PAD_CHAR] * pad_num).unsqueeze(0))
                    padded_ys.append(self.y_encoder.encode(y + [config.PAD_TAG] * pad_num).unsqueeze(0))
                batches.append((Variable(torch.cat(padded_xs, dim=0)),
                                Variable(torch.cat(padded_ys, dim=0))))
            else:
                for x in raw_batch_data:
                    pad_num = batch_max_len - len(x)
                    padded_xs.append(self.x_encoder.encode(x + [config.PAD_CHAR] * pad_num).unsqueeze(0))
                batches.append(Variable(torch.cat(padded_xs, dim=0)))
        return batches


    def train_model(self, training_data, cv_data, args, need_plot=False):
        self.train()
            
        losses_train = []
        losses_cv = []
        if need_plot:
            plt.figure(figsize=[20, 10])
        
        # training_data is a list of tuples (x, y), depending on batch_size, sort and build iterator
        batches_train = self.make_padded_batch(training_data, args.batch_size)
        batches_cv= self.make_padded_batch(cv_data, 1)
        
        for epoch in range(args.start_from_epoch + 1,
                           args.start_from_epoch + args.n_epoch + 1):
            losses_epoch = []
            for sentences, targets in batches_train:
                self.zero_grad()   # clear accumulated gradient before each instance
                if args.use_cuda and torch.cuda.is_available():
                    sentences = sentences.cuda()
                    targets = targets.cuda()
                outputs = self.forward(sentences)
                loss = self.calc_loss(outputs, targets)
                loss.backward(retain_graph=True)
                self.optimizer.step()
                losses_epoch.append(loss.item())
            losses_train.append(np.mean(losses_epoch))
        
            self.logger.info("Epoch {}".format(epoch))
            self.logger.info("Training Loss = {}".format(np.mean(losses_epoch)))
            print("Epoch {}".format(epoch))
            print("Training Loss = {}".format(np.mean(losses_epoch)))

            # Use CV data to validate
            losses_epoch_cv = []
            with torch.no_grad():
                for sentence, targets in batches_cv:
                    if args.use_cuda and torch.cuda.is_available():
                        sentence = Variable(sentence).cuda()
                        targets = Variable(targets).cuda()
                    outputs = self.forward(sentence)
                    loss = self.calc_loss(outputs, targets)
                    losses_epoch_cv.append(loss.item())
            losses_cv.append(np.mean(losses_epoch_cv))
            self.logger.info("CV Loss = {}".format(np.mean(losses_epoch_cv)))
            print("CV Loss = {}".format(np.mean(losses_epoch_cv)))
                
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
        
        
    

    def evaluate_model(self, test_data, args):
        """
        Take model and test data (list of (list of str, list of tag_true)),
        return list of (list of tag_pred, list of tag_true)
        """
        result_list = []
        batches = self.make_padded_batch(test_data, 1, contain_tag=True)
        
        with torch.no_grad():
            for test_sent, tags_true_encoded in batches:
                if len(test_sent) == 0:
                    continue
                if args.use_cuda and torch.cuda.is_available():
                    test_sent = test_sent.cuda()
                tag_scores = self.forward(test_sent)
                tag_seq = self.transform(tag_scores)
                tags_pred = self.y_encoder.decode(tag_seq)
                # tags_true_encoded was 1 x sent_len, need to flatten it
                tags_true_encoded = tags_true_encoded.view(-1) 
                tags_true= self.y_encoder.decode(tags_true_encoded)
                result_list.append((tags_pred, tags_true))
            return result_list
        
    def transform(self, tag_scores):
        """
        Convert tag_scores (tensor of sent_len x tag_num) to
        indices of most likely tags (tensor of sent_len)
        """
        return tag_scores.max(dim=1).indices
    
    
class LSTMTagger(Tagger):
        
    def model_setup(self):
        self.lstm = nn.LSTM(self.x_encoder.get_dim(), self.hidden_dim,
                            bidirectional=self.bidirectional,
                            batch_first=True)
        if self.bidirectional:
            self.hidden2tag = nn.Linear(self.hidden_dim*2, self.y_encoder.get_dim())
        else:
            self.hidden2tag = nn.Linear(self.hidden_dim, self.y_encoder.get_dim())

    def forward(self, sentence_batch):
        """
        Input dim: batch_size x batch_max_len x embed_dim
        Output dim: batch_size*batch_max_len x num_tags
        """
        lstm_out, _ = self.lstm(sentence_batch)
        # dim: batch_size x batch_max_len x lstm_hidden_dim
        lstm_out = lstm_out.reshape(-1, lstm_out.shape[2])
        # dim: batch_size*batch_max_len x lstm_hidden_dim
        tag_space = self.hidden2tag(lstm_out)
        # dim: batch_size*batch_max_len x num_tags
        return nn.functional.log_softmax(tag_space, dim=1)
       
        
#class TwoLayerLSTMTagger(Tagger):
#        
#    def model_setup(self):
#        self.lstm1 = nn.LSTM(self.x_encoder.get_dim(), self.hidden_dim,
#                             bidirectional=self.bidirectional)
#        self.lstm2 = nn.LSTM(self.hidden_dim, self.hidden_dim,
#                             bidirectional=self.bidirectional)
#        if self.bidirectional:
#            self.lstm2 = nn.LSTM(self.hidden_dim*2, self.hidden_dim,
#                                 bidirectional=self.bidirectional)
#            self.hidden2tag = nn.Linear(self.hidden_dim*2, self.y_encoder.get_dim())
#        else:
#            self.lstm2 = nn.LSTM(self.hidden_dim, self.hidden_dim,
#                                 bidirectional=self.bidirectional)
#            self.hidden2tag = nn.Linear(self.hidden_dim, self.y_encoder.get_dim())
#
#    def forward(self, sentence):
#        lstm_out1, _ = self.lstm1(sentence.view(sentence.shape[0], 1, -1))
#        lstm_out2, _ = self.lstm2(lstm_out1.view(sentence.shape[0], 1, -1))
#        tag_space = self.hidden2tag(lstm_out2.view(sentence.shape[0], -1))
#        tag_scores = nn.functional.log_softmax(tag_space, dim=1)
#        return tag_scores
       
        
class LSTMCRFTagger(LSTMTagger):
        
    def model_setup(self):
        super().model_setup()            
        self.crf = CRF(self.y_encoder.get_dim(),
                       batch_first=True)
        self.crf.reset_parameters()
        

    def forward(self, sentence_batch):
        """
        Input dim: batch_size x batch_max_len x embed_dim
        Output dim: batch_size*batch_max_len x num_tags
        """
        batch_size = sentence_batch.shape[0]
        lstm_out, _ = self.lstm(sentence_batch)
        # dim: batch_size x batch_max_len x lstm_hidden_dim
        lstm_out = lstm_out.reshape(-1, lstm_out.shape[2])
        # dim: batch_size*batch_max_len x lstm_hidden_dim
        tag_space = self.hidden2tag(lstm_out)
        # dim: batch_size*batch_max_len x num_tags
        tag_score = nn.functional.log_softmax(tag_space, dim=1)
        # dim: batch_size x batch_max_len x num_tags
        return tag_score.view(batch_size, -1, tag_score.shape[-1])
    
    def calc_loss(self, outputs, labels):
        """
        outputs: batch_size*batch_max_len x target_num
        labels: batch_size x batch_max_len
        """
        # mask: byte tensor with dim = batch_size x batch_max_len
        mask = (labels < self.y_encoder.get_num_unmask_tag())
        return -self.crf.forward(outputs, labels, mask=mask)
        
    def transform(self, tag_scores):
        """
        Convert tag_scores (tensor of 1 x sent_len x tag_num) to
        indices of most likely tags (tensor of sent_len)
        """
        # here we don't need mask because batch size is 1 so no PAD involved
        # dim: 1 x sent_len
        best_seq = torch.tensor(self.crf.decode(tag_scores),
                                dtype=torch.long)
        best_seq = best_seq.view(-1)
        return best_seq
        
        
class ModelFactory(object):
    def __init__(self):
        self.model_root_path = os.path.join(os.getcwd(), "models")
        
    def get_new_model(self, logger, args, x_encoder, y_encoder):
        if args.model_type == "LSTM":
            model = LSTMTagger(logger, args, x_encoder, y_encoder)
#        if args.model_type == "TwoLayerLSTM":
#            model = TwoLayerLSTMTagger(logger, args, x_encoder, y_encoder)
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
        
        # Model type setup
        if args.model_type == "LSTM":
            model = LSTMTagger(logger, args, x_encoder, y_encoder)
#        if args.model_type == "TwoLayerLSTM":
#            model = TwoLayerLSTMTagger(logger, args, x_encoder, y_encoder)
        if args.model_type == "LSTMCRF":
            model = LSTMCRFTagger(logger, args, x_encoder, y_encoder)
            
        # model filename setup
        if args.start_from_epoch >= 0:
            model_filename = os.path.join(model_path, "epoch{}.pt".format(args.start_from_epoch))
            if not os.path.exists(model_filename):
                raise IOError("No model found in " + model_filename)
        else:
            model_filename = os.path.join(model_path, "epoch_final.pt")
            
        # load model and specify device
        if not (args.use_cuda and torch.cuda.is_available()):
            model.load_state_dict(torch.load(model_filename,
                                             map_location=torch.device('cpu'))["model"])
        else:
            model.load_state_dict(torch.load(model_filename,
                                             map_location=torch.device('cuda:0'))["model"])
            model.cuda()
            
        model.optimizer.load_state_dict(torch.load(model_filename)["optimizer"])
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

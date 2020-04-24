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

import config
import lg_utils


class Tagger(nn.Module):

    def __init__(self, logger, args, x_encoder, y_encoder):
        super(Tagger, self).__init__()
        self.logger = logger
        self.args = args
        self.x_encoder = x_encoder
        self.y_encoder = y_encoder
        self.model_setup()
        self.save_path = None
        self.optimizer = self.get_optimizer(self.args)
        
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

    def make_padded_batch(self, raw_data, batch_size, contain_tag=True, need_original_str=False):
        '''
        raw_batch_data: list of (list of str, list of tag str)
            if contain_tag is false, then: list of list of str only
        return: list of (Variable(Tensor(x)) of size batch_size x sent_len x embed_dim,
                 Variable(Tensor(y)) of size batch_size x sent_len)
            if contain_tag is false, then: list of Variable(Tensor(x)) only
            if need_original_str is true, then list of (Var, Var, list of list of str)
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
            if need_original_str:
                padded_ss = []
            if contain_tag:
                for x, y in raw_batch_data:
                    pad_num = batch_max_len - len(x)
                    padded_xs.append(self.x_encoder.encode(x + [config.PAD_CHAR] * pad_num).unsqueeze(0))
                    padded_ys.append(self.y_encoder.encode(y + [config.PAD_TAG] * pad_num).unsqueeze(0))
                    if need_original_str:
                        padded_ss.append(x + [config.PAD_CHAR] * pad_num)
                if need_original_str:
                    batches.append((Variable(torch.cat(padded_xs, dim=0)),
                                    Variable(torch.cat(padded_ys, dim=0)),
                                    padded_ss))
                else:
                    batches.append((Variable(torch.cat(padded_xs, dim=0)),
                                    Variable(torch.cat(padded_ys, dim=0))))
            else:
                for x in raw_batch_data:
                    pad_num = batch_max_len - len(x)
                    padded_xs.append(self.x_encoder.encode(x + [config.PAD_CHAR] * pad_num).unsqueeze(0))
                    if need_original_str:
                        padded_ss.append(x + [config.PAD_CHAR] * pad_num)
                if need_original_str:
                    batches.append((Variable(torch.cat(padded_xs, dim=0)),
                                    padded_ss))
                else:
                    batches.append(Variable(torch.cat(padded_xs, dim=0)))
        return batches


    def train_model(self, training_data, cv_data, args):
        self.train()
        
        # training_data is a list of tuples (x, y), depending on batch_size, sort and build iterator
        batches_train = self.make_padded_batch(training_data, args.batch_size)
        single_batches_cv= self.make_padded_batch(cv_data, 1)
        
        for epoch in range(args.start_from_epoch + 1,
                           args.start_from_epoch + args.n_epoch + 1):
            self.logger.info("Epoch {}".format(epoch))
            print("Epoch {}".format(epoch))
            
            # Use train set
            for sentences, targets in batches_train:
                self.zero_grad()   # clear accumulated gradient before each instance
                if args.use_cuda and torch.cuda.is_available():
                    sentences = sentences.cuda()
                    targets = targets.cuda()
                outputs = self.forward(sentences)
                loss = self.calc_loss(outputs, targets)
                loss.backward(retain_graph=True)
                self.optimizer.step()

            # Evaluate on both train and cv
            self.logger.info("Train")
            print("Train")
            self.evaluate_core(batches_train)
            self.logger.info("CV")
            print("CV")
            self.evaluate_core(single_batches_cv)
                
            # Save model snapshot
            self.save_model(self.save_path, epoch)
        
        # Save final model
        self.save_model(self.save_path, "_final")
        
        
    def evaluate_core(self, batches):
        losses = []
        result_list = []
        with torch.no_grad():
            for sentence, targets in batches:
                if len(sentence) == 0:
                    continue
                
                if self.args.use_cuda and torch.cuda.is_available():
                    sentence = Variable(sentence).cuda()
                    targets = Variable(targets).cuda()
                outputs = self.forward(sentence)
                loss = self.calc_loss(outputs, targets)
                losses.append(loss.item())
                
                tag_scores = self.forward(sentence)
                tag_seq = self.transform(tag_scores)
                # tag_seq is batch_size*batch_max_len
                tags_pred = self.y_encoder.decode(tag_seq)
                # targets is batch_size x batch_max_len, need flatten
                targets = targets.view(-1) 
                tags_true = self.y_encoder.decode(targets)
                result_list.append((tags_pred, tags_true))
                
        self.logger.info("Loss = {}".format(np.mean(losses)))
        print("Loss = {}".format(np.mean(losses)))
        self.calc_metric_char(result_list)
        self.calc_correct_ratio_entity(result_list)
        
    
    def calc_metric_char(self, tags):
        # Both tag_pred and tag_true are list of list of tags
        tag_pred = [tag[0] for tag in tags]
        tag_true = [tag[1] for tag in tags]
        assert len(tag_pred) == len(tag_true)
                
        if self.args.task_type == "page":    # only calculate the EOS tag for page model
            tag_to_idx, confusion_matrix = lg_utils.prepare_confusion_matrix(tag_true, tag_pred,
                                                                             [config.INS_TAG, config.EOS_TAG])
            precision, recall, accuracy, f_score = lg_utils.process_confusion_matrix(confusion_matrix,
                                                                                     tag_to_idx[config.EOS_TAG])
            info_log = "P: {}, R: {}, A: {}, F: {}".format(precision, recall, accuracy, f_score)
            print(info_log)
            self.logger.info(info_log)
            
        else:       # ignore BEG, END etc for record model, although they are learned
            tag_list = sorted(list(self.y_encoder.tag_dict.keys()))
            tag_list = [t for t in tag_list if t not in config.special_tag_list]
            tag_to_idx, confusion_matrix = lg_utils.prepare_confusion_matrix(tag_true, tag_pred, tag_list)
            
            # Method 1: micro, without null tag
            precision, recall, accuracy, f_score = \
                lg_utils.process_confusion_matrix_micro(confusion_matrix, tag_to_idx, [config.NULL_TAG])
            info_log = "Micro w/o null: P: {}, R: {}, A: {}, F: {}".format(precision, recall, accuracy, f_score)
            print(info_log)
            self.logger.info(info_log)
            # Method 2: macro, without null tag
            precision, recall, accuracy, f_score = \
                lg_utils.process_confusion_matrix_macro(confusion_matrix, tag_to_idx, [config.NULL_TAG])
            info_log = "Macro w/o null: P: {}, R: {}, A: {}, F: {}".format(precision, recall, accuracy, f_score)
            print(info_log)
            self.logger.info(info_log)
            # Method 3: macro, including null tag
            precision, recall, accuracy, f_score = \
                lg_utils.process_confusion_matrix_macro(confusion_matrix, tag_to_idx)
            info_log = "Macro w/ null: P: {}, R: {}, A: {}, F: {}".format(precision, recall, accuracy, f_score)
            print(info_log)
            self.logger.info(info_log)
            
    
    def calc_correct_ratio_entity(self, tags):
        '''
        Return entity-level correct ratio only for record model
        '''
        tag_pred = [tag[0] for tag in tags]
        tag_true = [tag[1] for tag in tags]
        assert len(tag_pred) == len(tag_true)
        for x, y in zip(tag_pred, tag_true):
            assert len(x) == len(y)
        correct_and_total_counts = [lg_utils.word_count(ps, ts) for ps, ts in zip(tag_pred, tag_true)]
    #    lg_utils.output_entity_details(tag_pred, tag_true, sent_str, mismatch_only=False)
        entity_correct_ratio = sum([x[0] for x in correct_and_total_counts]) \
                                / float(sum([x[1] for x in correct_and_total_counts]))
                
        # Log info of correct ratio
        info_log = "Entity level correct ratio is {}".format(entity_correct_ratio)
        print(info_log)
        self.logger.info(info_log)
        
        tag_list = sorted(list(self.y_encoder.tag_dict.keys()))
        tag_list = [t for t in tag_list if t not in config.special_tag_list]
        precision, recall, accuracy, f_score, collocation_ratio = \
            lg_utils.calc_entity_metrics(tag_pred, tag_true, tag_list)
        info_log = "Entity micro w/ null: P: {}, R: {}, A: {}, F: {}, C:{}"\
            .format(precision, recall, accuracy, f_score, collocation_ratio)
        print(info_log)
        self.logger.info(info_log)
        
        return entity_correct_ratio
    

    def evaluate_model(self, test_data, args):
        """
        Take model and test data (list of (list of str)),
        return list of list of tag
        """
        result_list = []
        with torch.no_grad():
            for sent in test_data:
                if len(sent) == 0:
                    continue
                sent = Variable(self.x_encoder.encode(sent).unsqueeze(0))
                # list of (Variable(Tensor(x)) of size 1 x sent_len x embed_dim,
                if args.use_cuda and torch.cuda.is_available():
                    sent = sent.cuda()
                tag_scores = self.forward(sent)
                tag_seq = self.transform(tag_scores)
                tags_pred = self.y_encoder.decode(tag_seq)
                result_list.append(tags_pred)
            return result_list
        
        
    def transform(self, tag_scores):
        """
        Convert tag_scores (tensor of batch_size*batch_max_len x tag_num) to
        indices of most likely tags (tensor of batch_size*batch_max_len)
        """
        return tag_scores.max(dim=1).indices
    
    
class LSTMTagger(Tagger):
        
    def model_setup(self):
        self.lstm = nn.LSTM(self.x_encoder.get_dim(), self.args.hidden_dim,
                            bidirectional=self.args.bidirectional,
                            num_layers=self.args.lstm_layer,
                            batch_first=True)
        if self.args.bidirectional:
            self.hidden2tag = nn.Linear(self.args.hidden_dim*2, self.y_encoder.get_dim())
        else:
            self.hidden2tag = nn.Linear(self.args.hidden_dim, self.y_encoder.get_dim())

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
       
        
class LSTMCRFTagger(LSTMTagger):
        
    def model_setup(self):
        super().model_setup()            
        self.crf = CRF(self.y_encoder.get_dim(),
                       batch_first=True)
        self.crf.reset_parameters()
        

    def forward(self, sentence_batch):
        """
        Input dim: batch_size x batch_max_len x embed_dim
        Output dim: batch_size x batch_max_len x num_tags
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
        return -self.crf.forward(outputs, labels, mask=mask, reduction='mean')
        
    def transform(self, tag_scores):
        """
        Convert tag_scores (tensor of batch_size x batch_max_len x num_tags) to
        indices of most likely tags (tensor of batch_size*batch_max_len)
        """
        # here we produce tag prediction for padded positions too
        # dim: batch_size x batch_max_len
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

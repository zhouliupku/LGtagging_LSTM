# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:48:16 2019

@author: Zhou
"""

import torch
import pickle
from torch import nn
from pytorch_pretrained_bert import BertTokenizer, BertModel

import config
import lg_utils


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
    def __init__(self, args):
        if args.main_encoder == "BERT":
            self.main_encoder = BertEncoder(args)
        else:
            self.main_encoder = ErnieEncoder(args.main_encoder)
        if args.extra_encoder is None:
            self.extra_encoder = None
        else:
            self.extra_encoder = ErnieEncoder(args.extra_encoder)
        
    def encode(self, series):
        main_embed = self.main_encoder.encode(series)
        if self.extra_encoder is None:
            return main_embed
        extra_embed = self.extra_encoder.encode(series)
        assert main_embed.shape[0] == extra_embed.shape[0]
        final_embed = torch.cat((main_embed, extra_embed), 1)
        return final_embed
    
    def get_dim(self):
        if self.extra_encoder is None:
            return self.main_encoder.get_dim()
        else:
            return self.main_encoder.get_dim() + self.extra_encoder.get_dim()
        
    
class ErnieEncoder(Encoder):
    def __init__(self, embed_type):
        embedding_filename = lg_utils.get_filename_from_embed_type(embed_type)
        with open(embedding_filename, 'rb') as infile:
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
    def __init__(self, args):
        self.args = args
        self.model = BertModel.from_pretrained('bert-base-chinese')
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    def encode(self, series):
        if not isinstance(series, list):
            raise TypeError
        text = series[::]
        for i, x in enumerate(text):
            if x == config.BEG_CHAR:         #<S>,</S> for polyglot
                text[i] = " [CLS] "    #the space is to avoid messy code
            elif x == config.END_CHAR:
                text[i] = " [SEP] "
            elif x == config.PAD_CHAR:
                text[i] = " [PAD] "
            elif ord(x) > 0x9FFF or ord(x) < 0x4e00:
                text[i] = " [UNK] "
        tokenized_text = self.tokenizer.tokenize(''.join(text))
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0 for t in tokenized_text]

        # convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        
        if self.args.use_cuda and torch.cuda.is_available():
            tokens_tensor = tokens_tensor.to('cuda')
            segments_tensors = segments_tensors.to('cuda')
            self.model.to('cuda')

        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor, segments_tensors)
        res = encoded_layers[-1].reshape(-1, config.BERT_DIM)
        return res       # Use last layer as embedding
    # TODO:try other ways, e.g. avg of last 4 layers, etc.
    
    def get_dim(self):
        return config.BERT_DIM
        

class YEncoder(Encoder):
    """
    Each Y encoder must support BEG, END, PAD
    """
    def __init__(self, tag_list):
        self.tag_dict = dict()      # tag -> int
        self.tag_index_dict = dict()       # int -> tag
        self.num_normal_tag = len(tag_list)
        for x, y in enumerate(tag_list):
            self.tag_dict[y] = x
            self.tag_index_dict[x] = y
        # Here we cannot use list because those -1, -2 etc need to be kept as
        # negative to indicate that they are not normal tags and should be
        # skipped when calculate loss and correct ratios etc.
        self.tag_index_dict[self.num_normal_tag] = config.BEG_TAG
        self.tag_index_dict[self.num_normal_tag+1] = config.END_TAG
        self.tag_index_dict[self.num_normal_tag+2] = config.PAD_TAG
        self.tag_dict[config.BEG_TAG] = self.num_normal_tag
        self.tag_dict[config.END_TAG] = self.num_normal_tag+1
        self.tag_dict[config.PAD_TAG] = self.num_normal_tag+2
        self.num_tag = len(self.tag_dict)
    
    def encode(self, series):
        return torch.tensor([self.tag_dict[w] for w in series], dtype=torch.long)
    
    def decode(self, res_tensor):
        return [self.int_to_tag(t.item()) for t in res_tensor]
    
    def get_dim(self):
        return self.num_tag
    
    def get_num_unmask_tag(self):
        return self.num_normal_tag + 2   # normal tag and BEG, END should not be masked
    
    def int_to_tag(self, idx_int):
        return self.tag_index_dict[idx_int]
    

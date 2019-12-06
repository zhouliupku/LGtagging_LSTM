# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 21:31:29 2019

@author: Zhou
"""

import os
import itertools
import torch
from torch import optim

import lg_utils
import config
from Encoders import XEncoder, YEncoder
from model import LSTMTagger, TwoLayerLSTMTagger, LSTMCRFTagger


def get_encoders(args, rt):
    """
    Return X and Y encoder for the model
    This WILL modify args
    """
    char_encoder = XEncoder(config.EMBEDDING_PATH)
    vars(args)['embedding_dim'] = char_encoder.embedding_dim
    if args.task_type == "page":
        tag_encoder = YEncoder([config.INS_TAG, config.EOS_TAG])
    else:
        tagset = set(itertools.chain.from_iterable([r.orig_tags for r in rt]))
        tagset = ["<BEG>", "<END>"] + sorted(list(tagset))
        tag_encoder = YEncoder(tagset)
    return char_encoder, tag_encoder


def train(logger, args):
    """
    Training model
    """
    # Load in data
    raw_train = lg_utils.load_data_from_pickle("{}s_train.p".format(args.task_type),
                                               args.data_size)
    raw_cv = lg_utils.load_data_from_pickle("{}s_cv.p".format(args.task_type),
                                               args.data_size)
    
    char_encoder, tag_encoder = get_encoders(args, raw_train)
    model = LSTMCRFTagger(logger, args, char_encoder, tag_encoder)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    
    # Load models if it was previously saved and want to continue
    model_path = os.path.join(config.REGEX_PATH, "{}_model".format(args.task_type))
    if os.path.exists(model_path) and not args.need_train:
        model.load_state_dict(torch.load(os.path.join(model_path, "final.pt")))
        model.eval()
    
    # Training
    # Step 1. Data preparation
    training_data = lg_utils.get_data_from_samples(raw_train, char_encoder, tag_encoder)
    cv_data = lg_utils.get_data_from_samples(raw_cv, char_encoder, tag_encoder)
    
    # Step 2. Model training
    if args.need_train:
        model.train_model(training_data, cv_data, optimizer, args, model_path)
        
    # Step 3. Evaluation with correct ratio
    lg_utils.correct_ratio_calculation(raw_train, model, "train", char_encoder, tag_encoder)
    lg_utils.correct_ratio_calculation(raw_cv, model, "cv", char_encoder, tag_encoder)
    
    
def test(logger, args):
    """
    Test trained model on set-alone data; this should be done after all tunings
    """
    
    # TODO: put tagset as a part of model
    raw_train = lg_utils.load_data_from_pickle("{}s_train.p".format(args.task_type),
                                               args.data_size)
    raw_test = lg_utils.load_data_from_pickle("{}s_test.p".format(args.task_type),
                                               args.data_size)
    char_encoder, tag_encoder = get_encoders(args, raw_train)
    
    model_path = os.path.join(config.REGEX_PATH, "{}_model".format(args.task_type))
    if not os.path.exists(model_path):
        raise FileNotFoundError("No model found")
    model = LSTMCRFTagger(logger, args, char_encoder, tag_encoder)
    # TODO: how to make load_state_dict function knowing model type?
    model.load_state_dict(torch.load(os.path.join(model_path, "final.pt")))
    model.eval()
    lg_utils.correct_ratio_calculation(raw_test, model, "test", char_encoder, tag_encoder)
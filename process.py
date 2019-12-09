# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 21:31:29 2019

@author: Zhou
"""

import os
import re
import pickle
import datetime
import itertools
import torch
from torch import optim

import lg_utils
import config
from Encoders import XEncoder, YEncoder
from data_save import ExcelSaver, HtmlSaver
from model import LSTMTagger, TwoLayerLSTMTagger, LSTMCRFTagger


def train(logger, args):
    """
    Training model
    """
    # Load in data
    raw_train = lg_utils.load_data_from_pickle("{}s_train.p".format(args.task_type),
                                               args.data_size)
    raw_cv = lg_utils.load_data_from_pickle("{}s_cv.p".format(args.task_type),
                                               args.data_size)
    
    char_encoder = XEncoder(config.EMBEDDING_PATH)
#    vars(args)['embedding_dim'] = char_encoder.embedding_dim
    if args.task_type == "page":
        tag_encoder = YEncoder([config.INS_TAG, config.EOS_TAG])
    else:
        tagset = set(itertools.chain.from_iterable([r.orig_tags for r in raw_train]))
        tagset = ["<BEG>", "<END>"] + sorted(list(tagset))
        tag_encoder = YEncoder(tagset)
    model = LSTMCRFTagger(logger, args, char_encoder, tag_encoder)
    
    # Load models if it was previously saved and want to continue
    model_path = os.path.join(config.REGEX_PATH, "{}_model".format(args.task_type))
    if os.path.exists(model_path) and not args.need_train:
        model.load_state_dict(torch.load(os.path.join(model_path, "final.pt")))
        model.eval()
    # TODO: see if I can put optimizer initialization into model initialization
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    
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
    raw_test = lg_utils.load_data_from_pickle("{}s_test.p".format(args.task_type),
                                               args.data_size)
    
    model_path = os.path.join(config.REGEX_PATH, "{}_model".format(args.task_type))
    x_encoder = pickle.load(open(os.path.join(model_path, "x_encoder.p"), "rb"))
    y_encoder = pickle.load(open(os.path.join(model_path, "y_encoder.p"), "rb"))
    
    if not os.path.exists(model_path):
        raise FileNotFoundError("No model found")
    model = LSTMCRFTagger(logger, args, x_encoder, y_encoder)
    # TODO: how to make load_state_dict function knowing model type?
    model.load_state_dict(torch.load(os.path.join(model_path, "final.pt")))
    model.eval()
    lg_utils.correct_ratio_calculation(raw_test, model, "test", x_encoder, y_encoder)
    
    
def produce(logger, args):
    """
    Produce untagged data using model; this step is unsupervised
    """
    # Step 1. using page_to_sent_model, parse pages to sentences
    pages_produce = lg_utils.load_data_from_pickle("pages_produce.p", args.data_size)
    
    # Step 2. depending on whether user wants to use RegEx/model, process page splitting
    if args.regex:
        with open(os.path.join(config.REGEX_PATH, "surname.txt"), 'r', encoding="utf8") as f:
            surnames = f.readline().replace("\ufeff", '')
        tag_seq_list = []
        for p in pages_produce:
            tags = [config.INS_TAG for c in p.txt]
            for m in re.finditer(r"{}(".format(config.PADDING_CHAR) \
                                 + surnames \
                                 + ')',
                                p.txt):
                tags[m.start(0)] = config.EOS_TAG  # no need to -1, instead drop '○' before name
            tags[-1] = config.EOS_TAG
            tag_seq_list.append(tags)
    else:
        model_path = os.path.join(config.REGEX_PATH, "page_model")
        x_encoder = pickle.load(open(os.path.join(model_path, "x_encoder.p"), "rb"))
        y_encoder = pickle.load(open(os.path.join(model_path, "y_encoder.p"), "rb"))
        page_model = LSTMCRFTagger(logger, args, x_encoder, y_encoder)
        if not os.path.exists(model_path):
            raise ValueError("No model found!")
        page_model.load_state_dict(torch.load(os.path.join(model_path, "final.pt")))
        page_model.eval()
        # Data preparation
        data = lg_utils.get_data_from_samples(pages_produce, x_encoder, y_encoder)
        tag_seq_list = page_model.evaluate_model(data, y_encoder)
            
#   Step 3. using trained record model, tag each sentence
    # TODO: support model type in args
    # 3.1 Prepare data
    record_test_data = []
    records = []
    for p, pl in zip(pages_produce, 
                     lg_utils.get_sent_len_for_pages(tag_seq_list, config.EOS_TAG)):
        rs = p.separate_sentence(pl)
        records.extend(rs)
        record_test_data.extend([r.get_x(x_encoder) for r in rs])
        
    model_path = os.path.join(config.REGEX_PATH, "record_model")
    x_encoder = pickle.load(open(os.path.join(model_path, "x_encoder.p"), "rb"))
    y_encoder = pickle.load(open(os.path.join(model_path, "y_encoder.p"), "rb"))
    
    # 3.2 Prepare model
    record_model = LSTMCRFTagger(logger, args, x_encoder, y_encoder)
    if not os.path.exists(model_path):
        raise ValueError("No model found!")
    record_model.load_state_dict(torch.load(os.path.join(model_path, "final.pt")))
    record_model.eval()
    tagged_sent = record_model.evaluate_model(record_test_data, y_encoder)
    for record, tag_list in zip(records, tagged_sent):
        record.set_tag(tag_list)
        
    # Step 4. Saving
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.saver_type == "html":
        saver = HtmlSaver(records)
        filename = os.path.join(config.OUTPUT_PATH, "test_{}.txt".format(curr_time))
    else:
        saver = ExcelSaver(records)
        filename = os.path.join(config.OUTPUT_PATH, "test_{}.xlsx".format(curr_time))
    saver.save(filename, y_encoder.tag_dict.values())

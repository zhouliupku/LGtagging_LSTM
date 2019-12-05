# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 22:43:13 2019

@author: Zhou
"""

import os
import re
import torch
import numpy as np
from torch import optim
import datetime
import logging
import itertools
import argparse

from model import LSTMTagger, TwoLayerLSTMTagger, LSTMCRFTagger
from config import NULL_TAG, INS_TAG, EOS_TAG
from Encoders import XEncoder, YEncoder
import lg_utils
import config
from data_save import ExcelSaver, HtmlSaver


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--data_size', type=str, default='tiny',
                    choices=['tiny', 'small', 'medium', 'full'],
                    help='Size of training data')
parser.add_argument('--saver_type', type=str, default='html', 
                    choices=['html', 'excel'],
                    help='Type of saver')
parser.add_argument('--task_type', type=str, default='page', 
                    choices=['page', 'record'],
                    help='Type of task')
parser.add_argument('--loss_type', type=str, default='NLL', 
                    choices=['NLL'],
                    help='Type of loss function')
parser.add_argument('--n_epoch', type=int, default=50, 
                    help='Number of epoch')
parser.add_argument('--learning_rate', type=float, default=0.05, 
                    help='Learning rate')
parser.add_argument('--hidden_dim', type=int, default=6,
                    help='Hidden dim')
parser.add_argument('--bidirectional', type=str2bool, default=False, 
                    help='Boolean indicating whether bidirectional is enabled')
parser.add_argument('--need_train', type=str2bool, default=True, 
                    help='Boolean indicating whether need training model')
args = parser.parse_args()


if __name__ == "__main__":
    print("\nParameters:")
    for attr, value in args.__dict__.items():
        print("\t{} = {}".format(attr.upper(), value))
        
    model_path = os.path.join(config.REGEX_PATH, "{}_model".format(args.task_type))
    USE_REGEX = False
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Logging
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(filename=os.path.join("log", "run{}.log".format(curr_time)),
                        format='%(asctime)s %(message)s', 
                        filemode='w') 
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info("Started training at {}".format(curr_time))
    
    # TODO: train
    # TODO: test
    
    # Load in data
    raw_train = lg_utils.load_data_from_pickle("{}s_train.p".format(args.task_type),
                                               args.data_size)
    raw_cv = lg_utils.load_data_from_pickle("{}s_cv.p".format(args.task_type),
                                               args.data_size)
    raw_test = lg_utils.load_data_from_pickle("{}s_test.p".format(args.task_type),
                                               args.data_size)
    
    char_encoder = XEncoder(config.EMBEDDING_PATH)
    vars(args)['embedding_dim'] = char_encoder.embedding_dim
    if args.task_type == "page":
        tag_encoder = YEncoder([INS_TAG, EOS_TAG])
    else:
        tagset = set(itertools.chain.from_iterable([r.orig_tags for r in raw_train]))
        tagset = ["<BEG>", "<END>"] + sorted(list(tagset))
        tag_encoder = YEncoder(tagset)
#    vars(args)['tag_dim'] = char_encoder.embedding_dim
    model = LSTMCRFTagger(logger, args, tag_encoder.get_tag_dim())
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    
    # Load models if it was previously saved and want to continue
    if os.path.exists(model_path) and not args.need_train:
        model.load_state_dict(torch.load(os.path.join(model_path, "final.pt")))
        model.eval()
    
    # Training
    # Step 1. Data preparation
    training_data = lg_utils.get_data_from_samples(raw_train, char_encoder, tag_encoder)
    cv_data = lg_utils.get_data_from_samples(raw_cv, char_encoder, tag_encoder)
    test_data = lg_utils.get_data_from_samples(raw_test, char_encoder, tag_encoder)
    
    # TODO: train
    
    # Step 2. Model training
    if args.need_train:
        model.train_model(training_data, cv_data,  optimizer, args, model_path)
        
    # Step 3. Evaluation with correct ratio
    lg_utils.correct_ratio_calculation(raw_train, model, "train", char_encoder, tag_encoder)
    lg_utils.correct_ratio_calculation(raw_cv, model, "cv",char_encoder, tag_encoder)
    lg_utils.correct_ratio_calculation(raw_test, model, "test",char_encoder, tag_encoder)
    raise RuntimeError
    
    # TODO: produce
    # Step 1. using page_to_sent_model, parse pages to sentences
    if USE_REGEX:
        with open(os.path.join(config.REGEX_PATH, "surname.txt"), 'r', encoding="utf8") as f:
            surnames = f.readline().replace("\ufeff", '')
        tag_seq_list = []
        for p in pages_test:
            tags = [INS_TAG for c in p.txt]
            for m in re.finditer(r"○("+surnames+')', p.txt):
                tags[m.start(0)] = EOS_TAG  # no need to -1, instead drop '○' before name
            tags[-1] = EOS_TAG
            tag_seq_list.append(tags)
    else:
        tag_seq_list = page_model.evaluate_model(page_test_data, page_tag_encoder)
    record_test_data = []
    records = []
    for p, pl in zip(pages_test, lg_utils.get_sent_len_for_pages(tag_seq_list, EOS_TAG)):
        rs = p.separate_sentence(pl)
        records.extend(rs)
        record_test_data.extend([r.get_x(char_encoder) for r in rs])
            
#     Step 2. using sent_to_tag_model, tag each sentence
    tagged_sent = record_model.evaluate_model(record_test_data, record_tag_encoder)
    for record, tag_list in zip(records, tagged_sent):
        record.set_tag(tag_list)
    
        
#     Saving
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.saver_type == "html":
        saver = HtmlSaver(records)
        filename = os.path.join(config.OUTPUT_PATH, "test_{}.txt".format(curr_time))
    else:
        saver = ExcelSaver(records)
        filename = os.path.join(config.OUTPUT_PATH, "test_{}.xlsx".format(curr_time))
    saver.save(filename, interested_tags)

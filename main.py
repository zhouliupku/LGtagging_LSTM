# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 22:43:13 2019

@author: Zhou
"""

import os
import re
import torch
import numpy as np
import datetime
import logging
import argparse

import lg_utils
import config
import process
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
    
    process.train(logger, args)
    process.test(logger, args)
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

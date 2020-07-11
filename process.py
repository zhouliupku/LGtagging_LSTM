# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 21:31:29 2019

@author: Zhou
"""

import os
import re
import datetime
import itertools

import lg_utils
import config
from Encoders import XEncoder, YEncoder
from data_save import ExcelSaver, HtmlSaver
from model import ModelFactory


def train(logger, args):
    """
    Training model
    """
    # Load in data
    raw_train = lg_utils.load_data_from_pickle("{}s_train.p".format(args.task_type),
                                               args.data_size)
    raw_cv = lg_utils.load_data_from_pickle("{}s_cv.p".format(args.task_type),
                                               args.data_size)
    print("Loading done")
    logger.info("Loading done")
    
        
    # Set up model
    if args.start_from_epoch >= 0:
        model = ModelFactory().get_trained_model(logger, args)
    else:
        # Set up encoders
        char_encoder = XEncoder(args)
        if args.task_type == "page":
            tag_encoder = YEncoder([config.INS_TAG, config.EOS_TAG])
        else:
            tagset = sorted(list(set(itertools.chain.from_iterable([r.orig_tags for r in raw_train])))) 
            tag_encoder = YEncoder(tagset)
        model = ModelFactory().get_new_model(logger, args, char_encoder, tag_encoder)
    
    # Training
    # Step 1. Data preparation
    training_data = [(p.get_x(), p.get_y()) for p in raw_train]
    cv_data = [(p.get_x(), p.get_y()) for p in raw_cv]
    
    # Step 2. Model training
    if args.need_train:
        model.train_model(training_data, cv_data, args)
    else:
        model = ModelFactory().get_trained_model(logger, args)
        
        
def test(logger, args):
    """
    Test trained model on set-alone data; this should be done after all tunings
    """
    raw_test = lg_utils.load_data_from_pickle("{}s_test.p".format(args.task_type),
                                               args.data_size)
    data = [(p.get_x(), p.get_y()) for p in raw_test]
    model = ModelFactory().get_trained_model(logger, args)
    
    single_batches = model.make_padded_batch(data, 1)
    model.log_and_print("Test Evaluation")
    model.evaluate_core(single_batches, args)
    
    
def produce(logger, args):
    """
    Produce untagged data using model; this step is unsupervised
    """
    # Step 1. using page_to_sent_model, parse pages to sentences
    pages_produce = lg_utils.load_data_from_pickle("pages_produce.p", args.data_size)
#    pages_produce = lg_utils.load_data_from_pickle("pages_cv.p", args.data_size)
    
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
            tag_seq_list.append([config.BEG_TAG] + tags + [config.END_TAG])
    else:
        vars(args)["task_type"] = "page"
        page_model = ModelFactory().get_trained_model(logger, args)
        pages_data = [p.get_x() for p in pages_produce]
        tag_seq_list = page_model.evaluate_model(pages_data, args) # list of list of tag
            
    #   Step 3. using trained record model, tag each sentence
    # Get model
    vars(args)["task_type"] = "record"
    record_model = ModelFactory().get_trained_model(logger, args)
    record_test_data = []       # list of list of str
    records = []                # list of Record
    
    for p, pl in zip(pages_produce, 
                     lg_utils.get_sent_len_for_pages(tag_seq_list, config.EOS_TAG)):
        rs = p.separate_sentence(pl)        # get a list of Record instances
        records.extend(rs)
        record_test_data.extend([r.get_x() for r in rs])
        
    # Use trained model to process
    tagged_sent = record_model.evaluate_model(record_test_data, args)
    for record, tag_list in zip(records, tagged_sent):
        record.set_tag(tag_list)
        
    # Step 4. Saving
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.saver_type == "html":
        saver = HtmlSaver(records)
        filename = os.path.join(config.OUTPUT_PATH, "test_{}.txt".format(curr_time))
    else:
        raise ValueError("Unsupported save type: " + args.saver_type)
#        saver = ExcelSaver(records)
#        filename = os.path.join(config.OUTPUT_PATH, "test_{}.xlsx".format(curr_time))
    saver.save(filename, record_model.y_encoder.tag_dict.values())

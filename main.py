# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 22:43:13 2019

@author: Zhou
"""

import os
import torch
import numpy as np
import pandas as pd
from torch import optim
import datetime
import logging

from model import LSTMTagger, TwoLayerLSTMTagger
from config import NULL_TAG, INS_TAG, EOS_TAG
from Encoders import XEncoder, YEncoder
from data_load import XYDataLoader, HtmlDataLoader
import lg_utils

if __name__ == "__main__":
    #TODO: argparse
    # I/O settings
    OUTPUT_PATH = os.path.join(os.getcwd(), "Autoparse")
    MODEL_PATH = os.path.join(os.getcwd(), "models")
    PAGE_MODEL_PATH = os.path.join(MODEL_PATH, "page_model")
    RECORD_MODEL_PATH = os.path.join(MODEL_PATH, "record_model")
    EMBEDDING_PATH = os.path.join(os.getcwd(), "Embedding", "polyglot-zh_char.pkl")
    SOURCE_TYPE = "html"
        
    # Training settings
    N_SECTION_TRAIN = 30
    N_SECTION_TEST = 1
    CV_PERC = 0.5
    
    N_EPOCH_PAGE = 60
    N_CHECKPOINT_PAGE = 1
    N_SAVE_PAGE = 5
    LEARNING_RATE_PAGE = 0.25
    HIDDEN_DIM_PAGE = 12
    N_EPOCH_RECORD = 60
    N_CHECKPOINT_RECORD = 1
    N_SAVE_RECORD = 5
    LEARNING_RATE_RECORD = 0.3
    HIDDEN_DIM_RECORD = 12
    
    NEED_TRAIN_MODEL = True
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Logging
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(filename=os.path.join("log",
                                              "run{}.log".format(curr_time)),
                        format='%(asctime)s %(message)s', 
                        filemode='w') 
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info("Started training at {}".format(curr_time))
    
    # Set up data loaders    
    if SOURCE_TYPE == "XY":
        loader = XYDataLoader()
    elif SOURCE_TYPE == "html":
        loader = HtmlDataLoader()
    else:
        raise ValueError
    test_loader = XYDataLoader()
    
    # Model hyper-parameter definition
    EMBEDDING_DIM = 64          # depending on pre-trained word embedding model
    char_encoder = XEncoder(EMBEDDING_DIM, EMBEDDING_PATH)
#    interested_tags = ["人名", "任職時間", "籍貫", "入仕方法"]
    interested_tags = [loader.get_person_tag()]
    interested_tags.extend(["post_time", "office", "jiguan"])
#    interested_tags.extend(["任職時間"])
    page_tag_encoder = YEncoder([INS_TAG, EOS_TAG])
    record_tag_encoder = YEncoder([NULL_TAG, "<BEG>", "<END>"] + interested_tags)
    page_model = TwoLayerLSTMTagger(logger, EMBEDDING_DIM, HIDDEN_DIM_PAGE,
                                    page_tag_encoder.get_tag_dim(), bidirectional=True)
    record_model = LSTMTagger(logger, EMBEDDING_DIM, HIDDEN_DIM_RECORD, 
                              record_tag_encoder.get_tag_dim(), bidirectional=True)
    page_optimizer = optim.SGD(page_model.parameters(), lr=LEARNING_RATE_PAGE)
    record_optimizer = optim.SGD(record_model.parameters(), lr=LEARNING_RATE_RECORD)
    
    # Load training, CV and testing data
    pages_train, pages_cv, records_train, records_cv = loader.load_data(interested_tags,
                                                                    "train",
                                                                    N_SECTION_TRAIN,
                                                                    cv_perc=CV_PERC)
    pages_test, _ = test_loader.load_data(interested_tags, 
                                            "test",
                                            N_SECTION_TEST)
    
    
    # Load models if it was previously saved and want to continue
    if os.path.exists(PAGE_MODEL_PATH) and not NEED_TRAIN_MODEL:
        page_model.load_state_dict(torch.load(os.path.join(PAGE_MODEL_PATH, "final.pt")))
        page_model.eval()
    if os.path.exists(RECORD_MODEL_PATH) and not NEED_TRAIN_MODEL:
        record_model.load_state_dict(torch.load(os.path.join(RECORD_MODEL_PATH, "final.pt")))
        record_model.eval()
    
    # Training
    # Step 1. Data preparation
    page_training_data = lg_utils.get_data_from_samples(pages_train,
                                                        char_encoder,
                                                        page_tag_encoder)
    page_cv_data = lg_utils.get_data_from_samples(pages_cv,
                                                  char_encoder,
                                                  page_tag_encoder)
    page_test_data = [p.get_x(char_encoder) for p in pages_test]
    
    record_training_data = lg_utils.get_data_from_samples(records_train,
                                                          char_encoder,
                                                          record_tag_encoder)
    record_cv_data = lg_utils.get_data_from_samples(records_cv,
                                                    char_encoder,
                                                    record_tag_encoder)
    
    # Step 2. Model training
    if NEED_TRAIN_MODEL:
        # 2.a Train model to parse pages into sentences
        page_model.train_model(page_training_data, page_cv_data, 
                               page_optimizer, "NLL",
                               N_EPOCH_PAGE, N_CHECKPOINT_PAGE,
                               N_SAVE_PAGE, PAGE_MODEL_PATH)
        # 2.b Train model to tag sentences
        record_model.train_model(record_training_data, record_cv_data, 
                                 record_optimizer, "NLL",
                                 N_EPOCH_RECORD, N_CHECKPOINT_RECORD,
                                 N_SAVE_RECORD, RECORD_MODEL_PATH)
    
    # Evaluate on test set
    # Step 1. using page_to_sent_model, parse pages to sentences
    tag_seq_list = page_model.evaluate_model(page_test_data, page_tag_encoder)
    record_test_data = []
    records = []
    for p, pl in zip(pages_test, lg_utils.get_sent_len_for_pages(tag_seq_list, EOS_TAG)):
        rs = p.separate_sentence(pl)
        records.extend(rs)
        record_test_data.extend([r.get_x(char_encoder) for r in rs])
            
    # Step 2. using sent_to_tag_model, tag each sentence
    tagged_sent = record_model.evaluate_model(record_test_data, record_tag_encoder)
    tagged_result = pd.DataFrame(columns=interested_tags)
    for record, tag_list in zip(records, tagged_sent):
        record.set_tag(tag_list)
        tagged_result = tagged_result.append(record.get_tag_res_dict(interested_tags),
                                                 ignore_index=True)
    tagged_result.to_excel(os.path.join(OUTPUT_PATH, "test.xlsx"), index=False)

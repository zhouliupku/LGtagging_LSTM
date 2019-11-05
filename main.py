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

from model import LSTMTagger
from Encoders import XEncoder, YEncoder
from data_load import DataLoader
import lg_utils

if __name__ == "__main__":
    #TODO: argparse
    # I/O settings
    OUTPUT_PATH = os.path.join(os.getcwd(), "Autoparse")
    MODEL_PATH = os.path.join(os.getcwd(), "models")
    PAGE_MODEL_PATH = os.path.join(MODEL_PATH, "page_model.pt")
    TAG_MODEL_PATH = os.path.join(MODEL_PATH, "tag_model.pt")
    EMBEDDING_PATH = os.path.join(os.getcwd(), "Embedding", "polyglot-zh_char.pkl")
    SOURCE_TYPE = "xy"
        
    # Training settings
    N_EPOCH = 30
    N_CHECKPOINT = 2
    LEARNING_RATE = 0.3
    CV_PERC = 0.5
    NEED_TRAIN_MODEL = True
    NEED_SAVE_MODEL = True
    EOS_TAG = 'S'
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Model hyper-parameter definition
    EMBEDDING_DIM = 64          # depending on pre-trained word embedding model
    HIDDEN_DIM = 6
    char_encoder = XEncoder(EMBEDDING_DIM, EMBEDDING_PATH)
    interested_tag_tuples = [("人名", 'R'), ("任職時間", 'T'), ("籍貫", 'L'),
                             ("入仕方法", 'E')]
    interested_tags = [item[1] for item in interested_tag_tuples]
    page_tag_encoder = YEncoder(["N", EOS_TAG])
    sent_tag_encoder = YEncoder(["N", "<BEG>", "<END>"] + interested_tags)
    page_to_sent_model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, page_tag_encoder.get_tag_dim())
    sent_to_tag_model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, 
                                   sent_tag_encoder.get_tag_dim(), bidirectional=True)
    sent_optimizer = optim.SGD(page_to_sent_model.parameters(), lr=LEARNING_RATE)
    tag_optimizer = optim.SGD(sent_to_tag_model.parameters(), lr=LEARNING_RATE)
    
    # Load training, CV and testing data
    loader = DataLoader(SOURCE_TYPE)
    pages_train, pages_cv, pages_test = loader.load_data(CV_PERC, interested_tag_tuples)
    
    # Load models if it was previously saved and want to continue
    if os.path.exists(PAGE_MODEL_PATH) and not NEED_TRAIN_MODEL:
        page_to_sent_model.load_state_dict(torch.load(PAGE_MODEL_PATH))
        page_to_sent_model.eval()
    if os.path.exists(TAG_MODEL_PATH) and not NEED_TRAIN_MODEL:
        sent_to_tag_model.load_state_dict(torch.load(TAG_MODEL_PATH))
        sent_to_tag_model.eval()
    
    # Training
    # Step 1. Data preparation
    page_to_sent_training_data = lg_utils.get_page_data_from_pages(pages_train,
                                                                   char_encoder,
                                                                   page_tag_encoder)
    page_to_sent_cv_data = lg_utils.get_page_data_from_pages(pages_cv,
                                                             char_encoder,
                                                             page_tag_encoder)
    page_to_sent_test_data = [p.get_x(char_encoder) for p in pages_test]
    sent_to_tag_training_data = lg_utils.get_sent_data_from_pages(pages_train,
                                                                  char_encoder,
                                                                  sent_tag_encoder)
    sent_to_tag_cv_data = lg_utils.get_sent_data_from_pages(pages_cv,
                                                            char_encoder,
                                                            sent_tag_encoder)
    
    # Step 2. Model training
    if NEED_TRAIN_MODEL:
        # 2.a Train model to parse pages into sentences
        page_to_sent_model.train_model(page_to_sent_training_data, page_to_sent_cv_data, 
                                       sent_optimizer, "NLL", N_EPOCH, N_CHECKPOINT)
        # 2.b Train model to tag sentences
        sent_to_tag_model.train_model(sent_to_tag_training_data, sent_to_tag_cv_data, 
                                      tag_optimizer, "NLL", N_EPOCH, N_CHECKPOINT)
    
    # Save models
    if NEED_SAVE_MODEL:
        torch.save(page_to_sent_model.state_dict(), PAGE_MODEL_PATH)
        torch.save(sent_to_tag_model.state_dict(), TAG_MODEL_PATH)
    
    # Evaluate on test set
    # Step 1. using page_to_sent_model, parse pages to sentences
    tag_seq_list = page_to_sent_model.evaluate_model(page_to_sent_test_data, 
                                                     page_tag_encoder)
    sent_to_tag_test_data = []
    for p, pl in zip(pages_test, lg_utils.get_sent_len_for_pages(tag_seq_list, EOS_TAG)):
        p.separate_sentence(pl)
        for r in p.get_records():
            sent_to_tag_test_data.append(r.get_x(char_encoder))
            
    # Step 2. using sent_to_tag_model, tag each sentence
    tagged_sent = sent_to_tag_model.evaluate_model(sent_to_tag_test_data,
                                                   sent_tag_encoder)
    assert len(tagged_sent) == sum([len(p.get_records()) for p in pages_test])
    for p in pages_test:
        num_records = len(p.get_records())
        p.tag_records(tagged_sent[:num_records])
        tagged_sent = tagged_sent[num_records:]
        p.print_sample_records(3)
        
    # Step 3. Save results to csv files
    tagged_result = pd.DataFrame(columns=[item[0] for item in interested_tag_tuples])
    for p in pages_test:
        for r in p.get_records():
            tagged_result = tagged_result.append(r.get_tag_res_dict(interested_tag_tuples),
                                                 ignore_index=True)
    tagged_result.to_excel(os.path.join(OUTPUT_PATH, "test.xlsx"), index=False)

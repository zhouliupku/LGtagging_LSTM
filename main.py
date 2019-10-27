# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 22:43:13 2019

@author: Zhou
"""

import os
import torch
import pandas as pd
from torch import optim

from DataStructures import Page
from model import LSTMTagger
from Encoders import XEncoder, YEncoder
from utils import get_sent_len_for_pages

def load_data(text_filename, tag_filename, mode):
    with open(text_filename, 'r', encoding="utf8") as txtfile:
        lines = txtfile.readlines()
    if mode == "train":
        df = pd.read_excel(tag_filename)
    else:
        df = None
    pages = []
    for line in lines:
        if '【' not in line or '】' not in line:
            continue
        pages.append(Page(line, df, mode, interested_tag_tuples))
    return pages

      
if __name__ == "__main__":
    #TODO: argparse
    # I/O settings
    INPUT_PATH = os.path.join(os.getcwd(), "LSTMdata")
    OUTPUT_PATH = os.path.join(os.getcwd(), "Autoparse")
    MODEL_PATH = os.path.join(os.getcwd(), "models")
    PAGE_MODEL_PATH = os.path.join(MODEL_PATH, "page_model.pt")
    TAG_MODEL_PATH = os.path.join(MODEL_PATH, "tag_model.pt")
    EMBEDDING_PATH = os.path.join(os.getcwd(), "Embedding", "polyglot-zh_char.pkl")
        
    # Training settings
    N_EPOCH = 30
    N_CHECKPOINT = 5
    LEARNING_RATE = 0.2
    N_TRAIN = 15
    TRAIN_FROM_SCRATCH = False
    NEED_SAVE_MODEL = False
    EOS_TAG = 'S'
    
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
    
    # Load training and testing data
    pages_train = load_data(os.path.join(INPUT_PATH, "textTraining2.txt"),
                               os.path.join(INPUT_PATH, "tagTraining2.xlsx"),
                               "train")[:N_TRAIN]
    print("Loaded {} pages for training.".format(len(pages_train)))
    pages_test = load_data(os.path.join(INPUT_PATH, "textTraining2.txt"), None,
                               "test")[N_TRAIN:]
    print("Loaded {} pages for testing.".format(len(pages_test)))
    
    # Load models if it was previously saved and want to continue
    if os.path.exists(PAGE_MODEL_PATH) and not TRAIN_FROM_SCRATCH:
        page_to_sent_model.load_state_dict(torch.load(PAGE_MODEL_PATH))
        page_to_sent_model.eval()
    if os.path.exists(TAG_MODEL_PATH) and not TRAIN_FROM_SCRATCH:
        sent_to_tag_model.load_state_dict(torch.load(TAG_MODEL_PATH))
        sent_to_tag_model.eval()
    
    # Training
    # Step 1. Data preparation
    torch.manual_seed(1)
    page_to_sent_training_data = [(p.get_x(char_encoder), 
                                   p.get_y(page_tag_encoder)) for p in pages_train]
    page_to_sent_test_data = [p.get_x(char_encoder) for p in pages_test]
    sent_to_tag_training_data = []
    for p in pages_train:
        for r in p.get_records():
            sent_to_tag_training_data.append((r.get_x(char_encoder), 
                                              r.get_y(sent_tag_encoder)))
    
    # Step 2. Model training
    # 2.a Train model to parse pages into sentences
    page_to_sent_model.train(page_to_sent_training_data, sent_optimizer, "NLL",
                             N_EPOCH, N_CHECKPOINT)
    # 2.b Train model to tag sentences
    sent_to_tag_model.train(sent_to_tag_training_data, tag_optimizer, "NLL",
                            N_EPOCH, N_CHECKPOINT)
    
    # Save models
    if NEED_SAVE_MODEL:
        torch.save(page_to_sent_model.state_dict(), PAGE_MODEL_PATH)
        torch.save(sent_to_tag_model.state_dict(), TAG_MODEL_PATH)
    
    # Evaluate on test set
    # Step 1. using page_to_sent_model, parse pages to sentences
    tag_seq_list = page_to_sent_model.evaluate(page_to_sent_test_data, 
                                               page_tag_encoder)
    sent_to_tag_test_data = []
    for p, pl in zip(pages_test, get_sent_len_for_pages(tag_seq_list, EOS_TAG)):
        p.separate_sentence(pl)
        for r in p.get_records():
            sent_to_tag_test_data.append(r.get_x(char_encoder))
            
    # Step 2. using sent_to_tag_model, tag each sentence
    tagged_sent = sent_to_tag_model.evaluate(sent_to_tag_test_data,
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

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 22:43:13 2019

@author: Zhou
"""

import os
import torch
import pandas as pd
from torch import nn, optim

from DataStructures import Page
from model import LSTMTagger
from Encoders import XEncoder, YEncoder
import utils

def load_training_data(text_filename, tag_filename, mode):
    with open(text_filename, 'r', encoding="utf8") as txtfile:
        lines = txtfile.readlines()
    df = pd.read_excel(tag_filename)
    print(len(df))
    pages = []
    for line in lines:
        if '【' not in line or '】' not in line:
            continue
        pages.append(Page(line, df, mode))
    return pages

def train(training_data, model, optimizer, loss_function):
    '''
    model is modifiable
    '''
    #TODO: move into models
    for epoch in range(N_EPOCH):    #TODO: training settings as a class
        print(epoch)
        for sentence, tags in training_data:
            model.zero_grad()   # clear accumulated gradient before each instance
            targets = model.prepare_targets(tags)
            tag_scores = model(sentence)
            loss = loss_function(tag_scores, targets)
            loss.backward(retain_graph=True)
            optimizer.step()
        if epoch % N_CHECKPOINT == 0:
            print("Epoch {}".format(epoch))
            print("Loss = {}".format(loss.item()))
    return model

def evaluate(model, test_data):
    """
    Take model and test data (list of strings), return list of list of tags
    """
    print("Testing set:")
    result_list = []
    with torch.no_grad():
        for test_sent in test_data:
            if len(test_sent) == 0:
                continue
            tag_scores = model(test_sent)
            res = model.convert_results(tag_scores.max(dim=1).indices)
            result_list.append(res)
    #        if tag_only:
    #            result_list.extend([list(sent) for sent in utils.parse(test_sent, res)])
            print(res)
            #TODO: fix print so as not to print tensor
            print(test_sent)
        return result_list

      
if __name__ == "__main__":
    # I/O settings
    DATAPATH = os.path.join(os.getcwd(), "LSTMdata")
    MODEL_PATH = os.path.join(os.getcwd(), "models")
    PAGE_MODEL_PATH = os.path.join(MODEL_PATH, "page_model.pt")
    TAG_MODEL_PATH = os.path.join(MODEL_PATH, "tag_model.pt")
    EMBEDDING_PATH = os.path.join(os.getcwd(), "Embedding", "polyglot-zh_char.pkl")
        
    # Training settings
    N_EPOCH = 30
    N_CHECKPOINT = 10
    LEARNING_RATE = 0.3
    N_TRAIN = 15
    TRAIN_FROM_SCRATCH = True
    
    # Model hyper-parameter definition
    page_tag_dict = {"S": 0, "N": 1}
    sent_tag_dict = {"N": 0, "R": 1, "L": 2, "T": 3, "<BEG>": 4, "<END>": 5}
    EMBEDDING_DIM = 64          # depending on pre-trained word embedding model
    HIDDEN_DIM = 6
    char_encoder = XEncoder(EMBEDDING_DIM, EMBEDDING_PATH)
    page_to_sent_model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, page_tag_dict )
    sent_to_tag_model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, sent_tag_dict, bidirectional=True)
    loss_function = nn.NLLLoss()
    sent_optimizer = optim.SGD(page_to_sent_model.parameters(), lr=LEARNING_RATE)
    tag_optimizer = optim.SGD(sent_to_tag_model.parameters(), lr=LEARNING_RATE)
    
    # Load training and testing data
    pages_train = load_training_data(os.path.join(DATAPATH, "textTraining1.txt"),
                               os.path.join(DATAPATH, "tagTraining1.xlsx"),
                               "train")[:N_TRAIN]
    print("Loaded {} pages for training.".format(len(pages_train)))
    pages_test = load_training_data(os.path.join(DATAPATH, "textTraining1.txt"),
                               os.path.join(DATAPATH, "tagTraining1.xlsx"),
                               "test")[N_TRAIN:]
    print("Loaded {} pages for testing.".format(len(pages_test)))
    
    # Load model if it was previously saved and want to continue
    if os.path.exists(PAGE_MODEL_PATH) and not TRAIN_FROM_SCRATCH:
        page_to_sent_model.load_state_dict(torch.load(PAGE_MODEL_PATH))
        page_to_sent_model.eval()
    
    torch.manual_seed(1)
    
    # Step 1. Train model to parse pages into sentences
    page_to_sent_training_data = [(p.get_x(char_encoder), p.get_y()) for p in pages_train]
    page_to_sent_test_data = [p.get_x(char_encoder) for p in pages_test]
    page_to_sent_model = train(page_to_sent_training_data, page_to_sent_model,
                               sent_optimizer, loss_function)
    
    # Step 2. Train model to tag sentences
    sent_to_tag_training_data = []
    for p in pages_train:
        for r in p.get_records():
            sent_to_tag_training_data.append((r.get_x(char_encoder), r.get_y()))
    sent_to_tag_model = train(sent_to_tag_training_data, sent_to_tag_model, 
                              tag_optimizer, loss_function)
    
    # Save models
    torch.save(page_to_sent_model.state_dict(), PAGE_MODEL_PATH)
    torch.save(sent_to_tag_model.state_dict(), TAG_MODEL_PATH)
    
    # Evaluate on test set
    # Step 1. using page_to_sent_model, parse pages to sentences
    tag_seq_list = evaluate(page_to_sent_model, page_to_sent_test_data)
    parsed_sent_len_for_pages = []
    for tag_seq in tag_seq_list:
        # make list of int (i.e. sentence lengths) out of list of tags
        parsed_sent_len = []
        current_len = 0
        for tag in tag_seq:
            current_len += 1
            if tag == 'S':
                parsed_sent_len.append(current_len)
                current_len = 0
        parsed_sent_len_for_pages.append(parsed_sent_len)
    sent_to_tag_test_data = []
    for p, pl in zip(pages_test, parsed_sent_len_for_pages):
        p.separate_sentence(pl)
        for r in p.get_records():
            sent_to_tag_test_data.append(r.get_x(char_encoder))
            
    # Step 2. using sent_to_tag_model, tag each sentence
    tagged_sent = evaluate(sent_to_tag_model, sent_to_tag_test_data)

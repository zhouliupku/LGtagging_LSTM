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
import utils

def load_training_data(text_filename, tag_filename):
    with open(text_filename, 'r', encoding="utf8") as txtfile:
        lines = txtfile.readlines()
    df = pd.read_excel(tag_filename)
    print(len(df))
    pages = []
    for line in lines:
        if '【' not in line or '】' not in line:
            continue
        pages.append(Page(line, df, "train"))
    return pages

            
if __name__ == "__main__":
    # I/O settings
    DATAPATH = os.path.join(os.getcwd(), "LSTMdata")
    MODEL_PATH = os.path.join(os.getcwd(), "models")
    PAGE_MODEL_PATH = os.path.join(MODEL_PATH, "page_model.pt")
    
    pages = load_training_data(os.path.join(DATAPATH, "textTraining1.txt"),
                               os.path.join(DATAPATH, "tagTraining1.xlsx"))
    
    print("Loaded {} pages.".format(len(pages)))
    
    # Model hyper-parameter definition
    page_tag_dict = {"B": 0, "N": 1}
    EMBEDDING_DIM = 64          # depending on pre-trained word embedding model
    HIDDEN_DIM = 8
    page_to_sent_model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, page_tag_dict)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(page_to_sent_model.parameters(), lr=0.3)
    
    # Load model if it was previously saved
    if os.path.exists(PAGE_MODEL_PATH):
        page_to_sent_model.load_state_dict(torch.load(PAGE_MODEL_PATH))
        page_to_sent_model.eval()
    
    torch.manual_seed(1)
    
    # Step 1. Train model to parse pages into sentences
    training_data = [(p.get_x(), p.get_y()) for p in pages]    
    for epoch in range(50):
        print(epoch)
        #TODO: use full sample
        for sentence, tags in training_data[:10]:
            page_to_sent_model.zero_grad()   # clear accumulated gradient before each instance
            #TODO: see whether '○' is convert as <UNK>
            sentence_in = page_to_sent_model.prepare_sequence(sentence)
            targets = page_to_sent_model.prepare_targets(tags)
            tag_scores = page_to_sent_model(sentence_in)
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print("Epoch {}".format(epoch))
            print("Loss = {}".format(loss.item()))
            with torch.no_grad():
                for training_sent, _ in training_data[:3]:
                    inputs = page_to_sent_model.prepare_sequence(training_sent)
                    tag_scores = page_to_sent_model(inputs)
                    res = page_to_sent_model.convert_results(tag_scores.max(dim=1).indices)
                    print(res)
                    for sent in utils.parse(training_sent, res):
                        print(sent)
    # After training, save trained model
    torch.save(page_to_sent_model.state_dict(), PAGE_MODEL_PATH)
    

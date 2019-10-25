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

def load_data(text_filename, tag_filename):
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
    DATAPATH = os.path.join(os.getcwd(), "LSTMdata")
    torch.manual_seed(1)
    
    pages = load_data(os.path.join(DATAPATH, "textTraining1.txt"),
                      os.path.join(DATAPATH, "tagTraining1.xlsx"))
    
    print(pages[0].get_x())
    print(pages[0].get_y())
    print(len(pages[0].get_x()))
    print(len(pages[0].get_y()))
    
    
    tag_to_ix = {"B": 0, "N": 1}
    tagix_to_tag = {v:k for k,v in tag_to_ix.items()}
    
    def print_results(tagix):
        print("".join([tagix_to_tag[t.item()] for t in tagix]))
       
    EMBEDDING_DIM = 64          # depending on pre-trained word embedding model
    HIDDEN_DIM = 8
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(tag_to_ix))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.3)
    
    training_data = [(p.get_x(), p.get_y()) for p in pages]
    
    for epoch in range(2000):
        print(epoch)
        for sentence, tags in training_data:
            model.zero_grad()   # clear accumulated gradient before each instance
            sentence_in = model.prepare_sequence(sentence)
            targets = torch.tensor([tag_to_ix[w] for w in tags], dtype=torch.long)
            tag_scores = model(sentence_in)
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
        if epoch % 2 == 0:
            print("Epoch {}".format(epoch))
            print("Loss = {}".format(loss.item()))
            with torch.no_grad():
                for training_sent, _ in training_data[:3]:
                    inputs = model.prepare_sequence(training_sent)
                    tag_scores = model(inputs)
                    print_results(tag_scores.max(dim=1).indices)
    

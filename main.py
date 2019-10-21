# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 22:43:13 2019

@author: Zhou
"""

import os
import pandas as pd
from DataStructures import Page

def load_data(text_filename, tag_filename):
    with open(text_filename, 'r', encoding='utf8') as txtfile:
        lines = txtfile.readlines()
    df = pd.read_excel(tag_filename)
    pages = []
    for line in lines:
        if '【' not in line or '】' not in line:
            continue
        pages.append(Page(line, df, "train"))
    return pages
            
if __name__ == "__main__":
    DATAPATH = os.path.join(os.getcwd(), 'LSTMdata')
    pages = load_data(os.path.join(DATAPATH, 'textTraining1.txt'),
                      os.path.join(DATAPATH, 'tagTraining1.xlsx'))
    print(len(pages))
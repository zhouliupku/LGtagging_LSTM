# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 15:55:30 2019

@author: Zhou
"""
import pandas as pd
import os

DATAPATH = os.path.join(os.getcwd(), 'LSTMdata')
#with open(os.path.join(DATAPATH,'textTraining1.txt'), 'r', encoding='utf8') as txtfile:
#    lines = txtfile.readlines()
#    
## create the dict of text, the key is the pageID, and the value is the corresponding text   
#textDict = {}
#for line in lines:
#    if '【' not in line or '】' not in line:
#        continue
#    line = line.split('【')[1].strip().split('】')
#    textKeys = int(line[0])
#    textVal = line[1]
#    textDict[textKeys] = textVal
#
##create the dict of tag, the key is the pageID, and the value is the corresponding tag sequences  
#df = pd.read_excel(os.path.join(DATAPATH, 'tagTraining1.xlsx'))
#tagDict = {}
#for tagKey, subdf in df.groupby(by='Page No.'):
#    tagDict[tagKey] = subdf

class Page(object):
    def __init__(self, text_filename, tag_filename):
        with open(text_filename, 'r', encoding='utf8') as txtfile:
            lines = txtfile.readlines()
        df = pd.read_excel(tag_filename)
                    
        # store data in dict (int -> (str, pd.df))
        self.data = {}
        for line in lines:
            if '【' not in line or '】' not in line:
                continue
            line = line.split('【')[1].strip().split('】')
            key = int(line[0])
            text = line[1]
            subdf = df[df['Page No.'] == key]
            self.data[key] = (text, subdf)
            
    def get_all_id(self):
        return self.data.keys()
    
    def get_all_sent(self):
        return [x[0] for x in self.data.values()]
    
    def get_all_tags(self):
        return [x[1] for x in self.data.values()]
    
    def get_sent(self, key):
        if key not in self.data.keys():
            raise ValueError("Key {} not in dataset".format(key))
        return self.data[key][0]
    
    def get_tag(self, key):
        if key not in self.data.keys():
            raise ValueError("Key {} not in dataset".format(key))
        return self.data[key][1]
            
            
if __name__ == "__main__":
    DATAPATH = os.path.join(os.getcwd(), 'LSTMdata')
    test_page = Page(os.path.join(DATAPATH, 'textTraining1.txt'),
                     os.path.join(DATAPATH, 'tagTraining1.xlsx'))
    
    print(test_page.get_all_id())
    print(test_page.get_sent(913750))
    print(test_page.get_tag(913750))
    print(test_page.get_tag(91312313750))

    
    
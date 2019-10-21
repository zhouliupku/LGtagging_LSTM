# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 15:55:30 2019

@author: Zhou
"""

# space char in original text in LoGaRT
SPACE = '○'

class Page(object):
    def __init__(self, line, df, mode):
        line = line.split('【')[1].strip().split('】')
        self.page_id = int(line[0])
        self.orig_text = line[1]
        subdf = df[df['Page No.'] == self.page_id]
        if mode == "train":
            self.records = self.create_records(subdf)
        elif mode == "test":
            self.records = []
        else:
            raise ValueError("Unsupported mode:" + str(mode))
            
    def get_text(self):
        return self.orig_text
    
    def create_records(self, df):
        """
        Create a list of Record objects by parsing with self.orig_text and df
        """
        for _, row in df.iterrows():
            name = row['人名'].replace(' ', SPACE)
            if not name in self.orig_text:
                print(name)

#class Record(object):
    
    
    
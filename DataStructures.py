# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 15:55:30 2019

@author: Zhou
"""

import utils

class Page(object):
    def __init__(self, line, df, mode):
        line = line.split('【')[1].strip().split('】')
        self.page_id = int(line[0])
        self.orig_text = line[1]
        subdf = df[df["Page No."] == self.page_id]
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
        records = []
        datas = []
        last_cutoff = 0
        for _, row in df.iterrows():
            name = utils.convert_to_orig(row["人名"])
            if not name in self.orig_text:
                raise ValueError("Name {} not in original text!".format(name))
            cutoff = self.orig_text.find(name)
            datas.append((self.orig_text[last_cutoff : cutoff], row))
            last_cutoff = cutoff
        for idx, data in enumerate(datas[1:]):
            records.append(Record(idx, data))
        return records
    
    def get_length(self):
        return len(self.records)
    
    def get_orig_text(self):
        return self.orig_text
    
    def get_eos_markers(self):
        raise NotImplementedError()
            

class Record(object):
    def __init__(self, idx, data):
        self.record_id = idx
        self.orig_text = data[0]
        self.orig_tags = data[1]
        tag_seq = self.orig_text    #TODO: create list of Tags with same length
        self.chars = [CharSample(c, t) for c, t in zip(self.orig_text, tag_seq)]
        
    
class CharSample(object):
    def __init__(self, char, tag):
        self.char = char
        self.tag = tag

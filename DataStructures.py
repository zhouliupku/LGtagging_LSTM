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
        texts = []
        tags = []
        last_cutoff = 0
        for _, row in df.iterrows():
            name = utils.convert_to_orig(row["人名"])
            if not name in self.orig_text:
                raise ValueError("Name {} not in original text!".format(name))
            cutoff = self.orig_text.find(name)
            texts.append(self.orig_text[last_cutoff:cutoff])
            tags.append(row)
            last_cutoff = cutoff
        texts.append(self.orig_text[last_cutoff:])
        for idx, data in enumerate(zip(texts[1:], tags)):
            records.append(Record(idx, data))        
        return records
    
    def get_length(self):
        return len(self.records)
        
    def get_x(self):
        """
        get x sequence as tensor
        """
        return list(''.join([r.get_orig_text() for r in self.records]))
        
    def get_y(self):
        """
        get y sequence as tensor
        """
        tags = ['N' for i in range(len(self.get_x()))]
        target = 0
        tags[0] = 'B'
        for record in self.records[:-1]:
            length = record.get_orig_len()
            target += length
            tags[target] = 'B'
        return tags
        

            

class Record(object):
    def __init__(self, idx, data):
        self.record_id = idx
        self.orig_text = data[0]
        self.orig_tags = data[1]
        # Build tag sequence
        tag_seq = ['N' for _ in self.orig_text]
        interested_tags = [("人名", 'R'), ("任官地點", "L"), ("任職時間", "T")]
        for colname, tagname in interested_tags:
            utils.modify_tag_seq(self.orig_text, tag_seq,
                                 self.orig_tags[colname], tagname)
        self.chars = [CharSample(c, t) for c, t in zip(self.orig_text, tag_seq)]
        self.chars = [CharSample("<S>", "<BEG>")] + self.chars
        self.chars = self.chars + [CharSample("</S>", "<END>")]
        
    def get_orig_len(self):
        return len(self.orig_text)
    
    def get_orig_text(self):
        return self.orig_text
        
    def get_x(self):
        """
        get x sequence as tensor
        """
        raise NotImplementedError()
        
    def get_y(self):
        """
        get y sequence as tensor
        """
        raise NotImplementedError()
        
    
class CharSample(object):
    def __init__(self, char, tag):
        self.char = char    # both string
        self.tag = tag
        
    def get_char(self):
        return self.char
    
    def get_tag(self):
        return self.tag
    
    def get_x(self, encoder):
        return encoder.encode(self.char)
    
    def get_y(self, encoder):
        return encoder.encode(self.tag)


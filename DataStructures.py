# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 15:55:30 2019

@author: Zhou
"""

import lg_utils

class Page(object):
    def __init__(self, pid, txt, eos_idx):
        self.pid = pid
        self.txt = txt
        self.eos_idx = eos_idx
    
    def fill_with_xy_file(self, line, df, mode, interested_tags):
        line = line.split('【')[1].strip().split('】')
        self.page_id = int(line[0])
        self.orig_text = line[1]
        self.interested_tags = interested_tags
        if mode == "train":
            subdf = df[df["Page No."] == self.page_id]
            self.records = self.create_records(subdf, self.interested_tags)
            self.is_sentence_separated = True
        elif mode == "test":
            self.records = []
            self.is_sentence_separated = False
        else:
            raise ValueError("Unsupported mode:" + str(mode))
    
    def create_records(self, df, interested_tags):
        """
        Create a list of Record objects by parsing with self.orig_text and df
        """
        records = []
        texts = []
        tags = []
        last_cutoff = 0
        for _, row in df.iterrows():
            name = lg_utils.convert_to_orig(row["人名"])
            if not name in self.orig_text:
                raise ValueError("Name {} not in original text!".format(name))
            cutoff = self.orig_text.find(name)
            texts.append(self.orig_text[last_cutoff:cutoff])
            tags.append(row)
            last_cutoff = cutoff
        texts.append(self.orig_text[last_cutoff:])
        for idx, data in enumerate(zip(texts[1:], tags)):
            rec = Record()
            rec.fill_from_xy(idx, data, interested_tags)
            records.append(rec)
        return records
#    
#    def get_records(self):
#        return self.records
#    
    def separate_sentence(self, parsed_sent_len):
        """
        Separate page to sentences according to parsed_sent_len, list of int
        Return a list of Record
        """
        assert len(self.txt) == sum(parsed_sent_len)
        head_char_idx = 0
        records = []
        for sent_len in parsed_sent_len:
            text = self.txt[head_char_idx : (head_char_idx + sent_len)]
            records.append(Record(text, None))
            head_char_idx += sent_len
        return records
#        
#        # Step 2. Set flag to indicate that separation is done
#        self.is_sentence_separated = True
#        
#    def tag_records(self, tag_sequences):
#        for record, tag_seq in zip(self.get_records(), tag_sequences):
#            record.set_tag(tag_seq)
#            
#    def print_sample_records(self, n_sample):
#        print("Page {}:".format(self.page_id))
#        for r in self.records[0:n_sample]:
#            r.print_tag_results()
        
    def get_x(self, encoder):
        """
        get x sequence as tensor given encoder
        """
        return encoder.encode(self.txt)
        
    def get_y(self, encoder):
        """
        get y sequence as tensor
        """
        tags = ['N' for i in range(len(self.txt))]
        for i in self.eos_idx:
            tags[i] = 'S'
        return encoder.encode(tags)

            

class Record(object):
    def __init__(self, txt, tags):
        """
        idx: record id indicating the index of record in the page it belongs to
        data: tuple of (text, tags)
        """
        self.orig_text = txt        # as a string without <S>, </S>
        self.orig_tags = [None for i in txt] if tags is None else tags
        self.chars = [CharSample(c, t) for c, t in zip(self.orig_text, self.orig_tags)]
        self.chars = [CharSample("<S>", "<BEG>")] + self.chars
        self.chars = self.chars + [CharSample("</S>", "<END>")]
#        
#    def get_orig_len(self):
#        return len(self.orig_text)
#    
#    def get_orig_text(self):
#        return self.orig_text
#    
    def set_tag(self, tag_seq):
        assert len(tag_seq) == len(self.chars)
        for i in range(1, len(tag_seq) - 1):
            self.chars[i].set_tag(tag_seq[i])
        
    def get_tag_res_dict(self, interested_tags):
        """
        For a tagged record, return a dictionary {col_name: [content1, ...]}
        """
        tag_res_dict = {}
        for col_name in interested_tags:
            keywords = lg_utils.get_keywords_from_tagged_record(self.chars[1:-1], col_name)
            tag_res_dict[col_name] = keywords
        return tag_res_dict
        
    def get_x(self, encoder):
        """
        get x sequence as tensor
        """
        return encoder.encode([cs.get_char() for cs in self.chars])
        
    def get_y(self, encoder):
        """
        get y sequence as tensor
        """
        return encoder.encode([cs.get_tag() for cs in self.chars])
        
    
class CharSample(object):
    def __init__(self, char, tag):
        self.char = char    # both string
        self.tag = tag
        
    def get_char(self):
        return self.char
    
    def get_tag(self):
        return self.tag
    
    def set_tag(self, tag):
        self.tag = tag


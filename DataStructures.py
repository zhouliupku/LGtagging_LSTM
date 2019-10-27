# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 15:55:30 2019

@author: Zhou
"""

import utils

class Page(object):
    def __init__(self, line, df, mode, interested_tags):
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
            
    def get_text(self):
        if self.is_sentence_separated:
            return list(''.join([r.get_orig_text() for r in self.records]))
        else:
            return list(self.orig_text)
    
    def create_records(self, df, interested_tags):
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
            records.append(Record(idx, data, interested_tags))        
        return records
    
    def get_records(self):
        return self.records
    
    def separate_sentence(self, parsed_sent_len):
        """
        Separate page to sentences according to parsed_sent_len, list of int
        """
        # Step 1. fill self.records
        assert len(self.orig_text) == sum(parsed_sent_len)
        record_idx = 0
        head_char_idx = 0
        for sent_len in parsed_sent_len:
            text = self.orig_text[head_char_idx : (head_char_idx + sent_len)]
            self.records.append(Record(record_idx, (text, None), self.interested_tags))
            head_char_idx += sent_len
            record_idx += 1
        
        # Step 2. Set flag to indicate that separation is done
        self.is_sentence_separated = True
        
    def tag_records(self, tag_sequences):
        for record, tag_seq in zip(self.get_records(), tag_sequences):
            record.set_tag(tag_seq)
            
    def print_sample_records(self, n_sample):
        print("Page {}:".format(self.page_id))
        for r in self.records[0:n_sample]:
            r.print_tag_results()
        
    def get_x(self, encoder):
        """
        get x sequence as tensor given encoder
        """
        return encoder.encode(self.get_text())
        
    def get_y(self, encoder):
        """
        get y sequence as tensor
        """
        tags = ['N' for t in self.get_text()]
        eos_idx = -1
        for record in self.records:
            length = record.get_orig_len()
            eos_idx += length
            tags[eos_idx] = 'S'
        return encoder.encode(tags)

            

class Record(object):
    def __init__(self, idx, data, interested_tag_tuples):
        """
        idx: record id indicating the index of record in the page it belongs to
        data: tuple of (text, tags)
        """
        self.record_id = idx
        self.orig_text = data[0]        # as a string without <S>, </S>
        self.orig_tags = data[1]        # as single row of pd.df
        # Build tag sequence
        tag_seq = ['N' for _ in self.orig_text]
        
        if self.orig_tags is None:
            # when tags are not provided at initialization, set flag
            self.is_tagged = False
        else:
            # if provided, modify tag_seq and set flag
            for colname, tagname in interested_tag_tuples:
                utils.modify_tag_seq(self.orig_text, tag_seq,
                                     self.orig_tags[colname], tagname)
            self.is_tagged = True
        self.chars = [CharSample(c, t) for c, t in zip(self.orig_text, tag_seq)]
        self.chars = [CharSample("<S>", "<BEG>")] + self.chars
        self.chars = self.chars + [CharSample("</S>", "<END>")]
        
    def get_orig_len(self):
        return len(self.orig_text)
    
    def get_orig_text(self):
        return self.orig_text
    
    def set_tag(self, tag_seq):
        assert len(tag_seq) == len(self.chars)
        for i in range(1, len(tag_seq) - 1):
            self.chars[i].set_tag(tag_seq[i])
        self.is_tagged = True
        
    def print_tag_results(self):
        print("Record {}:".format(self.record_id))
        print(''.join([cs.get_char() for cs in self.chars[1:-1]]))
        print(''.join([cs.get_tag() for cs in self.chars[1:-1]]))
        
    def get_tag_res_dict(self, interested_tag_tuples):
        """
        For a tagged record, return a dictionary {col_name: [content1, ...]}
        """
        if not self.is_tagged:
            return None
        tag_res_dict = {}
        for col_name, tag_name in interested_tag_tuples:
            keywords = utils.get_keywords_from_tagged_record(self.chars[1:-1], tag_name)
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


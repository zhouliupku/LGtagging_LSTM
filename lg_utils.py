# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 00:11:24 2019

@author: Zhou
"""

import re
import numpy as np

# space char in original text in LoGaRT
SPACE = '○'

def convert_to_orig(s):
    """
    Convert string from database to corresponding original text
    """
    return s.replace(' ', SPACE)


def random_separate(xs, perc):
    """
    given a list of objects xs, split it randomly into two parts with the first
    one taking a percentage of perc
    """
    n1 = int(perc * len(xs))
    index_permuted = np.random.permutation(len(xs))
    x1 = [xs[i] for i in index_permuted[:(len(xs)-n1)]]
    x2 = [xs[i] for i in index_permuted[(len(xs)-n1):]]
    return x1, x2

def modify_tag_seq(text, tag_seq, keyword, tagname):
    """
    Modify tag_seq in the same location of keyword in text by tagname
    """
    if is_empty_cell(keyword):
        return
    keyword = convert_to_orig(keyword)
    if keyword not in text or len(text) != len(tag_seq):
        return
    begin_locs = [loc.start() for loc in re.finditer(keyword, text)]
    for begin_loc in begin_locs:
        for loc in range(begin_loc, begin_loc + len(keyword)):
            if tag_seq[loc] != 'N':
                raise ValueError("Same char cannot bear more than one tag!")
            tag_seq[loc] = tagname
    

def is_empty_cell(x):
    return (not isinstance(x, str)) or len(x) == 0


def get_sent_len_for_pages(tag_seq_list, eos_tag):
    parsed_sent_len_for_pages = []
    for tag_seq in tag_seq_list:
        # make list of int (i.e. sentence lengths) out of list of tags
        parsed_sent_len = []
        current_len = 0
        for tag in tag_seq:
            current_len += 1
            if tag == eos_tag:
                parsed_sent_len.append(current_len)
                current_len = 0
        # in case last char is not tagged as 'S'
        if current_len > 0:
            parsed_sent_len.append(current_len)
        parsed_sent_len_for_pages.append(parsed_sent_len)
    return parsed_sent_len_for_pages


def get_keywords_from_tagged_record(char_samples, tag_name):
    res = []
    is_inside_keyword = False
    current_keyword = ""
    for cs in char_samples:
        if cs.get_tag() == tag_name:
            is_inside_keyword = True
            current_keyword += cs.get_char()
        else:
            if is_inside_keyword:       # First char sample after keyword
                res.append(current_keyword)
                current_keyword = ""
            is_inside_keyword = False
    # In case last keyword is by end of sentence
    if len(current_keyword) > 0:
        res.append(current_keyword)
    return res

def get_data_from_pages(samples, x_encoder, y_encoder):
    return [(p.get_x(x_encoder), p.get_y(y_encoder)) for p in samples]

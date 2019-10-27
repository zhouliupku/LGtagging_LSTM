# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 00:11:24 2019

@author: Zhou
"""

import re

# space char in original text in LoGaRT
SPACE = '○'

def convert_to_orig(s):
    """
    Convert string from database to corresponding original text
    """
    return s.replace(' ', SPACE)


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

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 00:11:24 2019

@author: Zhou
"""

import os
import re
import pickle
import numpy as np

import config

def convert_to_orig(s):
    """
    Convert string from database to corresponding original text
    """
    return s.replace(' ', config.PADDING_CHAR)


def random_separate(xs, percs):
    """
    given a list of objects xs, split it randomly into n+1 parts where n=len(percs)
    """
    ns = list(map(int, [p * len(xs) for p in percs]))
    index_permuted = np.random.permutation(len(xs))
    bs = np.clip(np.array(ns).cumsum(), 0, len(xs))
    bs = [0] + list(bs) + [len(xs)]
    return [[xs[i] for i in index_permuted[beg:end]] for beg, end in zip(bs[:-1], bs[1:])]


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
            if tag_seq[loc] != config.NULL_TAG:
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


def get_data_from_samples(samples, x_encoder, y_encoder):
    retv = []
    for i, p in enumerate(samples):
        if i % 1000 == 0:
            print(i)
        retv.append((p.get_x(x_encoder), p.get_y(y_encoder)))
    return retv
        
#    return [(p.get_x(x_encoder), p.get_y(y_encoder)) for p in samples]


def tag_correct_ratio(samples, model, subset_name, args, logger):
    '''
    Return entity-level correct ratio only for record model
    '''
    inputs = [(s.get_x(), s.get_y()) for s in samples]
    tags = model.evaluate_model(inputs, args)  
    tag_pred = [tag[0] for tag in tags]
    tag_true = [tag[1] for tag in tags]
    assert len(tag_pred) == len(tag_true)
    for x, y in zip(tag_pred, tag_true):
        assert len(x) == len(y)
    correct_and_total_counts = [word_count(ps, ts) for ps, ts in zip(tag_pred, tag_true)]
    entity_correct_ratio = sum([x[0] for x in correct_and_total_counts]) \
                            / float(sum([x[1] for x in correct_and_total_counts]))
    
    # Log info of correct ratio
    info_log = "Entity level correct ratio of {} set is {}".format(subset_name,
                                                              entity_correct_ratio)
    print(info_log)
    logger.info(info_log)
    
    return tag_correct_ratio
    
    
def word_count(ps, ts):
    """
    given two lists of tags, count matched words
    """
    pred_cuts = get_cut(ps)
    matches = [c for c in get_cut(ts) if c in pred_cuts]
    return len(matches), len(pred_cuts)
    
    
def get_cut(seq):
    # TODO: see if other papers handle BEG, END, null tag
    if len(seq) == 0:
        return []
    triplets = []
    start, last = 0, seq[0]
    for i, x in enumerate(seq):
        if x != last:
            triplets.append((start, i, last))
            start, last = i, x
    triplets.append((start, len(seq), last))
    return triplets   
    
    
def correct_ratio_calculation(samples, model, args, subset_name, logger):
    '''
    Take in samples (pages / records), input_encoder, model, output_encoder 
    Get the predict tags and return the correct ratio
    '''
    inputs = [(s.get_x(), s.get_y()) for s in samples]    
    tags = model.evaluate_model(inputs, args)   # list of (list of tags, list of tags)
    tag_pred = [tag[0] for tag in tags]
    tag_true = [tag[1] for tag in tags]
    assert len(tag_pred) == len(tag_true)
    for x, y in zip(tag_pred, tag_true):
        assert len(x) == len(y)
    if args.task_type == "page":    # only calculate the EOS tag for page model
        upstairs = [sum([p==t for p,t in zip(ps, ts) if t == config.EOS_TAG]) \
                              for ps, ts in zip(tag_pred, tag_true)]
        downstairs = [len([r for r in rs if r == config.EOS_TAG]) for rs in tag_true]
    else:       # ignore BEG, END etc for record model, although they are learned
        upstairs = [sum([p==t for p,t in zip(ps, ts) if t not in config.special_tag_list]) \
                    for ps, ts in zip(tag_pred, tag_true)]
        downstairs = [len(r) for r in tag_true if r not in config.special_tag_list]
    # There should be no empty page/record so no check for divide-by-zero needed here
    correct_ratio = sum(upstairs) / float(sum(downstairs))
    
    # Log info of correct ratio
    info_log = "Correct ratio of {} set is {}".format(subset_name, correct_ratio)
    print(info_log)
    logger.info(info_log)
    
    return correct_ratio

def tag_count(samples, model, subset_name, args):
    '''
    Take in samples (pages / records), input_encoder, model, output_encoder 
    Get the counts of each tags
    '''
    inputs = [(s.get_x(), s.get_y()) for s in samples]    
    tags = model.evaluate_model(inputs, args)   # list of (list of tags, list of tags)
    tag_pred = [tag[0] for tag in tags]    # list of list of tags
    tag_true = [tag[1] for tag in tags]
    assert len(tag_pred) == len(tag_true)
    for x, y in zip(tag_pred, tag_true):
        assert len(x) == len(y)
    correct_pairs = []
    for ps,ts in zip(tag_pred, tag_true):
        for p,t in zip(ps,ts):
            if p == t and t not in config.special_tag_list:
                correct_pairs.append((p,t))
    print("correct pairs number: ", len(correct_pairs))
    tag_set = set([item[0] for item in correct_pairs])           
    print("tag categories: ", len(tag_set))
    tag_statistics = []              
    for tag in tag_set:
        count = 0
        for item in correct_pairs:
            if item[1] == tag:
                count += 1
        tag_statistics.append((tag, count))
    for t,c in tag_statistics:
        print("For {} data, the number of tag {} : {}".format(subset_name, t, c))
    return tag_statistics

def load_data_from_pickle(filename, size):
    path = os.path.join(config.DATA_PATH, size)
    return pickle.load(open(os.path.join(path, filename), "rb"))


def get_filename_from_embed_type(embed_type):
    return os.path.join(config.EMBEDDING_PATH,
                        config.EMBEDDING_FILENAME_DICT[embed_type])

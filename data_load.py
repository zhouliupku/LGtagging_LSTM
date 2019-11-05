# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 22:59:19 2019

@author: Zhou
"""

import os
import numpy as np
import pandas as pd

from DataStructures import Page

class DataLoader(object):
    def __init__(self, source_type):
        self.source_type = source_type
        
    def load_data_xy(self, text_filename, tag_filename, interested_tag_tuples, mode):
        """
        return a list of Page instances
        """
        with open(text_filename, 'r', encoding="utf8") as txtfile:
            lines = txtfile.readlines()
        if mode == "train":
            df = pd.read_excel(tag_filename)
        elif mode == "test":
            df = None
        else:
            raise ValueError("Unsupported mode: {}".format(mode))
        pages = []
        for line in lines:
            if '【' not in line or '】' not in line:
                continue
            page = Page()
            page.fill_with_xy_file(line, df, mode, interested_tag_tuples)
            pages.append(page)
        return pages
    
    
    def load_data(self, cv_perc, interested_tag_tuples):
        # I/O setting
        NUM_SECTION_TAGGED, NUM_SECTION_UNTAGGED = 2, 1
        
        # TODO: derived classes instead of like this
        # Training set, depending on source type, use different loader
        if self.source_type == "html":
            pages_tagged = []
            INPUT_PATH = os.path.join(os.getcwd(), "logart_html", "train")
            tagged_filelist = [os.path.join(INPUT_PATH, x) for x in os.listdir(INPUT_PATH)]
            tagged_filelist = tagged_filelist[:NUM_SECTION_TAGGED]
            for html_filename in tagged_filelist:
                pages_tagged.extend(self.load_data_html(html_filename, 
                                                 interested_tag_tuples, "train"))
        elif self.source_type == "xy":
            pages_tagged = []
            INPUT_PATH = os.path.join(os.getcwd(), "LSTMdata")
            tagged_filelist = [(os.path.join(INPUT_PATH, "textTraining{}.txt".format(i)),
                                os.path.join(INPUT_PATH, "tagTraining{}.xlsx".format(i))) \
                                for i in range(NUM_SECTION_TAGGED)]
            for txt_filename, db_filename in tagged_filelist:
                pages_tagged.extend(self.load_data_xy(txt_filename, db_filename, 
                                                 interested_tag_tuples, "train"))
        else:
            raise ValueError("Unsupported source: {}".format(self.source_type))
        n_cv = int(cv_perc * len(pages_tagged))
        index_permuted = np.random.permutation(len(pages_tagged))
        pages_train = [pages_tagged[i] for i in index_permuted[:(len(pages_tagged)-n_cv)]]
        pages_cv = [pages_tagged[i] for i in index_permuted[(len(pages_tagged)-n_cv):]]
        print("Loaded {} pages for training.".format(len(pages_train)))
        print("Loaded {} pages for cross validation.".format(len(pages_cv)))
        
        # Test set
        pages_test = []
        untagged_filelist = [os.path.join(INPUT_PATH, "textTest{}.txt".format(i))
                            for i in range(NUM_SECTION_UNTAGGED)]
        for txt_filename in untagged_filelist:
            pages_test.extend(self.load_data_xy(txt_filename, None, 
                                           interested_tag_tuples, "test"))
        print("Loaded {} pages for testing.".format(len(pages_test)))
        return pages_train, pages_cv, pages_test
          
            
    

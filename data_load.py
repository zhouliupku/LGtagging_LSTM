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
    def __init__(self):
        self.datapath = None
        
    def load_data(self, interested_tag_tuples, mode, n, cv_perc=0.5):
        
        # TODO: derived classes instead of like this
        # Training set, depending on source type, use different loader
#        if self.source_type == "html":
#            pages = []
#            input_path = os.path.join(os.getcwd(), "logart_html", "train")
#            tagged_filelist = [os.path.join(input_path, x) for x in os.listdir(input_path)]
#            tagged_filelist = tagged_filelist[:NUM_SECTION_TAGGED]
#            for html_filename in tagged_filelist:
#                pages.extend(self.load_data_html(html_filename, 
#                                                 interested_tag_tuples, "train"))
        pages = []
        for file in self.get_file(mode, n):
            pages.extend(self.load_section(file, interested_tag_tuples, mode))
            
        if mode == "train":
            n_cv = int(cv_perc * len(pages))
            index_permuted = np.random.permutation(len(pages))
            pages_train = [pages[i] for i in index_permuted[:(len(pages)-n_cv)]]
            pages_cv = [pages[i] for i in index_permuted[(len(pages)-n_cv):]]
            print("Loaded {} pages for training.".format(len(pages_train)))
            print("Loaded {} pages for cross validation.".format(len(pages_cv)))
            return pages_train, pages_cv
        else:
            print("Loaded {} pages for testing.".format(len(pages)))
            return pages
        
    def load_section(self, files, interested_tag_tuples, mode):
        raise NotImplementedError
        
    def get_file(self, mode, n):
        raise NotImplementedError
        
        
class XYDataLoader(DataLoader):
    def __init__(self):
        super().__init__()
        self.datapath = os.path.join(os.getcwd(), "LSTMdata")
        
    def load_section(self, files, interested_tag_tuples, mode):
        """
        return a list of Page instances
        """
        text_filename, tag_filename = files[0], files[1]
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
            
    def get_file(self, mode, n):
        if mode == "train":
            return [(os.path.join(self.datapath, "train", "text{}.txt".format(i)),
                        os.path.join(self.datapath, "train", "tag{}.xlsx".format(i))) \
                        for i in range(n)]
        elif mode == "test":
            return [(os.path.join(self.datapath, "test", "text{}.txt".format(i)), None)
                            for i in range(n)]
        else:
            raise ValueError


class HtmlDataLoader(DataLoader):
    def __init__(self):
        super().__init__()
        self.datapath = os.path.join(os.getcwd(), "logart_html")
        
    def load_section(self, files, interested_tag_tuples, mode):
        """
        return a list of Page instances
        """
        html_filename = files
        # TODO: bs
        return pages    
            
    def get_file(self, mode, n):
        if mode == "train":
            input_path = os.path.join(self.datapath, "train")
            tagged_filelist = [os.path.join(input_path, x) for x in os.listdir(input_path)]
            return tagged_filelist[:n]
        elif mode == "test":
            raise NotImplementedError
        else:
            raise ValueError
    
    

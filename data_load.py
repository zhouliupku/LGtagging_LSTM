# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 22:59:19 2019

@author: Zhou
"""

import os
import numpy as np
import pandas as pd
import itertools
from bs4 import BeautifulSoup as BS

from DataStructures import Page

class DataLoader(object):
    def __init__(self):
        self.datapath = None
        
    def load_data(self, interested_tag_tuples, mode, n, cv_perc=0.5):
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
        if mode != "train":
            raise ValueError("Unsupported mode: {}".format(mode))
        html_filename = files
        with open(html_filename, 'r', encoding = "utf8") as file:
            lines = file.readlines()
            
        pages = []
        all_text, all_tags = self.format_raw_data(lines)
        rest_tags = all_tags # list of tag, together page
        page_texts = all_text.split('【')
        rest_tags = rest_tags[len(page_texts[0]):]
        for page_text in page_texts[1:]:
            candi = page_text.split('】')
            if len(candi) != 2:
                raise ValueError
            pid, txt = int(candi[0]), candi[1]
            tags = rest_tags[(len(candi[0]) + 2):(len(candi[0]) + 2 + len(txt))]
            rest_tags = rest_tags[(len(candi[0]) + 2 + len(txt)):]
            page = Page()
            page.fill_with_html_file(pid, txt, tags, mode, interested_tag_tuples)
            pages.append(page)
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
            
    def format_raw_data(self, lines):
        """
        return (all_text, all_tags) for a given section represented as list of str
        all_text: a long str representing whole text for the section
        all_tags: list of tags (each tag as a str) with same len as all_text
        """
        Xs, Ys = [], []
        for line in lines:
            soup = BS(line, "html.parser")
            with_tag_item = list(soup.find_all(recursive=False))
            
            result = []
            without_tag_item = []
            rest_contents = str(soup)
            for item in with_tag_item:            
                item_str = str(item)
                title_flag = '〉' in item.text or '〈' in item.text
                item_start_idx = rest_contents.find(item_str)
                if not title_flag:
                    # white space around non-tagged string
                    str_before_tag = rest_contents[:item_start_idx].strip()
                    without_tag_item.append(str_before_tag)
                    
                    # Step 1. if there is non-trivial string before tag, add as null tag
                    if len(str_before_tag) > 0:
                        null_tag = soup.new_tag("null")
                        null_tag.string = str_before_tag.replace(' ', '')
                        result.append(null_tag)
                    
                    # Step 2. add the tag itself to result, with modification w.r.t. spaces
                    modified_str = item.text
                    if item.name == "person":
                        # Special person formatting
                        if len(modified_str) == 2:
                            modified_str = modified_str[0] + '○' + modified_str[1]
                    # Remove white space in tagged strings
                    modified_str = modified_str.replace(' ', '')
                    item.string = modified_str
                    result.append(item)
                
                # Step 3. update rest_contents so that it contains the part after this tag
                rest_contents = rest_contents[(item_start_idx + len(item_str)):]
                
            # Lastly, if there is anything left, these should be with null tag. Add it
            # Keep end-of-line white spaces
            rest_contents = rest_contents.lstrip()
            rest_contents = rest_contents.rstrip() \
                            + '○'*(len(rest_contents) - len(rest_contents.rstrip()))
            rest_contents = rest_contents.replace(' ', '')
            if len(rest_contents) > 0:
                null_tag = soup.new_tag("null")
                null_tag.string = rest_contents
                result.append(null_tag)
                without_tag_item.append(rest_contents)
            X = ''.join([t.text for t in result])
            Y = list(itertools.chain.from_iterable([[t.name] * len(t.text) for t in result]))
            
            # hierarchy: section -> page -> record -> char
            Xs.append(X)        # Xs is list of str, each str: record
            Ys.append(Y)        # Ys is list of list of tag
        return ''.join(Xs), list(itertools.chain.from_iterable(Ys))
    
    

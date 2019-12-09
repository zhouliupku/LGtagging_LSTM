# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 22:59:19 2019

@author: Zhou
"""

import os
import pandas as pd
import pickle
import itertools
from bs4 import BeautifulSoup as BS

import lg_utils
from config import NULL_TAG, PADDING_CHAR
from DataStructures import Page, Record

class DataLoader(object):
    def __init__(self):
        self.datapath = None
        
    def load_data(self, interested_tags, mode, n,
                  train_perc=0.5, cv_perc=0.2):
        '''
        return a tuple containing lists of pages or records, depending on mode
        '''
        pages = []
        records = []
        for file in self.get_file(mode, n):
            try:
                ps, rs = self.load_section(file, interested_tags, mode)
                pages.extend(ps)
                records.extend(rs)
            except ValueError:
                print("VALUE ERROR!")
                print(file)
            
        if mode == "train":
            pages_train, pages_cv, pages_test = lg_utils.random_separate(pages, 
                                                             [train_perc, cv_perc])
            records_train, records_cv, records_test = lg_utils.random_separate(records,
                                                             [train_perc, cv_perc])
            print("Loaded {} pages for training.".format(len(pages_train)))
            print("Loaded {} pages for cross validation.".format(len(pages_cv)))
            print("Loaded {} pages for testing.".format(len(pages_test)))
            return pages_train, pages_cv, pages_test, records_train, records_cv, records_test
        else:
            print("Loaded {} pages for testing.".format(len(pages)))
            return pages, records
        
    def load_section(self, files, interested_tags, mode):
        raise NotImplementedError
        
    def get_file(self, mode, n=None):
        raise NotImplementedError
        
    def get_person_tag(self):
        raise NotImplementedError
        
        
class XYDataLoader(DataLoader):
    def __init__(self):
        super().__init__()
        self.datapath = os.path.join(os.getcwd(), "LSTMdata")
        
    def load_section(self, files, interested_tags, mode):
        """
        files are passed in by looping a list of tuples, tuples[0] is the
        original text, tuples[1] is the excel with tag message
        return a list of Page instances and a list of Record instances
        """
        text_filename, tag_filename = files[0], files[1]
        with open(text_filename, 'r', encoding="utf8") as txtfile:
            lines = txtfile.readlines()
        if mode == "train":
            df = pd.read_excel(tag_filename)
        elif mode == "produce":
            df = None
        else:
            raise ValueError("Unsupported mode: {}".format(mode))
        pages = []
        records = []
        for line in lines:      # each line is a page
            if '【' not in line or '】' not in line:
                continue
            line = line.split('【')[1].strip().split('】')
            pid = int(line[0])
            txt = line[1]
            eos_idx = []
            if mode == "train":
                subdf = df[df["Page No."] == pid]       #every page is a subdf
                last_cutoff = 0
                record_txts = []
                record_dfs = []
                for _, row in subdf.iterrows():
                    name = lg_utils.convert_to_orig(row[self.get_person_tag()])  #convert space with "○"
                    if not name in txt:
                        raise ValueError("Name {} not in original text!".format(name))
                    cutoff = txt.find(name)       #name is the only beginning of a record
                    if cutoff > 0:
                        eos_idx.append(cutoff - 1)   
                    if last_cutoff > 0:
                        record_txts.append(txt[last_cutoff:cutoff])
                    record_dfs.append(row)   
                    last_cutoff = cutoff
                cutoff = len(txt) - 1      #the end of the txt's index
                eos_idx.append(cutoff - 1)
                record_txts.append(txt[last_cutoff:cutoff])   #record_txts: a list of str
                        
                if interested_tags is None:
                    interested_tags_used = record_dfs.columns
                else:
                    interested_tags_used = interested_tags
                for record_txt, row in zip(record_txts, record_dfs):
                    # Build tag sequence
                    tags = [NULL_TAG for _ in record_txt]
                    for colname in interested_tags_used:
                        lg_utils.modify_tag_seq(record_txt, tags, row[colname], colname)
                        r = Record(record_txt, tags)
                    records.append(r)
    
            elif mode == "produce":
                pass
            else:
                raise ValueError("Unsupported mode:" + str(mode))
            pages.append(Page(pid, txt, eos_idx))
        return pages, records
            
    def get_file(self, mode, n=None):
        if n is None:
            raise ValueError("Loader does not support unspecified size")
        if mode == "train":
            return [(os.path.join(self.datapath, "train", "text{}.txt".format(i)),
                        os.path.join(self.datapath, "train", "tag{}.xlsx".format(i))) \
                        for i in range(n)]
        elif mode == "produce":
            return [(os.path.join(self.datapath, "test", "text{}.txt".format(i)), None)
                            for i in range(n)]
        else:
            raise ValueError("Unsupported mode!")
            
    def get_person_tag(self):
        return "人名"


class HtmlDataLoader(DataLoader):
    def __init__(self, size="small"):
        '''
        size: "small" or "full"
        '''
        super().__init__()
        self.datapath = os.path.join(os.getcwd(), "logart_html", size)
        
    def load_section(self, files, interested_tags, mode):
        """
        return a list of Page instances and Record instances
        """
        if mode != "train":      #only train data needs this loader, test data is the same with XYDataloader
            raise ValueError("Unsupported mode: {}".format(mode))
        html_filename = files
        with open(html_filename, 'r', encoding = "utf8") as file:
            lines = file.readlines()
            
        pages = []
        records = []
        all_text, all_tags = self.format_raw_data(lines)
        rest_tags = all_tags # list of tag, together page
        page_texts = all_text.split('【')   # list of str, except the first str, all other str starts with "】"
        rest_tags = rest_tags[len(page_texts[0]):]    
        for page_text in page_texts[1:]:
            
            candi = page_text.split('】')
            if len(candi) != 2:
                raise ValueError
            pid, txt = int(candi[0]), candi[1]
            page_tags = rest_tags[(len(candi[0]) + 2):(len(candi[0]) + 2 + len(txt))]
            rest_tags = rest_tags[(len(candi[0]) + 2 + len(txt)):]
            
            eos_idx = []
            
            for i, tag in enumerate(page_tags):
                # EOS is the index just before a name' beginning
                if i == 0 and tag == self.get_person_tag():
                    continue
                if tag == self.get_person_tag() and page_tags[i-1] != self.get_person_tag():
                    eos_idx.append(i-1)
            eos_idx.append(len(txt)-1)
            page = Page(pid, txt, eos_idx)
            record_txt_len = page.get_sep_len()
            
            head_char_idx = 0
            records_in_page = []
            for sent_len in record_txt_len:
                text = page.txt[head_char_idx : (head_char_idx + sent_len)]
                tags = page_tags[head_char_idx : (head_char_idx + sent_len)]
                # substitution
                record_tag = []
                for tag in tags:
                    if (interested_tags is None) or (tag in interested_tags):
                        record_tag.append(tag)          # None means take all
                    else:
                        record_tag.append(NULL_TAG)
                records_in_page.append(Record(text, record_tag))
                head_char_idx += sent_len
            records.extend(records_in_page)
            pages.append(page)
        return pages, records
            
    def get_file(self, mode, n=None):
        if mode == "train":
            input_path = os.path.join(self.datapath, "train")
            tagged_filelist = [os.path.join(input_path, x) for x in os.listdir(input_path)]
            if n is None:
                return tagged_filelist
            else:
                return tagged_filelist[:n]
        elif mode == "produce":
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
                        null_tag = soup.new_tag(NULL_TAG)
                        null_tag.string = str_before_tag.replace(' ', '')
                        result.append(null_tag)
                    
                    # Step 2. add the tag itself to result, with modification w.r.t. spaces
                    modified_str = item.text
                    if item.name == self.get_person_tag():
                        # Special person formatting
                        if len(modified_str) == 2:
                            modified_str = modified_str[0] + PADDING_CHAR + modified_str[1]
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
                            + PADDING_CHAR*(len(rest_contents) - len(rest_contents.rstrip()))
            rest_contents = rest_contents.replace(' ', '')
            if len(rest_contents) > 0:
                null_tag = soup.new_tag(NULL_TAG)
                null_tag.string = rest_contents
                result.append(null_tag)
                without_tag_item.append(rest_contents)
            X = ''.join([t.text for t in result])
            Y = list(itertools.chain.from_iterable([[t.name] * len(t.text) for t in result]))
            
            # hierarchy: section -> page -> record -> char
            Xs.append(X)        # Xs is list of str, each str: record
            Ys.append(Y)        # Ys is list of list of tag
        return ''.join(Xs), list(itertools.chain.from_iterable(Ys))
    
    def get_person_tag(self):
        return "person"


def dump_data_to_pickle(d, filename, size):
    path = os.path.join(os.getcwd(), "data", size)
    pickle.dump(d, open(os.path.join(path, filename), "wb"))


if __name__ == "__main__":    
    N_SECTION = 1
    MODE = "produce"
    SOURCE_TYPE = "XY"
    SIZE = "tiny"
    # Set up data loaders
    if SOURCE_TYPE == "XY":
        loader = XYDataLoader()
    elif SOURCE_TYPE == "html":
        loader = HtmlDataLoader(SIZE)
    else:
        raise ValueError
    
    # Model hyper-parameter definition
    interested_tags = [loader.get_person_tag()]
    if SOURCE_TYPE == "XY":
        interested_tags.extend(["任職時間"])
#    interested_tags = ["人名", "任職時間", "籍貫", "入仕方法"]
    elif SOURCE_TYPE == "html":
        interested_tags.extend(["post_time", "office", "jiguan"])
    
    # We want to load all tags if it's in full mode, by setting interested_tags
    # to be None
    if SIZE == "full":
        interested_tags = None
    
    data = loader.load_data(interested_tags, MODE, N_SECTION)
    
    if MODE == "train":
        dump_data_to_pickle(data[0], "pages_train.p", SIZE)
        dump_data_to_pickle(data[1], "pages_cv.p", SIZE)
        dump_data_to_pickle(data[2], "pages_test.p", SIZE)
        dump_data_to_pickle(data[3], "records_train.p", SIZE)
        dump_data_to_pickle(data[4], "records_cv.p", SIZE)
        dump_data_to_pickle(data[5], "records_test.p", SIZE)
    else:
        dump_data_to_pickle(data[0], "pages_produce.p", SIZE)
        
    
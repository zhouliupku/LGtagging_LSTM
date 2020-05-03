# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 22:59:19 2019

Load from unstructured data and produce datasets

@author: Zhou
"""

import os
import pickle
import numpy as np
from bs4 import BeautifulSoup as BS

import lg_utils
import config
from config import NULL_TAG, PADDING_CHAR, MAX_LEN
from DataStructures import Page, Record

class DataLoader(object):
        
    def __init__(self, size="small"):
        '''
        size: "small" or "full"
        '''
        self.datapath = os.path.join(os.getcwd(), "logart_html", size)
        
    def load_data(self, interested_tags,
                  train_perc=0.6, cv_perc=0.2):
        '''
        return a tuple containing lists of pages or records, depending on mode
        '''
        pages = []
        records = []
        for file in self.get_file():
            try:
                ps, rs = self.load_section(file, interested_tags)
                pages.extend(ps)
                records.extend(rs)
            except ValueError:
                print("VALUE ERROR!")
                print(file)
            
        pages_train, pages_cv, pages_test = lg_utils.random_separate(pages, 
                                                         [train_perc, cv_perc])
        records_train, records_cv, records_test = lg_utils.random_separate(records,
                                                         [train_perc, cv_perc])
        print("Loaded {} pages for training.".format(len(pages_train)))
        print("Loaded {} pages for cross validation.".format(len(pages_cv)))
        print("Loaded {} pages for testing.".format(len(pages_test)))
        
        return pages_train, pages_cv, pages_test, records_train, records_cv, records_test
       
        
    def load_section(self, files, interested_tags):
        """
        return a list of Page instances and Record instances
        """
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
            if len(candi[1]) == 0:
#                print("Page {} is empty!".format(candi[0]))
                continue
            if len(candi[1]) >= MAX_LEN:
                print("Page {} is too long!".format(candi[0]))
                continue
            pid, txt = int(candi[0]), candi[1]
            page_tags = rest_tags[(len(candi[0]) + 2):(len(candi[0]) + 2 + len(txt))]
            rest_tags = rest_tags[(len(candi[0]) + 2 + len(txt)):]
            
            eos_idx = []
            
            for i, tag in enumerate(page_tags):
                # EOS is the index just before a name' beginning
                if i == 0 and tag == self.get_person_begin_tag():
                    continue
                if tag == self.get_person_begin_tag() \
                  and not self.is_person_tag(page_tags[i-1]):
                    # avoid consecutive person names
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
                
                for tag in tags:          # None means take all
                    if (interested_tags is None) or (tag in interested_tags):
                        record_tag.append(tag)
                    else:
                        record_tag.append(NULL_TAG)
                records_in_page.append(Record(text, record_tag))
                head_char_idx += sent_len
            records.extend(records_in_page)
            pages.append(page)
        return pages, records
            
    def get_file(self):
        input_path = os.path.join(self.datapath, "train")
        tagged_filelist = [os.path.join(input_path, x) for x in os.listdir(input_path)]
        return tagged_filelist
            
    def format_raw_data(self, lines):
        """
        return (all_text, all_tags) for a given section represented as list of str
        all_text: a long str representing whole text for the section
        all_tags: list of tags (each tag as a str) with same len as all_text
        """
        Xs, Ys = [], []
        for line in lines:
            line = line.replace(PADDING_CHAR, '')
#            line = line.replace(' ', '')
#            line = line.replace('[T4]', '')
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
                    item.string = item.text.replace(' ', '')
                    result.append(item)
                
                # Step 3. update rest_contents so that it contains the part after this tag
                rest_contents = rest_contents[(item_start_idx + len(item_str)):]
                
            # Lastly, if there is anything left, these should be with null tag. Add it
            rest_contents = rest_contents.strip().replace(' ', '')
            if len(rest_contents) > 0:
                null_tag = soup.new_tag(NULL_TAG)
                null_tag.string = rest_contents
                result.append(null_tag)
                without_tag_item.append(rest_contents)
            X = ''.join([t.text for t in result])
            Y = lg_utils.concat_lists([self.get_bio(t) for t in result])
            
            # hierarchy: section -> page -> record -> char
            Xs.append(X)        # Xs is list of str, each str: record
            Ys.append(Y)        # Ys is list of list of tag
        return ''.join(Xs), lg_utils.concat_lists(Ys)
    
    
    def get_bio(self, t):
        """
        input t: BS tag instance
        output: BIO style tags as list of string
        special treatment: due to historical reasons, entry_addr should be biog_addr
        """
        if len(t.text) == 0:
            return []
        elif t.name == NULL_TAG:
            return [NULL_TAG] * len(t.text)
        else:
            tag_name = "biog_addr" if t.name == "entry_addr" else t.name
            return [config.BEG_PREFIX + tag_name] + [config.IN_PREFIX + tag_name] * (len(t.text) - 1)
    
    
    def get_person_begin_tag(self):
        return config.BEG_PREFIX + "person"
    
    
    def is_person_tag(self, tag):
        return tag in [config.BEG_PREFIX + "person", config.IN_PREFIX + "person"]


def dump_data_to_pickle(d, filename, size):
    path = os.path.join(os.getcwd(), "data", size)
    if not os.path.exists(path):
        os.mkdir(path)
    pickle.dump(d, open(os.path.join(path, filename), "wb"))


if __name__ == "__main__":
    np.random.seed(0)
    for size in ["small", "medium", "full"]:
        loader = DataLoader(size)
        
        # Model hyper-parameter definition
        interested_tags = ["person", "post_time", "jiguan", "entry_way",
                           "post_type", "office", "entry_addr", "next_office",
                           "prev_office", "zi", "kins", "entry_time", "post_address",
                           "source_tag", "othername", "hao", "biog_addr"]
        interested_tags = lg_utils.concat_lists([[config.BEG_PREFIX + t, config.IN_PREFIX + t] for t in interested_tags])
        
        data = loader.load_data(interested_tags, train_perc=0.6, cv_perc=0.2)
        dump_data_to_pickle(data[0], "pages_train.p", size)
        dump_data_to_pickle(data[1], "pages_cv.p", size)
        dump_data_to_pickle(data[2], "pages_test.p", size)
        dump_data_to_pickle(data[3], "records_train.p", size)
        dump_data_to_pickle(data[4], "records_cv.p", size)
        dump_data_to_pickle(data[5], "records_test.p", size)
        
        
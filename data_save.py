# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 22:31:47 2019

@author: Zhou
"""

import pandas as pd
from bs4 import BeautifulSoup as BS

import config
import lg_utils

class DataSaver(object):
    def __init__(self, records):
        self.records = records
        
    def save(self, filename, interested_tags):
        raise NotImplementedError
        

class ExcelSaver(DataSaver):
    def save(self, filename, interested_tags):
        tagged_result = pd.DataFrame(columns=interested_tags)
        for record in self.records:
            res = record.get_tag_res_dict(interested_tags)
            res["yuanwen"] = str(record)
            tagged_result = tagged_result.append(res, ignore_index=True)
        tagged_result.to_excel(filename, index=False)


class HtmlSaver(DataSaver):
    def __init__(self, records):
        super(HtmlSaver, self).__init__(records)
        self.soup = BS("", "html.parser")
        
    def save(self, filename, interested_tags):
        with open(filename, 'w+', encoding="utf8") as f:
            for record in self.records:
                html_record = ""
                chunks = lg_utils.get_chunk([c.get_tag() for c in record.chars[1:-1]])
                chunks = [c for c in chunks if c[2] != config.NULL_TAG]
                txt = ''.join([c.get_char() for c in record.chars[1:-1]])
                last_char_pos = 0
                for chunk in chunks:
                    html_record += txt[last_char_pos : chunk[0]]
                    html_record += self.build_html_str(txt[chunk[0]:chunk[1]], chunk[2])
                    last_char_pos = chunk[1]
                html_record += txt[last_char_pos:] + '\n'
                f.write(html_record)

    def build_html_str(self, txt, tag):
        item = self.soup.new_tag(tag)
        item.string = txt
        return str(item)
        

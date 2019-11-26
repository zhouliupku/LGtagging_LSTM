# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 22:31:47 2019

@author: Zhou
"""

import pandas as pd
from bs4 import BeautifulSoup as BS

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
    # TODO: unify tag name in html
    def __init__(self, records):
        super(HtmlSaver, self).__init__(records)
        self.soup = BS("", "html.parser")
        
    def save(self, filename, interested_tags):
        with open(filename, 'w+', encoding="utf8") as f:
            for record in self.records:
                last_tag = None
                current_txt = ""
                html_record = ""
                for char, tag in [(c.get_char(), c.get_tag()) for c in record.chars[1:-1]]:
                    if last_tag is not None and tag != last_tag:
                        html_record += self.build_html_str(current_txt, last_tag)
                        current_txt = ""
                    current_txt += char
                    last_tag = tag
                html_record += self.build_html_str(current_txt, last_tag)
                html_record += '\n'
                f.write(html_record)

    def build_html_str(self, txt, tag):
        item = self.soup.new_tag(tag)
        item.string = txt
        return str(item)
        

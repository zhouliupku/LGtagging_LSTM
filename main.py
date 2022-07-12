# -*- coding: utf-8 -*-
"""
Create Time: 12/16/2021 5:39 PM
Author: Zhou
"""
import numpy as np
import logging
import datetime
import os
import torch
from argparse import Namespace

import config
from model import ModelFactory
from data_save import HtmlSaver
from DataStructures import Page
import lg_utils

DEFAULT_ARGS = Namespace(batch_size=4, bidirectional=True, data_size='full', extra_encoder=None, hidden_dim=64,
                         learning_rate=0.01, lstm_layer=2, main_encoder='BERT', model_alias='bert',
                         model_type='LSTMCRF', n_epoch=20, need_train=False, optimizer='Adam', process_type='produce',
                         regex=False, saver_type='html', start_from_epoch=-1, task_type='record', use_cuda=True)
DEFAULT_FILETYPES = (("text files", "*.txt"), ("all files", "*.*"))


def load():
    filename = "data/input.txt"
    with open(filename, "r", encoding='utf-8') as infile:
        input_pages = infile.readlines()
        id_page = []
        for page in input_pages:
            page_id = page.split("】")[0].strip("\ufeff").strip("【").strip()  # TODO: no hard code
            page_text = page.split("】")[1].strip()
            id_page.append([page_id, page_text])
    return id_page


def run():
    pages = load()  # list of list
    for page in pages:
        process_tagging(page[0], page[1])


def process_tagging(text_id, text):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(filename=os.path.join("log", "run{}.log".format(curr_time)),
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Step 1. using page_to_sent_model, parse pages to sentences
    # TODO: if too long, cut it short
    pages_produce = [Page(0, text, [])]
    vars(DEFAULT_ARGS)["task_type"] = "page"
    page_model = ModelFactory().get_trained_model(logger, DEFAULT_ARGS)
    pages_data = [p.get_x() for p in pages_produce]
    tag_seq_list = page_model.evaluate_model(pages_data, DEFAULT_ARGS)  # list of list of tag

    #   Step 3. using trained record model, tag each sentence
    vars(DEFAULT_ARGS)["task_type"] = "record"
    record_model = ModelFactory().get_trained_model(logger, DEFAULT_ARGS)
    record_test_data = []  # list of list of str
    records = []  # list of Record

    for p, pl in zip(pages_produce,
                     lg_utils.get_sent_len_for_pages(tag_seq_list, config.EOS_TAG)):
        rs = p.separate_sentence(pl)  # get a list of Record instances
        records.extend(rs)
        record_test_data.extend([r.get_x() for r in rs])

    # Use trained model to process
    tagged_sent = record_model.evaluate_model(record_test_data, DEFAULT_ARGS)
    for record, tag_list in zip(records, tagged_sent):
        record.set_tag(tag_list)
    saver = HtmlSaver(records)
    output = ""
    for record in saver.records:
        output += saver.make_record(record)
    # write output
    with open("result-test.txt", mode="a", encoding="utf-8") as outfile:
        outfile.write("【" + text_id + "】" + "\n")
        outfile.writelines(output)


if __name__ == "__main__":
    run()

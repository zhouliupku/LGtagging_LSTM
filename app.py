# -*- coding: utf-8 -*-

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

import tkinter as tk
from tkinter import filedialog as FD
from tkinter import messagebox as MSG
from tkinter import scrolledtext

WINDOW_SIZE = "1000x600"
DEFAULT_ARGS = Namespace(batch_size=4, bidirectional=True, data_size='full', extra_encoder=None, hidden_dim=64, learning_rate=0.01, lstm_layer=2, main_encoder='BERT', model_alias='bert', model_type='LSTMCRF', n_epoch=20, need_train=False, optimizer='Adam', process_type='produce', regex=False, saver_type='html', start_from_epoch=-1, task_type='record', use_cuda=True)
DEFAULT_FILETYPES = (("text files","*.txt"), ("all files","*.*"))

class App:
    def __init__(self, window, window_title):
        """
        Initialize app window and start main loop
        """
        self.root_path = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.root_path, "models")
        
        # Initialize app window
        self.window = window
        self.window.title(window_title)
        self.window.geometry(WINDOW_SIZE)
        
        # Widget setup
        self.load_button = tk.Button(text="Load from file", bg="white",
                                command=lambda:self.load())
        self.load_button.grid(column = 0, row = 0)
        self.process_button = tk.Button(text="Process Tagging", bg="white",
                                command=lambda:self.process_tagging())
        self.process_button.grid(column = 0, row = 1)
        self.export_button = tk.Button(text="Export to file", bg="white",
                                command=lambda:self.export())
        self.export_button.grid(column = 0, row = 2)
        
        self.orig_txt_label = tk.Label(window, text="Enter Your Text")
        self.orig_txt_label.grid(column = 1, row = 0)
        self.orig_txt_box = scrolledtext.ScrolledText(window, width=50, bd=1)
        self.orig_txt_box.grid(column = 1, row = 1)
        
        self.convert_txt_label = tk.Label(window, text="Tagged Text")
        self.convert_txt_label.grid(column = 2, row = 0)
        self.convert_txt_box = scrolledtext.ScrolledText(window, width=50, bd=1)
        self.convert_txt_box.grid(column = 2, row = 1)
        
        # Start main loop
        self.window.mainloop()
        
        
    def load(self):
        filename = FD.askopenfilename(initialdir=os.path.join(self.root_path),
                                     title="Load",
                                     filetypes=DEFAULT_FILETYPES)
        if filename == "":      # on cancel
            return
        with open(filename, "r", encoding='utf-8') as f:
            input_txt = f.readlines()
        self.show(self.orig_txt_box, input_txt) 
        
        
    def export(self):
        text = str(self.convert_txt_box.get("1.0", tk.END))
        filename = FD.asksaveasfilename(initialdir=os.path.join(self.root_path),
                                         title="save",
                                         filetypes=DEFAULT_FILETYPES,
                                         defaultextension=DEFAULT_FILETYPES)
        if filename == "":      # on cancel
            return
        with open(filename, "w", encoding='utf-8') as f:
            f.write(text)
        
        
    def process_tagging(self):
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
        #TODO: if too long, cut it short
        text = self.orig_txt_box.get("1.0", tk.END)
        pages_produce = [Page(0, text, [])]
        vars(DEFAULT_ARGS)["task_type"] = "page"
        page_model = ModelFactory().get_trained_model(logger, DEFAULT_ARGS)
        pages_data = [p.get_x() for p in pages_produce]
        tag_seq_list = page_model.evaluate_model(pages_data, DEFAULT_ARGS) # list of list of tag
                
        #   Step 3. using trained record model, tag each sentence
        vars(DEFAULT_ARGS)["task_type"] = "record"
        record_model = ModelFactory().get_trained_model(logger, DEFAULT_ARGS)
        record_test_data = []       # list of list of str
        records = []                # list of Record
        
        for p, pl in zip(pages_produce, 
                         lg_utils.get_sent_len_for_pages(tag_seq_list, config.EOS_TAG)):
            rs = p.separate_sentence(pl)        # get a list of Record instances
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
        self.show(self.convert_txt_box, output)    
        
            
    def show(self, widget, text):
        widget.delete('1.0', tk.END)
        widget.insert(tk.INSERT, text)
        widget.update_idletasks()
            
        
def run():
    """
    Create a window and pass it to the Application object
    """
    App(tk.Tk(), "LoGart Tagger")
        

if __name__ == "__main__":
    run()
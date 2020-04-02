# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 22:56:23 2019

@author: Zhou
"""

import os

# For pre-processing
NULL_TAG = "null"
PADDING_CHAR = '○'

# For x encoders
BEG_CHAR = "<S>"
END_CHAR = "</S>"
PAD_CHAR = "PAD"
UNK_CHAR = "<UNK>"
special_char_list = [BEG_CHAR, END_CHAR, PAD_CHAR, UNK_CHAR]

# For y encoders
INS_TAG = 'N'
EOS_TAG = 'S'
BEG_TAG = "<BEG>"
END_TAG = "<END>"
PAD_TAG = "<PAD>"
special_tag_list = [BEG_TAG, END_TAG, PAD_TAG]

ROOT_PATH = os.path.dirname(__file__)
OUTPUT_PATH = os.path.join(ROOT_PATH, "result")
REGEX_PATH = os.path.join(ROOT_PATH, "models")
DATA_PATH = os.path.join(ROOT_PATH, "data")
EMBEDDING_PATH = os.path.join(ROOT_PATH, "Embedding")
EMBEDDING_FILENAME_DICT = {"polyglot": "polyglot-zh_char.pkl",
                           "MCP": "MCP.p",
                           "Xingpang": "Xingpang.p",
                           "shiyi": "shiyi.p",
                           "shiyi_folded": "shiyi_folded.p"}
PLOT_PATH = os.path.join(ROOT_PATH, "plot")

BERT_DIM = 768
MAX_LEN = 512

MAX_ENTITY_RECORD = 1E7

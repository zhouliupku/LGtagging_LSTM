# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 22:53:36 2020

@author: Zhou
"""

import os
import numpy as np
import pickle

import config

if __name__ == "__main__":
    FOLD = 8
    
    chars, shiyi_orig = pickle.load(open(os.path.join(config.EMBEDDING_PATH, "shiyi.p"), "rb"))
    char_dim, shiyi_dim = shiyi_orig.shape
    folded_emb = np.mean(shiyi_orig.reshape(char_dim, shiyi_dim // FOLD, FOLD), axis=2)
    pickle.dump((chars, folded_emb),
                open(os.path.join(config.EMBEDDING_PATH, "shiyi_folded.p"), "wb"))
        
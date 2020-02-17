# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 21:01:11 2020

@author: Zhou
"""

import os
import pandas as pd
import numpy as np
import itertools
import pickle

import config

def gen_one_hot(col):
    """
    Return list
    """
    col_values = list(set(col))
    word_list_one_hot = []
    for item in col:
        base = [0 for _ in col_values]
        base[col_values.index(item)] = 1
        word_list_one_hot.append(base)
    return word_list_one_hot

if __name__ == "__main__":
    df = pd.read_excel(os.path.join(config.EMBEDDING_PATH, "guangyun.xlsx"),
                           encoding='utf-8')
    feature_names = ["聲紐", "韻部原貌-平上去入相配爲平(調整前)", "呼", "等", "聲調"]
    chars = df["廣韻字頭原貌(覈校前)"].values
    
    embs = [gen_one_hot(df[f]) for f in feature_names]
    for emb in embs:
        assert len(emb) == len(chars)
    
    final_emb = np.array([list(itertools.chain.from_iterable(z)) for z in zip(*embs)])
    for sc in config.special_char_list:
        chars = np.append(chars, sc)
        final_emb = np.concatenate((final_emb,
                                    np.array([[0 for _ in range(final_emb.shape[1])]])),
                    axis=0)
    
    pickle.dump((tuple(chars), final_emb),
                open(os.path.join(config.EMBEDDING_PATH, "MCP.p"), "wb"))

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 21:01:11 2020

@author: Zhou
"""

import os
import torch
import pandas as pd
import numpy as np
import itertools
import pickle
from pytorch_pretrained_bert import BertTokenizer, BertModel

import config
from Encoders import BertEncoder

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

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)



if __name__ == "__main__":
    df = pd.read_excel(os.path.join(config.EMBEDDING_PATH, "guangyun.xlsx"),
                           encoding='utf-8')
    
    
#    output_name = "Xingpang"
#    feature_names = ["聲形-形"]
    
#    output_name = "MCP"
#    feature_names = ["聲紐", "韻部原貌-平上去入相配爲平(調整前)", "呼", "等", "聲調"]
    
#    chars = df["廣韻字頭原貌(覈校前)"].values
    
    # Extra preprocessing for xing pang
#    if "聲形-形" in feature_names:
#        df = df.assign(tmp=df.apply(lambda row: row["聲形-形"] if str(row["字類"]) == "nan" else "", axis=1))
#        df.drop(columns=["聲形-形"], axis=1, inplace=True)
#        df.rename(columns={"tmp": "聲形-形"}, inplace=True)
#        new_cols = {"聲形-形": lambda row: row["聲形-形"] if row["字類"] == "" else ""}
#        df = df.assign(**new_cols)
    
#    embs = [gen_one_hot(df[f]) for f in feature_names]
#    for emb in embs:
#        assert len(emb) == len(chars)
#    
#    final_emb = np.array([list(itertools.chain.from_iterable(z)) for z in zip(*embs)])
#    for sc in config.special_char_list:
#        chars = np.append(chars, sc)
#        final_emb = np.concatenate((final_emb,
#                                    np.array([[0 for _ in range(final_emb.shape[1])]])),
#                    axis=0)
    
#    pickle.dump((tuple(chars), final_emb),
#                open(os.path.join(config.EMBEDDING_PATH, "{}.p".format(output_name)), "wb"))
   

    shiyi = [x if type(x) == str else "[UNK]" for x in list(df["廣韻釋義"])]
    
    for i, item in enumerate(shiyi):
        if item == "上同":
            shiyi[i] = shiyi[i-1]
            
    args = Namespace(use_cuda=True)
    encoder = BertEncoder(args)
    embs = []
    for item in shiyi:
        item = item[:config.MAX_LEN - 3]
        sentence = [config.BEG_CHAR] + list(item) + [config.END_CHAR]
        emb = encoder.encode(sentence)
        emb = torch.mean(emb, dim=0)
        embs.append(emb.cpu().numpy())
        
    chars = df["廣韻字頭原貌(覈校前)"].values   
    final_emb = np.array(embs)
    for sc in config.special_char_list:
        chars = np.append(chars, sc)
        final_emb = np.concatenate((final_emb,
                                    np.array([[0 for _ in range(final_emb.shape[1])]])),
                    axis=0)
    
    # dim of final_emb: n_char x n_dim
    pickle.dump((tuple(chars), final_emb),
                open(os.path.join(config.EMBEDDING_PATH, "{}.p".format("shiyi")), "wb"))
        
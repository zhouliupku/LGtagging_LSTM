# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 21:49:10 2020

@author: Zhou
"""

import os
import pandas as pd

import config


if __name__ == "__main__":
    df_base = pd.read_csv(os.path.join(config.OUTPUT_PATH, "entity_detail_base.csv"))
    df_impr = pd.read_csv(os.path.join(config.OUTPUT_PATH, "entity_detail_impr.csv"))
    
    df_base.rename(columns={"pred": "pred_b", "correct": "correct_b"}, inplace=True)
    df_impr.rename(columns={"pred": "pred_i", "correct": "correct_i"}, inplace=True)
    
    df = pd.concat([df_base, df_impr[["pred_i", "correct_i"]]], axis=1) 
    df.to_csv(os.path.join(config.OUTPUT_PATH, "entity_detail_combined.csv"),
              encoding='utf-8-sig',
              index=False)
    
    
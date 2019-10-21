# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 00:11:24 2019

@author: Zhou
"""

# space char in original text in LoGaRT
SPACE = '○'

def convert_to_orig(s):
    """
    Convert string from database to corresponding original text
    """
    return s.replace(' ', SPACE)

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 22:43:13 2019

@author: Zhou
"""

import os
import torch
import numpy as np
import datetime
import logging
import argparse

import process


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--data_size', type=str, default='tiny',
                    choices=['tiny', 'small', 'medium', 'full'],
                    help='Size of training data')
parser.add_argument('--saver_type', type=str, default='html', 
                    choices=['html', 'excel'],
                    help='Type of saver')
parser.add_argument('--task_type', type=str, default='page', 
                    choices=['page', 'record'],
                    help='Type of task')
parser.add_argument('--model_type', type=str, default='LSTM', 
                    choices=['LSTM', 'TwoLayerLSTM', 'LSTMCRF'],
                    help='Type of model')
parser.add_argument('--model_alias', type=str, default='unnamed_model', 
                    help='Alias to specify variants of same type of model')
parser.add_argument('--process_type', type=str, default='train', 
                    choices=['train', 'test', 'produce'],
                    help='Type of process')
parser.add_argument('--loss_type', type=str, default='NLL', 
                    choices=['NLL'],
                    help='Type of loss function')
parser.add_argument('--n_epoch', type=int, default=50, 
                    help='Number of epoch')
parser.add_argument('--learning_rate', type=float, default=0.05, 
                    help='Learning rate')
parser.add_argument('--hidden_dim', type=int, default=6,
                    help='Hidden dim')
parser.add_argument('--bidirectional', type=str2bool, default=False, 
                    help='Boolean indicating whether bidirectional is enabled')
parser.add_argument('--regex', type=str2bool, default=True, 
                    help='Whether to use RegEx in producing')
parser.add_argument('--need_train', type=str2bool, default=True, 
                    help='Boolean indicating whether need training model')
args = parser.parse_args()


if __name__ == "__main__":
    print("\nParameters:")
    for attr, value in args.__dict__.items():
        print("\t{} = {}".format(attr.upper(), value))
        
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Logging
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(filename=os.path.join("log", "run{}.log".format(curr_time)),
                        format='%(asctime)s %(message)s', 
                        filemode='w') 
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info("Started training at {}".format(curr_time))
    
    if args.process_type == "train":
        process.train(logger, args)
    elif args.process_type == "test":
        process.test(logger, args)
    else:
        process.produce(logger, args)

# LGtagging_LSTM
## Introduction
A tagging system of local gazetteers by LSTM algorithms

## Usage

To train, use --process_type train, e.g.

    python main.py --process_type train --task_type record --n_epoch 100 --start_from_epoch -1 --learning_rate 0.000001 --need_train T --model_alias bert_1e-6_64 --hidden_dim 64 --lstm_layer 2 --bidirectional T --data_size small --main_encoder BERT --model_type LSTMCRF --batch_size 4

To evaluate trained model, use --process_type test, e.g. 

    python main.py --process_type test --task_type record --model_alias bert --hidden_dim 64 --lstm_layer 2 --bidirectional T --data_size small --main_encoder BERT --model_type LSTMCRF --batch_size 4

To produce tagged results from raw data, use existing models including both page model and record model, use --process_type produce, e.g.

    python main.py --process_type produce --model_alias bert --data_size small --model_type LSTMCRF --hidden_dim 64 --lstm_layer 2 --bidirectional T
    
## Input Directory

LGtagging_LSTM/logart_html

## Requirement

torch

pytorch-pretrained-bert

pytorch-crf

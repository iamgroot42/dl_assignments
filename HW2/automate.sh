#!/bin/bash

python main.py --type segment --dimension 2 --learning_rate 0.01 --batch_size 8 --nb_epochs 10
python main.py --type classify --dimension 2 --learning_rate 1 --batch_size 128 --nb_epochs 10
python main.py --type classify --dimension 3 --learning_rate 1 --batch_size 128 --nb_epochs 10

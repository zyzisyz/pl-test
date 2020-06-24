#!/bin/bash

# Author: Yang Zhang 
# Mail: zyziszy@foxmail.com 

############
## config    
############
data_dir="./data"

############
## fit    
############
python -u main.py \
	--gpus 1 \
	--data_dir $data_dir \
	--learning_rate 0.01 \
	--max_epochs 100 \
	--batch_size 1000
    # --distributed_backend "dp" \

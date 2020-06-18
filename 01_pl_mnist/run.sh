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
python main.py \
	--data_dir $data_dir \
	--learning_rate 0.01

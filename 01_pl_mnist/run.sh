#!/bin/bash

#*************************************************************************
#	> File Name: run.sh
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com 
#	> Created Time: Tue Jun 16 13:22:50 2020
# ************************************************************************/

############
## config    
############
data_dir="./data"


############
## fit    
############
python main.py --data_dir=$data_dir


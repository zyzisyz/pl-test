#!/bin/bash

#*************************************************************************
#	> File Name: run.sh
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com 
#	> Created Time: Tue Jun 16 13:22:50 2020
# ************************************************************************/


############
## MNIST    
############
## give instructions on how to run your code.   

############
# CPU   
############
# python main.py     

############
# specific GPUs
############
# python main.py --gpus '0,3'
python main.py --gpu '0'

############
# Multiple-GPUs   
############
# python main.py --gpus 4


############
# On multiple nodes   
############
# python main.py --gpus 4 --nodes 4  --precision 16


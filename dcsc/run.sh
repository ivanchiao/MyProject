#!/bin/bash  
for i in $(seq 1 10)  
do   
python train.py --dataset 'klein' --times $i;  
done

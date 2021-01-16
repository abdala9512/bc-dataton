#!/bin/bash

# Make sure to install csvkit: pip install csvkit
months=(201902 201903 201904 201905 201907 201908 201909 201910 201911
        202001 202002 202003 202004 202005 202007 202008 202009 202010 202011)

for month in ${months[*]}
do
    csvgrep -H -c 1 -m "$month" Dataton_train.csv > train_$month.csv 
done 
#!/bin/bash

for i in $(seq 21 24)
do
    python3 main.py -m resnet34 -g 0 -t ${i} -c CR -f 5
done
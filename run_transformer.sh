#!/bin/bash

for i in $(seq 0 2)
do
    python3 main_transformer.py -m resnet34 -g 1 -t ${i} -c CR -f 1
done
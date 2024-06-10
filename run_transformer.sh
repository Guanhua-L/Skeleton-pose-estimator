#!/bin/bash

for i in $(seq 0 0)
do
    python3 main_transformer.py -m resnet34 -g 2 -t ${i} -c CR -f 5
done
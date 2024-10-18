#!/bin/bash

python train.py datasets/WaterDropSample \
    --output output \
    --epoch 1 \
    --eval-interval 100 \
    --vis-interval 100 \
    --save-interval 100


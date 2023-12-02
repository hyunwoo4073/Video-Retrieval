#!/bin/bash

spark-submit \
    --deploy-mode client \
    --num-executors 15 \
    data_desc.py

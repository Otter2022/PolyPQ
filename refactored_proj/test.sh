#!/bin/bash

# Run PQ recall evaluation with Jaccard‑ADC over your sparse .txt query set
python new_query.py \
    --query_file /home/otto/uni_filtered_pk-5e-06-5e-05-147-147 \
    --sparse --dimension 21609 \
    --database_codes_file outputpoly11024/polygon__quantized_codes.npy \
    --codebooks_prefix outputpoly11024/polygon \
    --groundtruth_dir ../../../warehouse/pk-fil_5e-06-5e-05 \
    --M 1 \
    --Ks 1024 \
    --topk 100 \
    --distance_metric adc \
    --adc_metric jaccard \
    > outputpoly11024/output.txt

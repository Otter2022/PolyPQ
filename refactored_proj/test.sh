#!/bin/bash

python New_PQ.py --data_file ../data/sift/sift_base.fvecs --M 8 --Ks 1024 --output_dir output81024 --output_prefix sift

python new_query.py --query_file ../data/sift/sift_query.fvecs \
                    --database_codes_file output81024/sift__quantized_codes.npy --codebooks_prefix sift --groundtruth_file ../data/sift/sift_groundtruth.ivecs \
                    --M 8 --Ks 1024 --topk 100 --output_dir output81024 > output81024/output.txt

python New_PQ.py --data_file ../data/sift/sift_base.fvecs --M 8 --Ks 64 --output_dir output864 --output_prefix sift

python new_query.py --query_file ../data/sift/sift_query.fvecs \
                    --database_codes_file output864/sift__quantized_codes.npy --codebooks_prefix sift --groundtruth_file ../data/sift/sift_groundtruth.ivecs \
                    --M 8 --Ks 64 --topk 100 --output_dir output864 > output864/output.txt

python New_PQ.py --data_file ../data/sift/sift_base.fvecs --M 8 --Ks 4096 --output_dir output84096 --output_prefix sift

python new_query.py --query_file ../data/sift/sift_query.fvecs \
                    --database_codes_file output84096/sift__quantized_codes.npy --codebooks_prefix sift --groundtruth_file ../data/sift/sift_groundtruth.ivecs \
                    --M 8 --Ks 4096 --topk 100 --output_dir output84096 > output84096/output.txt

python New_PQ.py --data_file ../data/sift/sift_base.fvecs --M 4 --Ks 64 --output_dir output464 --output_prefix sift

python new_query.py --query_file ../data/sift/sift_query.fvecs \
                    --database_codes_file output464/sift__quantized_codes.npy --codebooks_prefix sift --groundtruth_file ../data/sift/sift_groundtruth.ivecs \
                    --M 4 --Ks 64 --topk 100 --output_dir output464 > output464/output.txt

python New_PQ.py --data_file ../data/sift/sift_base.fvecs --M 4 --Ks 1024 --output_dir output41024 --output_prefix sift

python new_query.py --query_file ../data/sift/sift_query.fvecs \
                    --database_codes_file output41024/sift__quantized_codes.npy --codebooks_prefix sift --groundtruth_file ../data/sift/sift_groundtruth.ivecs \
                    --M 4 --Ks 1024 --topk 100 --output_dir output41024 > output41024/output.txt

python New_PQ.py --data_file ../data/sift/sift_base.fvecs --M 4 --Ks 4096 --output_dir output44096 --output_prefix sift

python new_query.py --query_file ../data/sift/sift_query.fvecs \
                    --database_codes_file output44096/sift__quantized_codes.npy --codebooks_prefix sift --groundtruth_file ../data/sift/sift_groundtruth.ivecs \
                    --M 4 --Ks 4096 --topk 100 --output_dir output44096 > output44096/output.txt

python New_PQ.py --data_file ../data/sift/sift_base.fvecs --M 16 --Ks 64 --output_dir output1664 --output_prefix sift

python new_query.py --query_file ../data/sift/sift_query.fvecs \
                    --database_codes_file output1664/sift__quantized_codes.npy --codebooks_prefix sift --groundtruth_file ../data/sift/sift_groundtruth.ivecs \
                    --M 16 --Ks 64 --topk 100 --output_dir output1664 > output1664/output.txt

python New_PQ.py --data_file ../data/sift/sift_base.fvecs --M 16 --Ks 1024 --output_dir output161024 --output_prefix sift

python new_query.py --query_file ../data/sift/sift_query.fvecs \
                    --database_codes_file output161024/sift__quantized_codes.npy --codebooks_prefix sift --groundtruth_file ../data/sift/sift_groundtruth.ivecs \
                    --M 16 --Ks 1024 --topk 100 --output_dir output161024 > output161024/output.txt

python New_PQ.py --data_file ../data/sift/sift_base.fvecs --M 16 --Ks 4096 --output_dir output164096 --output_prefix sift

python new_query.py --query_file ../data/sift/sift_query.fvecs \
                    --database_codes_file output164096/sift__quantized_codes.npy --codebooks_prefix sift --groundtruth_file ../data/sift/sift_groundtruth.ivecs \
                    --M 16 --Ks 4096 --topk 100 --output_dir output164096 > output164096/output.txt
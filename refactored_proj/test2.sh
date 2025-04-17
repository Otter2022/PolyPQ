#!/bin/bash

python New_PQ.py --data_file /home/otto/uni_filtered_pk-5e-06-5e-05-147-147 --M 1 --Ks 1024 --output_dir outputpoly11024x --output_prefix polygon --clustering jaccard > outputencodingx.txt

echo "_________________________" >> outputencoding.txt

python New_PQ.py --data_file /home/otto/uni_filtered_pk-5e-06-5e-05-147-147 --M 1 --Ks 256 --output_dir outputpoly1256x --output_prefix polygon --clustering jaccard >> outputencodingx.txt

echo "_________________________" >> outputencoding.txt

python New_PQ.py --data_file /home/otto/uni_filtered_pk-5e-06-5e-05-147-147 --M 49 --Ks 1024 --output_dir outputpoly491024x --output_prefix polygon --clustering jaccard >> outputencodingx.txt

echo "_________________________" >> outputencoding.txt

python New_PQ.py --data_file /home/otto/uni_filtered_pk-5e-06-5e-05-147-147 --M 49 --Ks 256 --output_dir outputpoly49256x --output_prefix polygon --clustering jaccard >> outputencodingx.txt

echo "_________________________" >> outputencoding.txt

python New_PQ.py --data_file /home/otto/uni_filtered_pk-5e-06-5e-05-147-147 --M 147 --Ks 1024 --output_dir outputpoly1471024x --output_prefix polygon --clustering jaccard >> outputencodingx.txt

echo "_________________________" >> outputencoding.txt

python New_PQ.py --data_file /home/otto/uni_filtered_pk-5e-06-5e-05-147-147 --M 147 --Ks 256 --output_dir outputpoly147256x --output_prefix polygon --clustering jaccard  >> outputencodingx.txt

# python New_PQ.py --data_file /home/otto/uni_filtered_pk-5e-06-5e-05-147-147 --M  --Ks 1024 --output_dir outputpoly2021024 --output_prefix polygon --clustering jaccard

# python New_PQ.py --data_file /home/otto/uni_filtered_pk-5e-06-5e-05-147-147 --M 202 --Ks 256 --output_dir outputpoly202256 --output_prefix polygon --clustering jaccard


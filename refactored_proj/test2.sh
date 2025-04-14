#!/bin/bash

python New_PQ.py --data_file ../data/tmp/shared/encoding/pk-50k0.002 --M 13 --Ks 1024 --output_dir outputpoly131024 --output_prefix polygon --clustering jaccard

python New_PQ.py --data_file ../data/tmp/shared/encoding/pk-50k0.002 --M 13 --Ks 256 --output_dir outputpoly13256 --output_prefix polygon --clustering jaccard

python New_PQ.py --data_file ../data/tmp/shared/encoding/pk-50k0.002 --M 26 --Ks 1024 --output_dir outputpoly261024 --output_prefix polygon --clustering jaccard

python New_PQ.py --data_file ../data/tmp/shared/encoding/pk-50k0.002 --M 26 --Ks 256 --output_dir outputpoly26256 --output_prefix polygon --clustering jaccard

python New_PQ.py --data_file ../data/tmp/shared/encoding/pk-50k0.002 --M 101 --Ks 1024 --output_dir outputpoly1011024 --output_prefix polygon --clustering jaccard

python New_PQ.py --data_file ../data/tmp/shared/encoding/pk-50k0.002 --M 101 --Ks 256 --output_dir outputpoly101256 --output_prefix polygon --clustering jaccard

python New_PQ.py --data_file ../data/tmp/shared/encoding/pk-50k0.002 --M 202 --Ks 1024 --output_dir outputpoly2021024 --output_prefix polygon --clustering jaccard

python New_PQ.py --data_file ../data/tmp/shared/encoding/pk-50k0.002 --M 202 --Ks 256 --output_dir outputpoly202256 --output_prefix polygon --clustering jaccard


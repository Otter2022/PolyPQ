#!/bin/bash

# python make_descriptors.py

# Define an associative array mapping descriptor keys to filenames and output directories.
declare -A descriptors=(
    ["css"]="css_descriptors.fvecs outputcss"
    ["descriptors"]="descriptors.fvecs outputfourier2"
    ["ehd"]="ehd_descriptors.fvecs outputehd"
    ["fourier"]="fourier_descriptors.fvecs outputfourier1"
    ["sift"]="sift_descriptors.fvecs outputsift"
    ["gist"]="gist_descriptors.fvecs outputgist"
    ["hog"]="hog_descriptors.fvecs outputhog"
)

# Loop over the M and Ks values
for M in 16 32 64; do
    for Ks in 1024 512; do
        echo "Processing with M=${M} and Ks=${Ks}"
        # Loop over each descriptor type
        for key in "${!descriptors[@]}"; do
            # Read the descriptor file and output directory from the mapping.
            read -r data_file output_dir <<< "${descriptors[$key]}"
            
            echo "  Running for descriptor: ${data_file} (output: ${output_dir})"
            
            # Run the New_PQ.py command.
            python New_PQ.py --data_file ./"${data_file}" --M ${M} --Ks ${Ks} --output_dir "${output_dir}" --output_prefix sift --clustering_metric l2
            
            # Run the new_query.py command.
            python new_query.py --query_file ./"${data_file}" \
                --database_codes_file "${output_dir}"/sift__quantized_codes.npy \
                --codebooks_prefix sift \
                --groundtruth_dir ../../../warehouse/pk-fil_5e-06-5e-05 \
                --M ${M} --Ks ${Ks} --topk 100 --output_dir "${output_dir}" \
                --distance_metric adc --adc_metric l2 > results/"${output_dir}"_"${M}"_"${Ks}".txt
        done
    done
done

#!/bin/bash

main_dir=$1
m2_filepath="${main_dir}/bea2019/nucle.train.gold.bea19.m2"
parallel_dir="${main_dir}/parallel"
output_dir="${main_dir}/gector"

data_scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"
m2_to_parallel_script_path="${data_scripts_dir}/parallel_from_m2.py"
preprocess_script_path="${data_scripts_dir}/preprocess.py"

# Create output directories
mkdir -p "$parallel_dir"
mkdir -p "$output_dir"

# Convert train, dev & test m2s to parallel texts
m2_filename_no_path=${m2_filepath##*/}
correct_filepath="${parallel_dir}/${m2_filename_no_path}_correct.txt"
incorrect_filepath="${parallel_dir}/${m2_filename_no_path}_incorrect.txt"
python "$m2_to_parallel_script_path" --m2 "$m2_filepath" -c "$correct_filepath" -i "$incorrect_filepath"

# Convert parallel files to GECToR format
python "$preprocess_script_path" -i "$incorrect_filepath" -c "$correct_filepath" -o "${output_dir}/nucle.txt"

echo "DONE"
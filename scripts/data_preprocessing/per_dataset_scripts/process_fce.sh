#!/bin/bash

main_dir=$1
m2_dir="${main_dir}/m2"
parallel_dir="${main_dir}/parallel"
output_dir="${main_dir}/gector"

data_scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"
m2_to_parallel_script_path="${data_scripts_dir}/parallel_from_m2.py"
preprocess_script_path="${data_scripts_dir}/preprocess.py"

# Create output directories
mkdir -p "$parallel_dir"
mkdir -p "$output_dir"

# Convert train, dev & test m2s to parallel texts
for m2_filepath in "$m2_dir"/*; do
    m2_filename_no_path=${m2_filepath##*/}
    echo "$m2_filename_no_path"
    correct_filepath="${parallel_dir}/${m2_filename_no_path}_correct.txt"
    incorrect_filepath="${parallel_dir}/${m2_filename_no_path}_incorrect.txt"
    python "$m2_to_parallel_script_path" --m2 "$m2_filepath" -c "$correct_filepath" -i "$incorrect_filepath"
done

# Concat text files
incorrect_merged_filepath="${parallel_dir}/merged_incorrect.txt"
rm -f "$incorrect_merged_filepath"
cat "${parallel_dir}/"*"_incorrect.txt" > "$incorrect_merged_filepath"

correct_merged_filepath="${parallel_dir}/merged_correct.txt"
rm -f "$correct_merged_filepath"
cat "${parallel_dir}/"*"_correct.txt" > "$correct_merged_filepath"

# Convert concatenated parallel files to GECToR format
python "$preprocess_script_path" -i "$incorrect_merged_filepath" -c "$correct_merged_filepath" -o \
"${output_dir}/fce.txt"

echo "DONE"
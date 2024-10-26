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

# Data conversion function
process_dataset () {
  if [ "$1" = "train" ]
  then
    m2_filename="ABC.train.gold.bea19.m2"
  else
    m2_filename="ABCN.dev.gold.bea19.m2"
  fi

  # Convert to parallel texts
  m2_filepath="${m2_dir}/${m2_filename}"
  correct_filepath="${parallel_dir}/${m2_filename}_correct.txt"
  incorrect_filepath="${parallel_dir}/${m2_filename}_incorrect.txt"
  python "$m2_to_parallel_script_path" --m2 "$m2_filepath" -c "$correct_filepath" -i "$incorrect_filepath"

  # Convert concatenated parallel files to GECToR format
  python "$preprocess_script_path" -i "$incorrect_filepath" -c "$correct_filepath" -o \
  "${output_dir}/wi+locness_${1}.txt"
}

process_dataset "train"
process_dataset "dev"

echo "DONE"
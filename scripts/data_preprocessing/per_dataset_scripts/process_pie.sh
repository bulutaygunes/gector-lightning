#!/bin/bash

main_dir=$1
output_dir="${main_dir}/gector"
python_script_path="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )/preprocess.py"

# Create output directories
mkdir -p "$output_dir"

for subset in "a1" "a2" "a3" "a4" "a5"
do
    incorrect_filepath="${main_dir}/${subset}/${subset}_train_incorr_sentences.txt"
    correct_filepath="${main_dir}/${subset}/${subset}_train_corr_sentences.txt"
    output_filepath="${output_dir}/${subset}.txt"
    python "$python_script_path" -i "$incorrect_filepath" -c "$correct_filepath" -o "$output_filepath" &
done
echo "Processing..."
wait
echo "DONE"
#!/bin/bash

main_dir=$1
input_filepath="${main_dir}/alt/official-2014.combined-withalt.m2"
output_filpath="${main_dir}/conll14-errant-auto.m2"

# Re-annotate with ERRANT
errant_m2 "$input_filepath" -out "$output_filpath" -auto

echo "DONE"
#!/bin/bash

# This script takes a root directory as input, finds all .h5 files in that directory,
# and then processes each .h5 file in parallel using split_probes.py.
# Usage: ./split_probes_all.sh <input_root_directory>
#
# For each .h5 file found directly in <input_root_directory>, it invokes:
#   python preprocess/allen_vc_2019_vis/split_probes.py --input-fname <file>

if [ -z "$1" ]; then
  echo "Usage: $0 <input_root_directory>"
  exit 1
fi
INPUT_ROOT="$1"

h5_files=($(find "$INPUT_ROOT" -maxdepth 1 -type f -name "*.h5"))
echo "Found ${#h5_files[@]} .h5 files in $INPUT_ROOT"

# Parallel processing of each .h5 file using split_probes.py
printf "%s\n" "${h5_files[@]}" | xargs -n 1 -P 16 -I {} python utils/split_probes.py --input-fname {}

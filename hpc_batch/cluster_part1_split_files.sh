#!/bin/bash

# Default input folder
INPUT_FOLDER="../input/"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input_folder) INPUT_FOLDER="$2"; shift ;;
        --num_splits) NUM_SPLITS="$2"; shift ;;
        *) INPUT_FILE="$1" ;;
    esac
    shift
done

# Check if input file was provided
if [ -z "$INPUT_FILE" ]; then
    echo "Error: No input file provided."
    echo "Usage: $0 [--input_folder <folder>] --num_splits <number> <input_file>"
    exit 1
fi

# Check if num_splits was provided
if [ -z "$NUM_SPLITS" ]; then
    echo "Error: --num_splits argument is required."
    echo "Usage: $0 [--input_folder <folder>] --num_splits <number> <input_file>"
    exit 1
fi

# Construct full input path
FULL_INPUT_PATH="${INPUT_FOLDER%/}/${INPUT_FILE}"

# Check if file exists
if [ ! -f "$FULL_INPUT_PATH" ]; then
    echo "Error: File '$FULL_INPUT_PATH' not found."
    exit 1
fi

# Variables
HEADER=$(head -1 "$FULL_INPUT_PATH")
TOTAL_LINES=$(wc -l < "$FULL_INPUT_PATH")
DATA_LINES=$((TOTAL_LINES - 1))  # Exclude header
LINES_PER_FILE=$((DATA_LINES / NUM_SPLITS))
REMAINDER=$((DATA_LINES % NUM_SPLITS))

# Function to create a split file
create_split() {
    local start=$1
    local count=$2
    local file_num=$3
    local output_file="${FULL_INPUT_PATH%.*}_${file_num}.csv"
    
    echo "$HEADER" > "$output_file"
    tail -n +$((start + 1)) "$FULL_INPUT_PATH" | head -n $count >> "$output_file"
}

# Split the file
start_line=2  # Start after header
for ((i=0; i<NUM_SPLITS; i++)); do
    count=$LINES_PER_FILE
    if [ $i -lt $REMAINDER ]; then
        count=$((count + 1))
    fi
    
    create_split $start_line $count $(printf "%03d" $i)
    start_line=$((start_line + count))
done

echo "Split complete. Created $NUM_SPLITS files."
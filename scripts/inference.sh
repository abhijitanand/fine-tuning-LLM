#!/bin/bash

echo "Inference using the trained model on the prompts.txt file."

python3 inference_script.py \
    --model_path ./saved_model --input_file prompts.txt \
    --output_file responses.tsv --max_length 150

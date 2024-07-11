#!/bin/bash
outfile='out file path'
datapath='dataset path'
for file in 'trecdl22_for_llm.tsv'
do
  python ../llm_rewrites/main_llm.py $datapath \
    --data_file $file \
    --final_out_file $out_file \
    --is_chatgpt --model_name gpt-4-turbo-preview \
    --skip_row 0 --max_tokens 1200 --temperature 1
    # gpt-3.5-turbo # gpt-4-turbo-preview
done
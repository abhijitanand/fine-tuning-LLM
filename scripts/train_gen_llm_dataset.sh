#!/bin/bash
echo "Fine-tuning the generative LLM model on WIKI data"

python3 /home/aanand/llm_training/models/train_wiki_gen_llm.py \
        --model_name gpt2 --epochs 5 --batch_size 8 --learning_rate 3e-5 --max_length 150

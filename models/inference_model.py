import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

def load_prompts(input_file):
    with open(input_file, 'r') as file:
        prompts = file.readlines()
    return [prompt.strip() for prompt in prompts]

def generate_response(model, tokenizer, prompt, max_length):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def write_responses(output_file, prompt_response_pairs):
    df = pd.DataFrame(prompt_response_pairs, columns=["Prompt", "Response"])
    df.to_csv(output_file, sep='\t', index=False)

def main(args):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Load prompts
    prompts = load_prompts(args.input_file)

    # Generate responses
    prompt_response_pairs = []
    for prompt in prompts:
        response = generate_response(model, tokenizer, prompt, max_length=args.max_length)
        prompt_response_pairs.append((prompt, response))

    # Write responses to output file
    write_responses(args.output_file, prompt_response_pairs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--input_file", type=str, required=True, help="File containing input prompts")
    parser.add_argument("--output_file", type=str, required=True, help="File to write prompt-response pairs")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum length of generated responses")

    args = parser.parse_args()
    main(args)

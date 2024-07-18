import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader

# Fine-tuning the basic generative LLM model on dummy data: prompt-response pairs.
# Step 1: Define Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.data = []
        with open(file_path, 'r') as f:
            for line in f:
                prompt, target = line.strip().split('\t')
                tokenizer.pad_token = tokenizer.eos_token
                encoding = tokenizer(prompt, truncation=True, max_length=max_length, padding="max_length")
                self.data.append((encoding.input_ids, encoding.attention_mask, tokenizer(target, truncation=True, max_length=max_length, padding="max_length").input_ids))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            'input_ids': torch.tensor(self.data[index][0]),
            'attention_mask': torch.tensor(self.data[index][1]),
            'labels': torch.tensor(self.data[index][2])
        }

# Step 2: Fine-tuning Function
def train(model, tokenizer, train_dataset, epochs=3, batch_size=4, learning_rate=2e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}')

    return model

# Step 3: Inference Function
def generate_text(model, tokenizer, prompt, max_length=100):
    model.eval()
    device = next(model.parameters()).device  # Get the device of the model
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    # Generate text
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)

    # Decode and return generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Step 4: Main Execution

def main():
    # Initialize tokenizer and model
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Fine-tune the model
    train_dataset = CustomDataset('../data/data.txt', tokenizer, max_length=128)
    model = train(model, tokenizer, train_dataset, epochs=3, batch_size=4, learning_rate=2e-5)

    # Save the fine-tuned model (optional)
    # model.save_pretrained('fine_tuned_model')

    # Perform inference
    prompt = "Tell me a joke."
    generated_text = generate_text(model, tokenizer, prompt)

    print("Input Prompt:", prompt)
    print("Generated Text:", generated_text)

if __name__ == "__main__":
    main()

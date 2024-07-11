import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.data = []
        tokenizer.pad_token = tokenizer.eos_token
        for text in texts:
            if text.strip():  # Check if text is not empty
                encoding = tokenizer(text, truncation=True, max_length=max_length, padding="max_length")
                self.data.append((encoding.input_ids, encoding.attention_mask))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            'input_ids': torch.tensor(self.data[index][0]),
            'attention_mask': torch.tensor(self.data[index][1])
        }

# Distillation Training Function
def distill(teacher_model, student_model, tokenizer, train_dataset, epochs, batch_size, learning_rate, max_length, temperature):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model.to(device)
    student_model.to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(student_model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    for epoch in range(epochs):
        student_model.train()
        teacher_model.eval()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
                teacher_logits = teacher_outputs.logits

            student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
            student_logits = student_outputs.logits

            loss = distillation_loss(student_logits, teacher_logits, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_train_loss:.4f}')

    return student_model

# Distillation Loss Function
def distillation_loss(student_logits, teacher_logits, temperature):
    loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
    student_probs = torch.nn.functional.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = torch.nn.functional.softmax(teacher_logits / temperature, dim=-1)
    loss = loss_fn(student_probs, teacher_probs) * (temperature ** 2)
    return loss

# Inference Function
def generate_text(model, tokenizer, prompt, max_length):
    model.eval()
    device = next(model.parameters()).device  # Get the device of the model
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    # Generate text
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)

    # Decode and return generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Main Function
def main(args):
    # Load tokenizer and models
    tokenizer = GPT2Tokenizer.from_pretrained(args.teacher_model_name)
    teacher_model = GPT2LMHeadModel.from_pretrained(args.teacher_model_name)
    student_model = GPT2LMHeadModel.from_pretrained(args.student_model_name)

    # Load dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    train_texts = dataset['train']['text']
    
    # Filter out empty texts
    train_texts = [text for text in train_texts if text.strip()]

    # Prepare dataset
    train_dataset = CustomDataset(train_texts, tokenizer, max_length=args.max_length)
    
    # Distill the teacher model to the student model
    student_model = distill(teacher_model, student_model, tokenizer, train_dataset, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, max_length=args.max_length, temperature=args.temperature)

    # Save the distilled student model
    #student_model.save_pretrained(args.save_model_path)

    # Perform inference
    prompt = "Tell me a joke."
    generated_text = generate_text(student_model, tokenizer, prompt, max_length=args.max_length)

    print("Input Prompt:", prompt)
    print("Generated Text:", generated_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument("--teacher_model_name", type=str, default="gpt2-large", help="Teacher model name or path")
    parser.add_argument("--student_model_name", type=str, default="gpt2-medium", help="Student model name or path")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum input length")
    parser.add_argument("--temperature", type=float, default=2.0, help="Distillation temperature")
    parser.add_argument("--save_model_path", type=str, default="./distilled_student_model", help="Path to save the distilled student model")
    
    args = parser.parse_args()
    main(args)

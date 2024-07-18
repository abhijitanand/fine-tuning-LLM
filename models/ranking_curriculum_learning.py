import argparse
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import ir_datasets
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

class RankingDataset(Dataset):
    def __init__(self, tokenizer, queries, documents, labels, max_length):
        self.tokenizer = tokenizer
        self.queries = queries
        self.documents = documents
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        query = self.queries[index]
        document = self.documents[index]
        label = self.labels[index]

        inputs = self.tokenizer(query, document, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}  # Remove batch dimension

        return {'inputs': inputs, 'labels': torch.tensor(label, dtype=torch.float)}

def train(model, tokenizer, train_dataset, val_dataset, epochs, batch_size, learning_rate, max_length, curriculum_stages):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataset) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    loss_fn = torch.nn.MSELoss()

    for stage, (start, end) in enumerate(curriculum_stages):
        stage_dataset = Subset(train_dataset, range(start, end))
        train_loader = DataLoader(stage_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f'Stage {stage+1}/{len(curriculum_stages)}, Epoch {epoch+1}/{epochs}')
            for batch in progress_bar:
                inputs = {k: v.to(device) for k, v in batch['inputs'].items()}
                labels = batch['labels'].to(device)

                outputs = model(**inputs)
                loss = loss_fn(outputs.logits.squeeze(-1), labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            avg_train_loss = total_loss / len(train_loader)
            print(f'Stage [{stage+1}/{len(curriculum_stages)}], Epoch [{epoch+1}/{epochs}], Average Loss: {avg_train_loss:.4f}')

            evaluate(model, val_loader, device, loss_fn)

def evaluate(model, val_loader, device, loss_fn):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: v.to(device) for k, v in batch['inputs'].items()}
            labels = batch['labels'].to(device)

            outputs = model(**inputs)
            loss = loss_fn(outputs.logits.squeeze(-1), labels)
            total_loss += loss.item()

    avg_val_loss = total_loss / len(val_loader)
    print(f'Validation Loss: {avg_val_loss:.4f}')

def load_msmarco_data():
    dataset = ir_datasets.load("msmarco-passage/train")
    queries = dataset.queries_iter()
    docs = dataset.docs_iter()
    qrels = dataset.qrels_iter()

    query_dict = {query.query_id: query.text for query in queries}
    doc_dict = {doc.doc_id: doc.text for doc in docs}

    queries = []
    documents = []
    labels = []

    for qrel in qrels:
        queries.append(query_dict[qrel.query_id])
        documents.append(doc_dict[qrel.doc_id])
        labels.append(qrel.relevance)

    return queries, documents, labels

def sort_by_difficulty(queries, documents, labels):
    data = list(zip(queries, documents, labels))
    data.sort(key=lambda x: x[2], reverse=True)  # Sort by relevance score
    queries, documents, labels = zip(*data)
    return list(queries), list(documents), list(labels)

def main(args):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1)

    # Load and sort dataset
    queries, documents, labels = load_msmarco_data()
    queries, documents, labels = sort_by_difficulty(queries, documents, labels)

    # For demo purposes, select a subset
    subset_size = 1000
    queries = queries[:subset_size]
    documents = documents[:subset_size]
    labels = labels[:subset_size]

    # Split into train and validation sets
    train_queries, val_queries, train_documents, val_documents, train_labels, val_labels = train_test_split(queries, documents, labels, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = RankingDataset(tokenizer, train_queries, train_documents, train_labels, max_length=args.max_length)
    val_dataset = RankingDataset(tokenizer, val_queries, val_documents, val_labels, max_length=args.max_length)

    # Define curriculum stages (start, end indices of dataset)
    num_stages = args.num_stages
    stage_size = len(train_dataset) // num_stages
    curriculum_stages = [(i * stage_size, (i + 1) * stage_size) for i in range(num_stages)]

    # Train the model using curriculum learning
    train(model, tokenizer, train_dataset, val_dataset, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, max_length=args.max_length, curriculum_stages=curriculum_stages)

    # Save the trained model
    #model.save_pretrained(args.save_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Model name or path")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum input length")
    parser.add_argument("--num_stages", type=int, default=5, help="Number of curriculum learning stages")
    parser.add_argument("--save_model_path", type=str, default="./trained_model", help="Path to save the trained model")

    args = parser.parse_args()
    main(args)

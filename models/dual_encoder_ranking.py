import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import ir_datasets

# Custom Dataset Class
class DualEncoderDataset(torch.utils.data.Dataset):
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

        query_inputs = self.tokenizer(query, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        doc_inputs = self.tokenizer(document, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")

        query_inputs = {k: v.squeeze(0) for k, v in query_inputs.items()}  # Remove batch dimension
        doc_inputs = {k: v.squeeze(0) for k, v in doc_inputs.items()}  # Remove batch dimension

        return {'query_inputs': query_inputs, 'doc_inputs': doc_inputs, 'labels': torch.tensor(label, dtype=torch.float)}

# Dual Encoder Model
class DualEncoderModel(torch.nn.Module):
    def __init__(self, model_name):
        super(DualEncoderModel, self).__init__()
        self.query_encoder = AutoModel.from_pretrained(model_name)
        self.doc_encoder = AutoModel.from_pretrained(model_name)

    def forward(self, query_inputs, doc_inputs):
        query_embeds = self.query_encoder(**query_inputs).pooler_output
        doc_embeds = self.doc_encoder(**doc_inputs).pooler_output
        scores = torch.matmul(query_embeds, doc_embeds.t())
        return scores

# Training Function
def train(model, tokenizer, train_dataset, val_dataset, epochs, batch_size, learning_rate, max_length):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    loss_fn = torch.nn.MarginRankingLoss(margin=1.0)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in progress_bar:
            query_inputs = {k: v.to(device) for k, v in batch['query_inputs'].items()}
            doc_inputs = {k: v.to(device) for k, v in batch['doc_inputs'].items()}
            labels = batch['labels'].to(device)

            scores = model(query_inputs, doc_inputs)
            positive_scores = scores.diag()
            negative_scores = scores.fill_diagonal_(float('-inf')).max(dim=1)[0]

            targets = torch.ones_like(positive_scores)
            loss = loss_fn(positive_scores, negative_scores, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_train_loss:.4f}')

        evaluate(model, val_loader, device, loss_fn)

def evaluate(model, val_loader, device, loss_fn):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            query_inputs = {k: v.to(device) for k, v in batch['query_inputs'].items()}
            doc_inputs = {k: v.to(device) for k, v in batch['doc_inputs'].items()}
            labels = batch['labels'].to(device)

            scores = model(query_inputs, doc_inputs)
            positive_scores = scores.diag()
            negative_scores = scores.fill_diagonal_(float('-inf')).max(dim=1)[0]

            targets = torch.ones_like(positive_scores)
            loss = loss_fn(positive_scores, negative_scores, targets)
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

def main(args):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = DualEncoderModel(args.model_name)

    # Load dataset
    queries, documents, labels = load_msmarco_data()

    # For demo purposes, select a subset
    subset_size = 100
    queries = queries[:subset_size]
    documents = documents[:subset_size]
    labels = labels[:subset_size]

    # Split into train and validation sets
    split_idx = int(0.8 * subset_size)
    train_queries = queries[:split_idx]
    train_documents = documents[:split_idx]
    train_labels = labels[:split_idx]

    val_queries = queries[split_idx:]
    val_documents = documents[split_idx:]
    val_labels = labels[split_idx:]

    # Create datasets
    train_dataset = DualEncoderDataset(tokenizer, train_queries, train_documents, train_labels, max_length=args.max_length)
    val_dataset = DualEncoderDataset(tokenizer, val_queries, val_documents, val_labels, max_length=args.max_length)

    # Fine-tune the model
    train(model, tokenizer, train_dataset, val_dataset, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, max_length=args.max_length)

    # Save the fine-tuned model
    #model.save_pretrained(args.save_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Model name or path")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum input length")
    parser.add_argument("--save_model_path", type=str, default="./fine_tuned_model", help="Path to save the fine-tuned model")

    args = parser.parse_args()
    main(args)

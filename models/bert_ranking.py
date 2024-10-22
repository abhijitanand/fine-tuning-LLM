import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import ir_datasets

# Custom Dataset Class
class DocumentRankingDataset(torch.utils.data.Dataset):
    
    def __init__(self, tokenizer, queries, documents, labels, max_length):
            """
            Initializes the BERTRanking model.

            Args:
                tokenizer (Tokenizer): The tokenizer used to tokenize the input data.
                queries (list): A list of query strings.
                documents (list): A list of document strings.
                labels (list): A list of label values.
                max_length (int): The maximum length of the input sequences.

            Returns:
                None
            """
            self.tokenizer = tokenizer
            self.queries = queries
            self.documents = documents
            self.labels = labels
            self.max_length = max_length

    def __len__(self):
            """
            Returns the length of the queries list.

            Returns:
                int: The length of the queries list.
            """
            return len(self.queries)

    def __getitem__(self, index):
        """
        Retrieves the item at the given index from the dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the inputs and labels for the item.
                The dictionary has the following keys:
                - 'input_ids': A tensor containing the input token IDs.
                - 'attention_mask': A tensor containing the attention mask.
                - 'token_type_ids': A tensor containing the token type IDs.
                - 'labels': A tensor containing the labels for the item.
        """
        query = self.queries[index]
        document = self.documents[index]
        label = self.labels[index]

        inputs = self.tokenizer(query, document, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}  # Remove batch dimension
        return {**inputs, 'labels': torch.tensor(label, dtype=torch.long)}

# Training Function
def train(model, tokenizer, train_dataset, val_dataset, epochs, batch_size, learning_rate, max_length):
    """
    Trains the given model using the provided training dataset.

    Args:
        model (torch.nn.Module): The model to be trained.
        tokenizer: The tokenizer used to preprocess the input data.
        train_dataset: The training dataset.
        val_dataset: The validation dataset.
        epochs (int): The number of training epochs.
        batch_size (int): The batch size for training.
        learning_rate (float): The learning rate for the optimizer.
        max_length (int): The maximum length of input sequences.

    Returns:
        None
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in progress_bar:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_train_loss:.4f}')

        evaluate(model, val_loader, device)

def evaluate(model, val_loader, device):
    """
    Evaluate the model on the validation set.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        val_loader (torch.utils.data.DataLoader): The data loader for the validation set.
        device (torch.device): The device to run the evaluation on.

    Returns:
        float: The average validation loss.
    """
    
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

    avg_val_loss = total_loss / len(val_loader)
    print(f'Validation Loss: {avg_val_loss:.4f}')
    return avg_val_loss

def load_msmarco_data():
    """
    Loads the MSMARCO dataset and returns the queries, documents, and labels.

    Returns:
        queries (list): A list of strings representing the queries.
        documents (list): A list of strings representing the documents.
        labels (list): A list of integers representing the relevance labels.
    """
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
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    # Load dataset
    queries, documents, labels = load_msmarco_data()

    # For demo purposes, select a subset
    subset_size = 10000
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
    train_dataset = DocumentRankingDataset(tokenizer, train_queries, train_documents, train_labels, max_length=args.max_length)
    val_dataset = DocumentRankingDataset(tokenizer, val_queries, val_documents, val_labels, max_length=args.max_length)

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

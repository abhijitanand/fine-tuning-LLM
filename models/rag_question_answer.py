import ir_datasets
from rank_bm25 import BM25Okapi
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from tqdm import tqdm
import json

def create_bm25_index(dataset_name):
    """
    Create a BM25 index from the specified dataset.

    Args:
        dataset_name (str): The name of the dataset to load and index.

    Returns:
        bm25 (BM25Okapi): The BM25 index built from the dataset.
        pids (list): A list of passage IDs from the dataset.
        passages (list): A list of passages from the dataset.
    """
    print("Loading dataset...")
    dataset = ir_datasets.load(dataset_name)
    pids = []
    passages = []
    
    print("Indexing passages...")
    for doc in tqdm(dataset.docs_iter(), desc="Indexing Passages", unit="doc"):
        pids.append(doc.doc_id)
        passages.append(doc.text)
    
    print("Tokenizing passages...")
    tokenized_passages = [passage.split() for passage in tqdm(passages, desc="Tokenizing Passages", unit="passage")]
    bm25 = BM25Okapi(tokenized_passages)
    
    return bm25, pids, passages

def retrieve_top_k_passages(query, bm25, passages, k=100):
    """
    Retrieve the top-k passages for a given query using the BM25 index.

    Args:
        query (str): The query for which to retrieve passages.
        bm25 (BM25Okapi): The BM25 index.
        passages (list): The list of all passages.
        k (int): The number of top passages to retrieve (default is 100).

    Returns:
        top_k_passages (list): The top-k retrieved passages.
    """
    tokenized_query = query.split()
    doc_scores = bm25.get_scores(tokenized_query)
    top_k_indices = doc_scores.argsort()[-k:][::-1]
    top_k_passages = [passages[idx] for idx in top_k_indices]
    return top_k_passages

def generate_answer(query, passages, model, tokenizer, device, max_length=512):
    """
    Generate an answer for a query based on the retrieved passages using a language model.

    Args:
        query (str): The input query.
        passages (list): The list of passages retrieved for the query.
        model (PreTrainedModel): The pre-trained language model.
        tokenizer (PreTrainedTokenizer): The tokenizer for the language model.
        device (torch.device): The device to run the model on.
        max_length (int): The maximum length of the generated answer (default is 512).

    Returns:
        answer (str): The generated answer.
    """
    context = " ".join(passages)
    input_text = f"question: {query} context: {context}"
    inputs = tokenizer(input_text, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
    inputs = inputs.to(device)
    outputs = model.generate(inputs['input_ids'], max_length=max_length)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def main():
    """
    Main function to orchestrate the overall process:
    - Creates the BM25 index from the MSMARCO passage dataset.
    - Retrieves the top-k passages for queries from the TREC DL 2019 dataset.
    - Generates answers using a pre-trained language model.
    """
    # Parameters
    msmarco_dataset_name = "msmarco-passage/train"
    trecdl_query_dataset_name = "msmarco-passage/trec-dl-2019/judged"
    index_path = 'msmarco-passage-bm25-index'
    model_name = 't5-large'
    top_k = 100
    max_length = 512

    # Create BM25 Index
    bm25, pids, passages = create_bm25_index(msmarco_dataset_name)

    # Save pids and passages for later use
    print("Saving PIDs and passages...")
    with open("pids.json", "w") as pid_file:
        json.dump(pids, pid_file)
    
    with open("passages.json", "w") as passage_file:
        json.dump(passages, passage_file)
    
    print("PIDs and passages saved for later use")

    # Load the TREC DL 2019 queries
    print("Loading TREC DL 2019 queries...")
    queries = ir_datasets.load(trecdl_query_dataset_name).queries_iter()

    # Load the pre-trained language model and tokenizer
    print(f"Loading pre-trained model '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Process queries and generate answers
    print("Processing queries and generating answers...")
    for query in tqdm(queries, desc="Processing Queries", unit="query"):
        query_text = query.text
        top_k_passages = retrieve_top_k_passages(query_text, bm25, passages, k=top_k)
        answer = generate_answer(query_text, top_k_passages, model, tokenizer, device, max_length)
        print(f"Query: {query_text}")
        print(f"Generated Answer: {answer}\n")

if __name__ == "__main__":
    main()

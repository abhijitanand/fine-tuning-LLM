import argparse
import os
import subprocess
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import ir_datasets
import numpy as np

class RankingDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, queries, documents, qids, docids, max_length):
        self.tokenizer = tokenizer
        self.queries = queries
        self.documents = documents
        self.qids = qids
        self.docids = docids
        self.max_length = max_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        query = self.queries[index]
        document = self.documents[index]
        qid = self.qids[index]
        docid = self.docids[index]

        inputs = self.tokenizer(query, document, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}  # Remove batch dimension

        return {'inputs': inputs, 'qid': qid, 'docid': docid}

def evaluate(model, dataloader, device, output_path):
    model.eval()
    results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = {k: v.to(device) for k, v in batch['inputs'].items()}
            qids = batch['qid']
            docids = batch['docid']

            outputs = model(**inputs)
            scores = outputs.logits.squeeze(-1)

            for qid, docid, score in zip(qids, docids, scores.cpu().numpy()):
                results.append(f"{qid}\tQ0\t{docid}\t0\t{score}\tBERT")

    with open(output_path, 'w') as f:
        for result in results:
            f.write(result + '\n')

def parse_trec_eval_output(output):
    map_score, ndcg_10, rr = None, None, None
    for line in output.split('\n'):
        if "map" in line and "all" in line:
            map_score = float(line.split()[-1])
        elif "ndcg_cut_10" in line:
            ndcg_10 = float(line.split()[-1])
        elif "recip_rank" in line:
            rr = float(line.split()[-1])
    return map_score, ndcg_10, rr

def run_trec_eval(qrels_path, results_path, model_name, output_file):
    command = f"./trec_eval -m ndcg_cut.10 -m map -m recip_rank {qrels_path} {results_path}"
    output = subprocess.getoutput(command)
    print(output)

    map_score, ndcg_10, rr = parse_trec_eval_output(output)
    
    with open(output_file, 'a') as f:
        #f.write(f"{model_name}\t{map_score:.4f}\t{ndcg_10:.4f}\t{rr:.4f}\n")
        print(model_name + "\t" + round(float(map_score),4) + "\t" + round(float(ndcg_10),4) + "\t" + round(float(rr),4) + "\n")
        f.write(model_name + "\t" + round(float(map_score),4) + "\t" + round(float(ndcg_10),4) + "\t" + round(float(rr),4) + "\n")

def load_trecdl_data(year):
    if year == "19":
        dataset = ir_datasets.load("msmarco-passage/trec-dl-2019")
    elif year == "20":
        dataset = ir_datasets.load("msmarco-passage/trec-dl-2020")
    else:
        raise ValueError(f"Unsupported year: {year}")

    queries = {query.query_id: query.text for query in dataset.queries_iter()}
    docs = {doc.doc_id: doc.text for doc in dataset.docs_iter()}

    qids, docids, query_texts, doc_texts = [], [], [], []

    for qrel in dataset.qrels_iter():
        qids.append(qrel.query_id)
        docids.append(qrel.doc_id)
        query_texts.append(queries[qrel.query_id])
        doc_texts.append(docs[qrel.doc_id])

    return query_texts, doc_texts, qids, docids

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1)
    model.to(device)

    # Load test data
    queries, documents, qids, docids = load_trecdl_data(args.year)

    # Create dataset and dataloader
    test_dataset = RankingDataset(tokenizer, queries, documents, qids, docids, max_length=args.max_length)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Evaluate and save results
    results_path = f"results_trecdl_{args.year}.txt"
    evaluate(model, test_dataloader, device, results_path)

    # Run trec_eval
    qrels_path = f"qrels_trecdl_{args.year}.txt"  # Adjust this path based on where your qrels file is stored
    run_trec_eval(qrels_path, results_path, args.model_name, args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Model name or path")
    parser.add_argument("--year", type=str, choices=["19", "20"], default="19", help="TRECDL test set year (19 or 20)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum input length")
    parser.add_argument("--output_file", type=str, default="evaluation_results.txt", help="Output file to save evaluation metrics")
    
    args = parser.parse_args()
    main(args)

import argparse
import time
import timeit
from pathlib import Path
from typing import Any, Dict, Tuple, Union
import pandas as pd
from openai import OpenAI

# Initialize OpenAI client
openai_client = OpenAI(api_key="your api key")
pd.options.mode.copy_on_write = True

class QueryIntentExpander:
    def __init__(self, data_file: str, hyperparameters: Dict[str, Any]) -> None:
        """
        Initializes the QueryIntentExpander with data file and hyperparameters.

        Parameters:
            data_file (str): Path to the input data file.
            hyperparameters (Dict[str, Any]): Dictionary containing model and request parameters.
        """
        self.skip_rows = hyperparameters["skip_row"]
        self.model_name = hyperparameters["model_name"]
        self.temperature = hyperparameters["temperature"]
        self.top_p = hyperparameters["top_p"]
        self.max_tokens = hyperparameters["max_tokens"]
        self.frequency_penalty = hyperparameters["frequency_penalty"]
        self.presence_penalty = hyperparameters["presence_penalty"]
        self.output_file = hyperparameters["final_out_file"]
        self.is_chatgpt = hyperparameters["is_chatgpt"]
        
        print(f"Model: {self.model_name}, Data File: {data_file}, Is ChatGPT: {self.is_chatgpt}")
        
        # Load data from file
        self.data_frame = pd.read_csv(data_file, delimiter="\t", header=None, skiprows=self.skip_rows,
                                     names=["q_id", "doc_id", "query", "doc"])
        print(f"Number of queries: {self.data_frame.q_id.nunique()}")

    def _generate_model_response(self, query: str, document: str) -> Dict[str, Any]:
        """
        Generates a model response for a given query and document based on the selected model.

        Parameters:
            query (str): The query string.
            document (str): The document string.

        Returns:
            Dict[str, Any]: The response from the model.
        """
        if self.is_chatgpt:
            response = openai_client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                messages=[
                    {"role": "system", "content": "You are an intelligent system and your job is to predict the intention behind the user question given a list of documents."},
                    {"role": "user", "content": f"A person wants to find out distinct intentions behind the question {query}. Provide five descriptive (max. 15 words) distinct intentions which are easy to understand. Consider all documents in your response. Response should be in this format: Intention:: <intention>, Doc_list::<list of documents with the intention>\n\nDocuments: {document}"}
                ]
            )
        else:
            response = openai_client.chat.completions.create(
                model=self.model_name,
                prompt=f"Being a ranking model, your task is to do query expansion. Given a query and a document, expand the query to be relevant to the document. Expand and contextualize the query in one or two short sentences. Do not ask questions. Only answer with the expanded query. If you cannot expand, just output UNKNOWN.\nQuery: {query}\nDocument: {document}",
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty
            )
        return response

    def _chunk_sequence(self, sequence: pd.DataFrame, chunk_size: int) -> pd.DataFrame:
        """
        Chunks a DataFrame into smaller parts for processing.

        Parameters:
            sequence (pd.DataFrame): The DataFrame to be chunked.
            chunk_size (int): Size of each chunk.

        Returns:
            pd.DataFrame: Generator yielding chunks of the DataFrame.
        """
        for start in range(0, len(sequence), chunk_size):
            yield sequence[start:start + chunk_size]

    def process_queries(self) -> pd.DataFrame:
        """
        Processes each query in the dataset, retrieves model completions, and saves the results to a file.

        Returns:
            pd.DataFrame: DataFrame containing query ID, query, and model response for each query.
        """
        result_data = {"q_id": [], "doc_id": [], "query": [], "description": []}
        index = 0
        chunk_size = 10
        start_time = timeit.default_timer()
        query_ids = self.data_frame.q_id.unique()
        
        for query_id in query_ids:
            subset_df = self.data_frame[self.data_frame['q_id'] == query_id]
            for chunk in self._chunk_sequence(subset_df, chunk_size):
                document_ids = ','.join(chunk.doc_id)
                query_text = subset_df['query'].iloc[0]
                merged_docs = '\n'.join(chunk["doc_id"] + ":" + chunk["doc"])
                
                result_data['query'].append(query_text)
                result_data['q_id'].append(query_id)
                result_data['doc_id'].append(document_ids)
                
                try:
                    if index % 20 == 0 and index != 0:
                        print(f"Processing chunk {index}")
                    response = self._generate_model_response(query_text, merged_docs)
                except Exception as e:
                    print(f"Error occurred: {e}")
                    time.sleep(60)
                    response = self._generate_model_response(query_text, merged_docs)
                
                if self.is_chatgpt:
                    model_response = response.choices[0].message.content.strip().replace('\n', '').replace('\t', '')
                else:
                    model_response = response['choices'][0]['text'].strip()
                
                print(f"Query ID: {query_id}, Query: {query_text}, Response: {model_response}")
                result_data["description"].append(model_response)
                
                # Save results incrementally to avoid data loss
                pd.DataFrame(result_data).to_csv(self.output_file, sep="\t", index=False)
                
                if index % 10000 == 0:
                    print(f"Queries completed: {index}")
                if index % 100 == 0:
                    print(f'Time for {index} requests: {timeit.default_timer() - start_time}')  
                
                index += chunk_size
        
        return pd.DataFrame(result_data)

def main() -> None:
    """
    Main function to parse arguments, initialize the QueryIntentExpander, and process the queries.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('DATA_DIR', help='Folder with all preprocessed files')
    parser.add_argument('--data_file', type=str, default="pass_all_train_query_car_12k.tsv", help='Dataset file name')
    parser.add_argument('--model_name', type=str, default="text-davinci-003", help='LLM model name')
    parser.add_argument('--final_out_file', type=str, default="pass_rewrite_query_davinci_car_12k.tsv", help='Output file name')
    parser.add_argument('--temperature', type=float, default=0.6, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top-p (nucleus sampling) probability')
    parser.add_argument('--frequency_penalty', type=float, default=0.0, help='Frequency penalty')
    parser.add_argument('--presence_penalty', type=float, default=0.0, help='Presence penalty')
    parser.add_argument('--max_tokens', type=int, default=512, help='Maximum tokens in response')
    parser.add_argument('--skip_row', type=int, default=0, help='Number of rows to skip in the input file')
    parser.add_argument('--is_chatgpt', action='store_true', help='Flag to use ChatGPT model')
    parser.add_argument('--only_query', action='store_true', help='Flag to generate only from query')
    parser.add_argument('--if_groundtruth', action='store_true', help='Flag for groundtruth generation')
    
    args = parser.parse_args()
    data_directory = Path(args.DATA_DIR)
    data_file_path = data_directory / args.data_file
    
    print(f"Final output file: {args.final_out_file}")
    
    expander = QueryIntentExpander(data_file_path, vars(args))
    results_df = expander.process_queries()
    
    print(f'Number of Generations: {len(results_df)}')

if __name__ == '__main__':
    main()

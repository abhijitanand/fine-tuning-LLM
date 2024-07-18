# conda env bert_qpp
from openai import OpenAI
import pandas as pd
import time
from typing import Any, Dict, Tuple, Union
import timeit
import argparse
from pathlib import Path

client = OpenAI(api_key = "your api key")
pd.options.mode.copy_on_write = True
class LLMRewriter:
    def __init__(self, data_file, hparams: Dict[str, Any]) -> None:
        self.skip_row = hparams["skip_row"]
        self.model_name = hparams["model_name"]
        self.temperature = hparams["temperature"]
        self.top_p = hparams["top_p"]
        self.max_tokens = hparams["max_tokens"]
        self.frequency_penalty = hparams["frequency_penalty"]
        self.presence_penalty = hparams["presence_penalty"]
        self.final_out_file = hparams["final_out_file"]
        self.is_chatgpt = hparams["is_chatgpt"]
        print("Model: {}, data_file: {}, is_ChatGPT: {}".format(self.model_name, data_file, self.is_chatgpt))
        self.df_final = pd.read_csv(data_file, delimiter="\t", header=None, skiprows=self.skip_row,
                                    names=["q_id", "doc_id", "query", "doc"])
        print("number of queries:", self.df_final.q_id.nunique())

    def model_prompts(self, query, doc):        
        if self.is_chatgpt:
            response = client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                messages=[
                    {
                    "role": "system",
                    "content": f"You are an intelligent system and your job is to predict the intention behind the user question given a list of documents."
                    },
                    
                    {
                    "role": "user",
                    "content": f"A person want to find out distinct intention behind the question {query}. Give five descriptive (max. 15 words) distinct intention which are easy to understand. Consider all documents in your response.Response should be in this format: Intention:: <intention> , Doc_list::<list of documents with the intention>\n\nDocuments: {doc}"
                    },
                ]
            )
        else:
            response = client.chat.completions.create(
                model=self.model_name,
                prompt="Being a ranking model your task is to do query expansion. This means given a query and a document expand the query such that it is relevant to the document. Expand and contextualise query as best as you can in one or two short sentences. Please do not ask any questions. Only answer with the expanded query as a question. If you are still unable to then just output UNKNOWN.\nquery:" + query + "\ndocument:" + doc,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty
            )
        return response

    def chunk(self, seq, size):
            return (seq[pos:pos + size] for pos in range(0, len(seq), size))
    
    def get_gpt3_completions(self):
            """
            Retrieves GPT-3 completions for each query in the dataset.

            Returns:
                final_df (pandas.DataFrame): DataFrame containing the query ID, query, and GPT-3 completion for each query.
            """
            job_df = {"q_id":[], "doc_id":[], "query":[],"description":[]}
            i=0
            n=10
            index = 0
            start = timeit.default_timer()
            qid_list = self.df_final.q_id.unique()
            
            for id in qid_list:
                df = self.df_final[self.df_final['q_id']==id]
                for df_chunk in self.chunk(df, n):
                    # merge ids to report
                    str_doc_ids = ','.join(df_chunk.doc_id)
                    query_str = df['query'].iloc[0]

                    # merge doc_id and doc and then merge all n rows.
                    df_chunk["doc_merge"] = df_chunk["doc_id"] + ":" + df_chunk["doc"]
                    str_final_doc= '\n'.join(df_chunk.doc_merge)
                    job_df['query'].append(query_str)
                    job_df['q_id'].append(id)
                    job_df['doc_id'].append(str_doc_ids)

                    try:
                        if index % 20 == 0 and index != 0:
                            print("************", index)
                        response = self.model_prompts(query_str,str_final_doc)

                    except Exception as e:
                        print("Error is:", e)
                        time.sleep(60)
                        response = self.model_prompts(query_str,str_final_doc)
                    if self.is_chatgpt:
                        llm_response = response.choices[0].message.content.strip().replace('\n', '').replace('\t', '')
                    else:
                        llm_response = response['choices'][0]['text'].strip()
                    print("q_id:{}, query:{}, Response:{}".format(id, query_str, llm_response))

                    job_df["description"].append(llm_response)
                    final_df = pd.DataFrame(job_df)
                    final_df.to_csv(self.final_out_file, sep="\t", index=False)

                    if i%10000 == 0:
                        print("Queries completed:{}".format(i))
                    if i%100 == 0:
                        print('Time for {} request: {}'.format(i,timeit.default_timer() - start))  
                    i+=1
                    index +=10
            return final_df

def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('DATA_DIR', help='Folder with all preprocessed files')
    ap.add_argument('--data_file', type=str, default="pass_all_train_query_car_12k.tsv", help='dataset file name')
    ap.add_argument('--model_name', type=str, default="text-davinci-003", help='llm model name')
    ap.add_argument('--final_out_file', type=str, default="pass_rewrite_query_davinci_car_12k.tsv", help='out file_name')
    ap.add_argument('--temperature', type=float, default=0.6)
    ap.add_argument('--top_p', type=float, default=1.0)
    ap.add_argument('--frequency_penalty', type=float, default=0.0)
    ap.add_argument('--presence_penalty', type=float, default=0.0)
    ap.add_argument('--max_tokens', type=int, default=512)
    ap.add_argument('--skip_row', type=int, default=0)
    ap.add_argument('--is_chatgpt', action='store_true', help='if model is chatgpt')
    ap.add_argument('--only_query', action='store_true', help='if generating only from query')
    ap.add_argument('--if_groundtruth', action='store_true', help='for groundtruth generation')
    args = ap.parse_args()

    data_dir = Path(args.DATA_DIR)
    data_file = data_dir / args.data_file

    print("Final out file:{}".format(args.final_out_file))
    llm_rewriter = LLMRewriter(data_file,vars(args))
    df = llm_rewriter.get_gpt3_completions()
    print('Number of Generations:{}'.format(len(df)))

if __name__ == '__main__':
    main()
# Fine-tuning LLM using different approaches and for differnt tasks.

This repository contains code examples demonstrating different methods for fine-tuning Language Models (LLMs) across a range of tasks. These implementations provide a starting point for your work. It is recommended that you have prior knowledge of the various approaches to understand their advantages and limitations.

## Ranking Task
1. ***BERT(MonoBERT)-Single Encoder*** :  Fine-tuning a BERT model on the MS MARCO dataset for Document Ranking task. This approach uses a classical transformer model with query and document as input. [Code Link](./models/bert_ranking.py)

2. ***Dual Encoder Model*** : Fine-tuning BERT based Dual Encoder model on Ms-Marco dataset for Document Ranking task. This technique, used in models like TCT-Colber, involves dense retrieval and ranking. [Code Link](dual_encoder_ranking.py). [Paper Link](https://arxiv.org/pdf/2010.11386).

3. ***Distilled Model*** : Distill a teacher model (BERT) to student model(DistillBERT) for document ranking. This process creates a smaller, more efficient model based on a larger model, such as DistilBERT. Eg; DistillBERT. [Code Link](./models/distill_ranker.py). [Paper Link](https://arxiv.org/pdf/1910.01108)

4. ***Distilled Dual Encoder Model*** : Distill a dual encoder teacher model (BERT) to dual encoder student model(DistillBERT) for document ranking. [Code Link](./models/distil_dual_ranker.py).

5. ***Curriculum Learning*** : Fine-Tuning a BERT model for ranking using **Curriculum Learning** approach. This implementation focuses on difficulty and pacing as described in the [Paper Link](https://arxiv.org/pdf/1912.08555). [Code Link](./models/ranking_curriculum_learning.py)

## Text Generation using LLM
1. ***Basic fine-tuning***: Basic example code to fine-tune a small generative model on prompt-response toy data. Ideal for those starting with fine-tuning ***Generative Models***. [Code link](./models/genai_train_prompt_response.py). 

2. ***Fine-Tuning using Wiki Dataset***: Fine-tune a generative model on WIKI dataset using casual approach for ***Text Generation*** task. ou can substitute this dataset with your own data to tailor the model to specific domains. [Code Link](./models/genai_train_wiki.py)

3. ***Model Distilled***: Model Distillation from a teacher model(GPT2-medium) to Student Model(GPT-2). This approach, similar to the one described above, is useful for creating smaller, more efficient models. [Code Link](./models/genai_distill.py)

4. ***RAG Approach***:  Implementing a simple Retrieval-Augmented Generation (RAG) approach where the input is a query and the retrieved relevant documents, with the output being the query intent. [Code Link](./models/genai_opeanai_intent_gen.py)

5. ***RAG for Question Answering***: Implementing a RAG approach for question answering, where we retrieve documents given a query using BM25 that are then used by an LLM to generate answers based on the retrieved context. [Code Link](./models/rag_question_answer.py)

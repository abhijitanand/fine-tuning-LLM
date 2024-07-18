# Fine-tuning LLM using different approaches and for differnt tasks.

In the repository you will find code on different ways to fine-tune a LLM for different tasks. This only contains basic implementation to kick start your work. You have to have prior knowledge about different approaches to know about their pros and limitaions.

## Ranking Task
1. ***BERT(MonoBERT)-Single Encoder*** :  Fine-tuning a BERT model on the MS MARCO dataset for Document Ranking task(bert_ranking.py). This is classical transformer based ranking using query and document as input.
2. ***Dual Encoder Model*** : Fine-tuning BERT based Dual Encoder model on Ms-Marco dataset for Document Ranking task(dual_encoder_ranking.py). Dense retrieval and ranking technique, used in model like TCT-Colber(https://arxiv.org/pdf/2010.11386).
3. ***Distilled Model*** : Distill a teacher model (BERT) to student model(DistillBERT) for document ranking. Coming up with a smaller model using a larger model to boost efficiency. Eg; DistillBERT(https://arxiv.org/pdf/1910.01108).
4. ***Distilled Dual Encoder Model*** : Distill a dual encoder teacher model (BERT) to dual encoder student model(DistillBERT) for document ranking(distil_dual_ranker).
5. ***Curriculum Learning*** : Fine-Tuning a BERT model for ranking using **Curriculum Learning** approach(ranking_curriculum_learning.py). Important to know there are many variations of this approach I only use the difficulty and pacing described in https://arxiv.org/pdf/1912.08555.

## Text Generation using LLM
1. ***Basic fine-tuning***: Basic code to fine-tune a small generative model on prompt, response toy data(genai_train_prompt_response.py). Try to adjust this code if you are starting with fine-tuning ***Genrative models**.
2. ***Fine-Tuning using Wiki Dataset***: Fine-tune a generative model on WIKI dataset using casual approach for ***Text Generation*** task(genai_train_wiki.py). You can swap with your own data to make model more domain/data specific.
3. ***Model Distilled***: Model Distillation from a teacher model(GPT2-medium) to Student Model(GPT-2)(genai_distill.py). Similar to distillation approach described above. This is really helpful when we want to use smaller efficeint models.
4. ***RAG Approach***: Simple RAG approach where the input is a query and retrieved relevant document and the output expected is query intent(genai_opeanai_intent_gen.py).

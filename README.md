# Code For Training LLM.

In the repository you will find code on different ways to fine-tune a LLM for different tasks.
## Ranking Task
1. ***BERT(MonoBERT)-Single Encoder*** :  Fine-tuning a BERT model on the MS MARCO dataset for Document Ranking task.
2. ***Dual Encoder Model*** : Fine-tuning BERT based Dual Encoder model on Ms-Marco dataset for Document Ranking task.
3. ***Distilled Model*** : Distill a teacher model (BERT) to student model(DistillBERT) for document ranking.
4. ***Distilled Dual Encoder Model*** : Distill a dual encoder teacher model (BERT) to dual encoder student model(DistillBERT) for document ranking.

## Text Generation using LLM
1. ***Basic fine-tuning***: Basic code to train a small casual model on prompt, response toy data.
2. ***Fine-Tuning using Wiki Dataset***: Train on WIKI dataset using casual approach for ***Text Generation*** task.
3. ***Model Distilled***: Model Distillation from a teacher model(GPT2-medium) to Student Model(GPT-2).

# LLM based RAGAS-like Finetuning Embedding Model Performance Summary

This is from code in ragas_finetune_evaluate.py (confusing name - its ragas like)

We compared the original `all-MiniLM-L6-v2` model with the its finetuned version using the LLM based generated query/prompt data (`finetuned-prompt-retriever`).
Hugging Face: [geetach/finetuned-prompt-retriever]https://huggingface.co/geetach/finetuned-prompt-retriever

The finetuning **did improve** the model's performance across all measured aspects.

Here's a breakdown:

*   **Overall Similarity:** The finetuned model was slightly better (around **3.1%** relative improvement) at understanding the connection between a search query and the relevant writing prompt.
*   **Simulated Context Precision:** This measures how well the returned prompt actually matches the *intent* of the search query. The finetuned model showed a **2.4%** relative improvement here.
*   **Simulated Context Recall:** This measures how much of the *relevant information* from the prompt is captured in relation to the query. The finetuned model improved by **3.1%** here (the highest relative gain).
*   **Simulated Faithfulness:** This estimates how well the model's understanding sticks to the actual content of the prompt. There was a **2.0%** relative improvement.
*   **Simulated Answer Relevancy:** This checks if the matched prompt is relevant to the search query. The finetuned model improved by **2.8%** here.

### In simple terms:

The finetuning process made the model slightly better at matching relevant writing prompts to potential search queries. While the improvements are modest (around 2-3%), they are consistently positive across all simulated metrics. This indicates the model learned *something* useful from the specific data you provided, making it a slightly better "prompt retriever" than the general-purpose base model.

Here is what the LLM-based Simulated-Ragas code does:

- Uses an LLM's help to generate the golden dataset
- It takes that query, context, and a SentenceTransformer model.
- It calculates the cosine similarity between the embeddings of the query and the context using the provided sentence transformer model.
- Then, it uses this raw similarity score to derive other "simulated" metrics (like simulated_context_precision, simulated_context_recall, simulated_faithfulness, simulated_answer_relevancy). These derivations are custom formulas within that function, essentially scaling or slightly perturbing the base similarity score to produce numbers that are analogous to what RAGAS might measure, but based purely on retriever performance (embedding similarity) rather than full generative QA evaluation.


## Model Details
- Base Model: `all-MiniLM-L6-v2`
- Finetuned Model: https://huggingface.co/geetach/finetuned-prompt-retriever

## Metrics Comparison (Simulated RAGAS)

| Metric | Base Model | Finetuned Model | Absolute Improvement | Relative Improvement |
|--------|------------|-----------------|---------------------|--------------------|
| similarity | 0.680 | 0.701 | 0.021 | 3.1% |
| simulated_context_precision | 0.545 | 0.558 | 0.013 | 2.4% |
| simulated_context_recall | 0.613 | 0.632 | 0.019 | 3.1% |
| simulated_faithfulness | 0.481 | 0.491 | 0.010 | 2.0% |
| simulated_answer_relevancy | 0.578 | 0.594 | 0.016 | 2.8% |

## Analysis

- Highest relative improvement in **simulated_context_recall**: 3.1%

### Simulated Faithfulness Analysis
- Base model simulated_faithfulness: 0.481
- Finetuned model simulated_faithfulness: 0.491
- Absolute improvement: 0.010
- Relative improvement: 2.0%


# Some non LLM based fun I had with doing heuristic algorithms based eval pipeline.

## Overview
This is code in generate_prompt_eval_data.py (golden test dataset) and prompt_evaluation.py (compare a few candidates for what should be our base model) and just for kicks and hipefully NOT added confusion: finetune_embeddings.py to finetune the chosen base model using the non llm golden test dataset we just generated.

## Comparsion of base models

| Metric | all-MiniLM-L6-v2 | all-mpnet-base-v2 | Difference | Better Model |
|--------|------------------|-------------------|------------|--------------|
| Precision@1 | 0.767 | 0.800 | +0.033 | all-mpnet-base-v2 |
| Precision@3 | 0.578 | 0.578 | 0.000 | Tie |
| Precision@5 | 0.427 | 0.560 | +0.133 | all-mpnet-base-v2 |
| MRR | 0.133 | 0.133 | 0.000 | Tie |
| NDCG@5 | 0.133 | 0.133 | 0.000 | Tie |
| Context Precision | 0.370 | 0.329 | -0.041 | all-MiniLM-L6-v2 |
| Context Recall | 0.359 | 0.315 | -0.044 | all-MiniLM-L6-v2 |
| Semantic Similarity | 0.342 | 0.292 | -0.050 | all-MiniLM-L6-v2 |
| Faithfulness | 0.387 | 0.350 | -0.037 | all-MiniLM-L6-v2 |
| Answer Relevancy | 0.363 | 0.323 | -0.040 | all-MiniLM-L6-v2 |
| Faithfulness Impact | 0.007 | -0.011 | -0.018 | all-MiniLM-L6-v2 |

## Results of this extra for-fun finetuning

The fine-tuned model showed significant improvements over the base model:

| Metric | Base Model | Fine-tuned Model | Improvement |
|--------|------------|------------------|-------------|
| Precision@1 | 0.547 | 0.612 | +11.9% |
| Precision@3 | 0.483 | 0.531 | +9.9% |
| Precision@5 | 0.459 | 0.502 | +9.4% |
| MRR | 0.193 | 0.224 | +16.1% |
| NDCG@5 | 0.216 | 0.248 | +14.8% |

- Hugging Face Hub: [geetach/prompt-retrieval-midterm-finetuned](https://huggingface.co/geetach/prompt-retrieval-midterm-finetuned)

## Why did I not use THIS finetuned model?

 (it shows great improvement) BUT BUT - because this datset did not use LLMs and I think the LLM generated data is more robust.


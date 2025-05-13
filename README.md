## RAGAS based Finetuning Embedding Model Performance Summary

We compared the original `all-MiniLM-L6-v2` model with the its finetuned version using the ragas generated query/prompt data (`finetuned-prompt-retriever`).

The finetuning **did improve** the model's performance across all measured aspects.

Here's a breakdown:

*   **Overall Similarity:** The finetuned model was slightly better (around **3.1%** relative improvement) at understanding the connection between a search query and the relevant writing prompt.
*   **Simulated Context Precision:** This measures how well the returned prompt actually matches the *intent* of the search query. The finetuned model showed a **2.4%** relative improvement here.
*   **Simulated Context Recall:** This measures how much of the *relevant information* from the prompt is captured in relation to the query. The finetuned model improved by **3.1%** here (the highest relative gain).
*   **Simulated Faithfulness:** This estimates how well the model's understanding sticks to the actual content of the prompt. There was a **2.0%** relative improvement.
*   **Simulated Answer Relevancy:** This checks if the matched prompt is relevant to the search query. The finetuned model improved by **2.8%** here.

### In simple terms:

The finetuning process made the model slightly better at matching relevant writing prompts to potential search queries. While the improvements are modest (around 2-3%), they are consistently positive across all simulated metrics. This indicates the model learned *something* useful from the specific data you provided, making it a slightly better "prompt retriever" than the general-purpose base model.

## NON RAGAS based Finetuning Embedding Model Performance Summary
Ok, below are some findings on when I did a RAGAS like evaluation with custom code. So, I could control for my use case better.

## Model Details
- Base Model: `all-MiniLM-L6-v2`
- Finetuned Model: `ragas_results/finetuned_all-MiniLM-L6-v2_20250512_152546`

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


Fine-Tuning Embedding Models for Creative Writing Prompts

## Overview
This report summarizes the process and results of fine-tuning the `all-MiniLM-L6-v2` embedding model for improved retrieval of creative writing prompts based on keyword queries.

## Process Steps

1. **Data Preparation**
   - Used 532 creative writing prompts with genre and theme tags
   - Generated 932 training examples using positive and negative pairs
   - Created keyword-based queries from prompt content and tags

2. **Model Selection**
   - Base Model: `all-MiniLM-L6-v2`
   - Chosen for its balance of performance and efficiency
   - Initial evaluation showed good baseline performance for semantic search

3. **Fine-Tuning**
   - Training Parameters:
     - Epochs: 3
     - Batch Size: 16
     - Learning Rate: 2e-5
     - Loss Function: Multiple Negatives Ranking Loss
   - Training Time: ~60 seconds on M1 Mac (MPS)
   - Final Training Loss: 1.33

4. **Evaluation**
   - Metrics:
     - Precision@1, Precision@3, Precision@5
     - Mean Reciprocal Rank (MRR)
     - Normalized Discounted Cumulative Gain (NDCG@5)
   - Test Dataset: Same prompts with keyword-based queries

## Results

The fine-tuned model showed significant improvements over the base model:

| Metric | Base Model | Fine-tuned Model | Improvement |
|--------|------------|------------------|-------------|
| Precision@1 | 0.547 | 0.612 | +11.9% |
| Precision@3 | 0.483 | 0.531 | +9.9% |
| Precision@5 | 0.459 | 0.502 | +9.4% |
| MRR | 0.193 | 0.224 | +16.1% |
| NDCG@5 | 0.216 | 0.248 | +14.8% |

## Recommendation

The fine-tuned model demonstrates substantial improvements in retrieval performance, particularly in top-1 precision and mean reciprocal rank. We recommend using the fine-tuned model for production use.

### How to Use

The fine-tuned model is available on Hugging Face Hub:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('geetach/prompt-retrieval-midterm-finetuned')
```

### Model Location
- Hugging Face Hub: [geetach/prompt-retrieval-midterm-finetuned](https://huggingface.co/geetach/prompt-retrieval-midterm-finetuned)
- Local Path: `models/finetuned_all-MiniLM-L6-v2_20250511_235648`

## Next Steps
1. Monitor model performance in production
2. Collect user feedback for further improvements
3. Consider periodic retraining with new prompts 

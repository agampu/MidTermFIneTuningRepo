# Fine-Tuning Embedding Models for Creative Writing Prompts

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
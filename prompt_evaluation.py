import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
import torch
from IPython.display import display, HTML
import json
from datetime import datetime
import os

class PromptRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with a sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        self.prompt_embeddings = None
        self.prompts = None
    
    def index_prompts(self, prompts: List[str]):
        """Create embeddings for all prompts."""
        self.prompts = prompts
        self.prompt_embeddings = self.model.encode(prompts, convert_to_tensor=True)
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve top k most relevant prompts for a query."""
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(
            query_embedding.cpu().numpy().reshape(1, -1),
            self.prompt_embeddings.cpu().numpy()
        )[0]
        
        # Get top k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            results.append({
                'prompt': self.prompts[idx],
                'score': float(similarities[idx])
            })
        
        return results

def evaluate_retriever(retriever: PromptRetriever, eval_data: pd.DataFrame):
    """Evaluate the retriever using various metrics."""
    results = {
        'precision@1': [],
        'precision@3': [],
        'precision@5': [],
        'mrr': [],  # Mean Reciprocal Rank
        'ndcg@5': []  # Normalized Discounted Cumulative Gain
    }
    
    unique_prompts = eval_data['context'].unique()
    retriever.index_prompts(unique_prompts)
    
    for _, row in eval_data.iterrows():
        query = row['query']
        relevant_prompt = row['context']
        relevance = row['relevance']
        
        # Get top 5 results
        retrieved = retriever.retrieve(query, k=5)
        
        # Calculate metrics
        # Precision@k
        for k in [1, 3, 5]:
            top_k = retrieved[:k]
            hits = sum(1 for r in top_k if r['prompt'] == relevant_prompt)
            precision = hits / k if relevance == 1.0 else 1.0 if hits == 0 else 0.0
            results[f'precision@{k}'].append(precision)
        
        # MRR
        rank = 0
        for i, r in enumerate(retrieved, 1):
            if r['prompt'] == relevant_prompt:
                rank = i
                break
        mrr = 1/rank if rank > 0 and relevance == 1.0 else 0.0
        results['mrr'].append(mrr)
        
        # NDCG@5
        dcg = 0
        idcg = sum(1/np.log2(i+1) for i in range(1, min(6, 2)))  # ideal DCG with one relevant doc
        for i, r in enumerate(retrieved, 1):
            if r['prompt'] == relevant_prompt and relevance == 1.0:
                dcg += 1/np.log2(i+1)
        ndcg = dcg/idcg if idcg > 0 else 0.0
        results['ndcg@5'].append(ndcg)
    
    # Calculate average metrics
    return {metric: np.mean(values) for metric, values in results.items()}

def test_query(retriever: PromptRetriever, query: str):
    """Test a single query and display results."""
    results = retriever.retrieve(query, k=3)
    return {
        'query': query,
        'results': [
            {
                'rank': i,
                'score': result['score'],
                'prompt': result['prompt']
            }
            for i, result in enumerate(results, 1)
        ]
    }

def save_evaluation_results(metrics_by_model, example_results, dataset_info):
    """Save evaluation results to files."""
    # Create results directory if it doesn't exist
    os.makedirs('evaluation_results', exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save raw results as JSON
    results = {
        'timestamp': timestamp,
        'dataset_info': dataset_info,
        'model_metrics': metrics_by_model,
        'example_queries': example_results
    }
    
    with open(f'evaluation_results/eval_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate markdown report
    report = f"""# Prompt Retrieval Model Evaluation Report
    
## Dataset Information
- Total examples: {dataset_info['total_examples']}
- Evaluation date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Dataset composition: {dataset_info['composition']}

## Model Comparison

| Model | Precision@1 | Precision@3 | Precision@5 | MRR | NDCG@5 |
|-------|------------|-------------|-------------|-----|--------|
"""
    
    for model_name, metrics in metrics_by_model.items():
        report += f"| {model_name} | {metrics['precision@1']:.3f} | {metrics['precision@3']:.3f} | {metrics['precision@5']:.3f} | {metrics['mrr']:.3f} | {metrics['ndcg@5']:.3f} |\n"
    
    # Add model selection rationale
    best_model = max(metrics_by_model.items(), key=lambda x: x[1]['mrr'])[0]
    report += f"""
## Selected Model: {best_model}

### Selection Rationale
1. **Best Overall Performance**: {best_model} achieved the highest Mean Reciprocal Rank (MRR) of {metrics_by_model[best_model]['mrr']:.3f}, indicating better ranking of relevant results.
2. **Consistent Precision**: Maintains strong precision across different k values:
   - P@1: {metrics_by_model[best_model]['precision@1']:.3f}
   - P@3: {metrics_by_model[best_model]['precision@3']:.3f}
   - P@5: {metrics_by_model[best_model]['precision@5']:.3f}
3. **NDCG Performance**: Shows strong ranking quality with NDCG@5 of {metrics_by_model[best_model]['ndcg@5']:.3f}

### Example Queries and Results
"""
    
    for example in example_results:
        report += f"""
#### Query: "{example['query']}"
"""
        for result in example['results']:
            report += f"""
{result['rank']}. **Score: {result['score']:.3f}**
   ```
   {result['prompt']}
   ```
"""
    
    report += """
## Evaluation Metrics Explained
- **Precision@k**: The proportion of relevant results in the top k retrieved items
- **MRR (Mean Reciprocal Rank)**: Average of 1/rank of the first relevant result
- **NDCG@5 (Normalized Discounted Cumulative Gain)**: Measures ranking quality considering position

## Next Steps and Recommendations
1. Consider fine-tuning the selected model on domain-specific data
2. Implement post-processing rules to boost genre tag matches
3. Add relevance feedback mechanism for continuous improvement
"""
    
    with open(f'evaluation_results/evaluation_report_{timestamp}.md', 'w') as f:
        f.write(report)
    
    return timestamp

def main():
    # Load the evaluation dataset
    eval_data = pd.read_csv('data/prompt_evaluation_dataset.csv')
    print(f"Loaded {len(eval_data)} evaluation examples")
    
    # Dataset info for the report
    dataset_info = {
        'total_examples': len(eval_data),
        'composition': {
            'positive_examples': len(eval_data[eval_data['relevance'] == 1.0]),
            'negative_examples': len(eval_data[eval_data['relevance'] == 0.0])
        }
    }
    
    # Initialize retriever with different models to compare
    models = [
        "all-MiniLM-L6-v2",  # Fast, lightweight model
        "all-mpnet-base-v2",  # More powerful, but slower
        "multi-qa-mpnet-base-dot-v1"  # Specifically tuned for retrieval
    ]
    
    print("\nEvaluating different embedding models...")
    metrics_by_model = {}
    
    for model_name in models:
        print(f"\nEvaluating {model_name}")
        retriever = PromptRetriever(model_name)
        metrics = evaluate_retriever(retriever, eval_data)
        metrics_by_model[model_name] = metrics
        
        print("\nResults:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.3f}")
    
    # Use the best performing model for example queries
    best_model = max(metrics_by_model.items(), key=lambda x: x[1]['mrr'])[0]
    print(f"\nUsing best model: {best_model}")
    
    retriever = PromptRetriever(best_model)
    unique_prompts = eval_data['context'].unique()
    retriever.index_prompts(unique_prompts)
    
    # Test example queries
    test_queries = [
        "Find a fantasy story about magic and potions",
        "A horror story with a supernatural twist",
        "Romance in a bookstore",
        "Science fiction about time travel"
    ]
    
    example_results = []
    print("\nTesting example queries:")
    for query in test_queries:
        results = test_query(retriever, query)
        example_results.append(results)
        print(f"\nQuery: {results['query']}")
        for r in results['results']:
            print(f"\n{r['rank']}. Score: {r['score']:.3f}")
            print(f"Prompt: {r['prompt']}")
    
    # Save results and generate report
    timestamp = save_evaluation_results(metrics_by_model, example_results, dataset_info)
    print(f"\nEvaluation results saved to evaluation_results/eval_results_{timestamp}.json")
    print(f"Evaluation report saved to evaluation_results/evaluation_report_{timestamp}.md")

if __name__ == "__main__":
    main() 
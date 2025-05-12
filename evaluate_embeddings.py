import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from ragas.metrics import faithfulness, answer_relevancy, context_relevancy
from datasets import Dataset
import torch

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

def main():
    # Load the evaluation dataset
    eval_data = pd.read_csv('data/prompt_evaluation_dataset.csv')
    
    # Initialize retriever with different models to compare
    models = [
        "all-MiniLM-L6-v2",  # Fast, lightweight model
        "all-mpnet-base-v2",  # More powerful, but slower
        "multi-qa-mpnet-base-dot-v1"  # Specifically tuned for retrieval
    ]
    
    print("Evaluating different embedding models...")
    for model_name in models:
        print(f"\nEvaluating {model_name}")
        retriever = PromptRetriever(model_name)
        metrics = evaluate_retriever(retriever, eval_data)
        
        print("\nResults:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.3f}")

if __name__ == "__main__":
    main() 
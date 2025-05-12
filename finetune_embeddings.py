import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import random
import re
from typing import List, Tuple, Dict
import os
import logging
from datetime import datetime
import json
from huggingface_hub import HfApi, create_repo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_tags_and_content(prompt: str) -> Tuple[List[str], str]:
    """Extract tags and content from a prompt."""
    tags = re.findall(r'<([^>]+)>', prompt)
    # Remove closing tags and duplicates
    tags = [tag for tag in tags if not tag.startswith('/')]
    tags = list(dict.fromkeys(tags))  # Remove duplicates while preserving order
    
    # Get the actual content after 'prompt'
    content = prompt.split('prompt ')[-1].strip()
    return tags, content

def generate_alternative_queries(tags: List[str], content: str) -> List[str]:
    """Generate alternative queries for a prompt based on its tags and content."""
    queries = []
    
    # Genre-based queries
    genre_tags = [tag for tag in tags if tag in ['Fantasy', 'Science Fiction', 'Horror', 'Mystery', 'Romance']]
    if genre_tags:
        queries.append(f"Find a {genre_tags[0]} writing prompt")
        
        # Genre + theme combinations
        other_tags = [tag for tag in tags if tag not in genre_tags and tag != 'prompt']
        if other_tags:
            queries.append(f"Find a {genre_tags[0]} story about {other_tags[0].lower()}")
    
    # Keyword-based queries
    words = content.split()
    key_nouns = [word for word in words if len(word) > 4 and word.isalpha()]
    if key_nouns:
        # Select 1-3 random keywords
        num_keywords = min(len(key_nouns), random.randint(1, 3))
        selected_keywords = random.sample(key_nouns, num_keywords)
        queries.append(f"Find a writing prompt about {' and '.join(selected_keywords).lower()}")
    
    return queries

def create_training_examples(df: pd.DataFrame, num_negatives: int = 3) -> List[InputExample]:
    """Create training examples for fine-tuning."""
    examples = []
    all_prompts = df['context'].unique()
    
    for _, row in df.iterrows():
        if row['relevance'] != 1.0:  # Skip negative examples from the dataset
            continue
        
        query = row['query']
        positive_prompt = row['context']
        
        # Generate negative examples (prompts that don't match the query)
        negative_prompts = random.sample([p for p in all_prompts if p != positive_prompt], num_negatives)
        
        # Create an example with one positive and multiple negatives
        examples.append(InputExample(
            texts=[query, positive_prompt] + negative_prompts
        ))
        
        # Generate alternative queries for data augmentation
        tags, content = extract_tags_and_content(positive_prompt)
        alt_queries = generate_alternative_queries(tags, content)
        
        for alt_query in alt_queries:
            examples.append(InputExample(
                texts=[alt_query, positive_prompt] + negative_prompts
            ))
    
    return examples

def evaluate_model(model: SentenceTransformer, test_df: pd.DataFrame) -> Dict[str, float]:
    """Evaluate model performance on test dataset."""
    queries = test_df['query'].tolist()
    contexts = test_df['context'].unique().tolist()
    relevance_dict = dict(zip(test_df['query'] + '|||' + test_df['context'], test_df['relevance']))
    
    # Compute embeddings
    query_embeddings = model.encode(queries, convert_to_tensor=True)
    context_embeddings = model.encode(contexts, convert_to_tensor=True)
    
    # Compute similarity scores
    similarities = query_embeddings @ context_embeddings.T
    
    # Calculate metrics
    precision_at_1 = 0
    precision_at_3 = 0
    precision_at_5 = 0
    mrr = 0
    ndcg_at_5 = 0
    
    for i, query in enumerate(queries):
        # Get top k results
        scores = similarities[i].cpu().numpy()
        top_indices = np.argsort(scores)[::-1]
        
        # Calculate precision@k
        for k in [1, 3, 5]:
            top_k_contexts = [contexts[idx] for idx in top_indices[:k]]
            relevant = sum(1 for ctx in top_k_contexts if relevance_dict.get(f"{query}|||{ctx}", 0) == 1.0)
            if k == 1:
                precision_at_1 += relevant / k
            elif k == 3:
                precision_at_3 += relevant / k
            else:
                precision_at_5 += relevant / k
        
        # Calculate MRR
        for rank, idx in enumerate(top_indices, 1):
            if relevance_dict.get(f"{query}|||{contexts[idx]}", 0) == 1.0:
                mrr += 1 / rank
                break
        
        # Calculate NDCG@5
        dcg = 0
        idcg = 0
        top_5_contexts = [contexts[idx] for idx in top_indices[:5]]
        relevances = [relevance_dict.get(f"{query}|||{ctx}", 0) for ctx in top_5_contexts]
        
        # Get all relevance scores for this query
        query_relevances = [
            relevance_dict.get(f"{query}|||{ctx}", 0)
            for ctx in contexts
        ]
        ideal_relevances = sorted(query_relevances, reverse=True)[:5]
        
        for i, (rel, ideal_rel) in enumerate(zip(relevances, ideal_relevances), 1):
            dcg += rel / np.log2(i + 1)
            idcg += ideal_rel / np.log2(i + 1)
        
        if idcg > 0:
            ndcg_at_5 += dcg / idcg
    
    num_queries = len(queries)
    return {
        'precision@1': precision_at_1 / num_queries,
        'precision@3': precision_at_3 / num_queries,
        'precision@5': precision_at_5 / num_queries,
        'mrr': mrr / num_queries,
        'ndcg@5': ndcg_at_5 / num_queries
    }

def compare_models(base_model_path: str, finetuned_model_path: str, test_data_path: str) -> Dict:
    """Compare base and finetuned models and save results."""
    # Load models
    base_model = SentenceTransformer(base_model_path)
    finetuned_model = SentenceTransformer(finetuned_model_path)
    
    # Load test data
    test_df = pd.read_csv(test_data_path)
    
    # Evaluate both models
    logger.info("Evaluating base model...")
    base_metrics = evaluate_model(base_model, test_df)
    
    logger.info("Evaluating finetuned model...")
    finetuned_metrics = evaluate_model(finetuned_model, test_df)
    
    # Calculate improvements
    improvements = {
        metric: {
            'base': base_metrics[metric],
            'finetuned': finetuned_metrics[metric],
            'improvement': finetuned_metrics[metric] - base_metrics[metric],
            'relative_improvement': (finetuned_metrics[metric] - base_metrics[metric]) / base_metrics[metric] * 100
        }
        for metric in base_metrics.keys()
    }
    
    # Create comparison report
    comparison = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'base_model': base_model_path,
        'finetuned_model': finetuned_model_path,
        'metrics': improvements
    }
    
    # Save results
    os.makedirs('evaluation_results/finetuning_comparison', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save JSON results
    json_path = f'evaluation_results/finetuning_comparison/comparison_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Generate markdown report
    markdown_path = f'evaluation_results/finetuning_comparison/comparison_{timestamp}.md'
    with open(markdown_path, 'w') as f:
        f.write('# Model Finetuning Comparison Report\n\n')
        f.write(f'Generated on: {comparison["timestamp"]}\n\n')
        f.write('## Model Details\n')
        f.write(f'- Base Model: `{base_model_path}`\n')
        f.write(f'- Finetuned Model: `{finetuned_model_path}`\n\n')
        f.write('## Performance Metrics\n\n')
        f.write('| Metric | Base Model | Finetuned Model | Absolute Improvement | Relative Improvement |\n')
        f.write('|--------|------------|-----------------|---------------------|--------------------|\n')
        
        for metric, values in improvements.items():
            f.write(f'| {metric} | {values["base"]:.3f} | {values["finetuned"]:.3f} | ')
            f.write(f'{values["improvement"]:.3f} | {values["relative_improvement"]:.1f}% |\n')
    
    logger.info(f"Comparison results saved to {json_path} and {markdown_path}")
    return comparison

def push_to_huggingface(
    model_path: str,
    hf_token: str,
    repo_name: str,
    organization: str = None
) -> str:
    """Push the finetuned model to Hugging Face Hub."""
    api = HfApi()
    
    # Create the full repository name
    if organization:
        repo_id = f"{organization}/{repo_name}"
    else:
        repo_id = repo_name
    
    try:
        # Create the repository
        logger.info(f"Creating repository: {repo_id}")
        api.create_repo(
            repo_id=repo_id,
            token=hf_token,
            private=False,
            repo_type="model",
            exist_ok=True  # Won't fail if repo already exists
        )
        
        # Push the model to the hub
        logger.info(f"Uploading model to {repo_id}")
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model",
            token=hf_token
        )
        
        logger.info(f"Model successfully pushed to Hugging Face Hub: {repo_id}")
        return f"https://huggingface.co/{repo_id}"
    except Exception as e:
        logger.error(f"Error pushing to Hugging Face Hub: {str(e)}")
        raise

def fine_tune_model(
    base_model_name: str = "all-MiniLM-L6-v2",
    train_data_path: str = "data/prompt_evaluation_dataset.csv",
    output_path: str = "models",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    num_negatives: int = 3
):
    """Fine-tune the embedding model on our prompt dataset."""
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_save_path = os.path.join(output_path, f"finetuned_{base_model_name}_{timestamp}")
    os.makedirs(model_save_path, exist_ok=True)
    
    # Load base model
    logger.info(f"Loading base model: {base_model_name}")
    model = SentenceTransformer(base_model_name)
    logger.info(f"Model loaded successfully. Device: {model.device}")
    
    # Load and prepare training data
    logger.info("Loading training data from: " + train_data_path)
    train_df = pd.read_csv(train_data_path)
    logger.info(f"Loaded {len(train_df)} training examples")
    
    logger.info("Creating training examples with negatives...")
    train_examples = create_training_examples(train_df, num_negatives=num_negatives)
    logger.info(f"Created {len(train_examples)} training examples with negatives")
    
    # Create data loader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    logger.info(f"Created DataLoader with batch size {batch_size}")
    
    # Define the loss function (Multiple Negatives Ranking Loss)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    logger.info("Using MultipleNegativesRankingLoss for training")
    
    # Configure training arguments
    logger.info(f"Starting training for {epochs} epochs with learning rate {learning_rate}")
    logger.info(f"Model will be saved to: {model_save_path}")
    
    # Train the model
    logger.info("Starting training...")
    warmup_steps = int(len(train_dataloader) * 0.1)  # 10% of training data for warmup
    logger.info(f"Using {warmup_steps} warmup steps")
    
    try:
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            optimizer_params={'lr': learning_rate},
            output_path=model_save_path,
            show_progress_bar=True,
            callback=lambda step, epoch, score: logger.info(f"Step: {step}, Epoch: {epoch}, Score: {score}")
        )
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise
    
    logger.info(f"Training completed. Model saved to {model_save_path}")
    return model_save_path

if __name__ == "__main__":
    import argparse
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )
    
    parser = argparse.ArgumentParser(description='Fine-tune and evaluate embedding model')
    parser.add_argument('--hf_token', type=str, help='Hugging Face API token')
    parser.add_argument('--repo_name', type=str, help='Repository name for the model')
    parser.add_argument('--organization', type=str, default=None, help='Hugging Face organization name')
    args = parser.parse_args()
    
    try:
        # Fine-tune the model
        logger.info("Starting fine-tuning process...")
        model_path = fine_tune_model(
            base_model_name="all-MiniLM-L6-v2",
            train_data_path="data/prompt_evaluation_dataset.csv",
            epochs=3,
            batch_size=16
        )
        
        # Compare models and save results
        logger.info("Starting model comparison...")
        comparison_results = compare_models(
            base_model_path="all-MiniLM-L6-v2",
            finetuned_model_path=model_path,
            test_data_path="data/prompt_evaluation_dataset.csv"
        )
        
        # Push to Hugging Face if token is provided
        if args.hf_token and args.repo_name:
            logger.info("Pushing model to Hugging Face Hub...")
            try:
                hf_url = push_to_huggingface(
                    model_path=model_path,
                    hf_token=args.hf_token,
                    repo_name=args.repo_name,
                    organization=args.organization
                )
                logger.info(f"Model pushed to Hugging Face Hub: {hf_url}")
            except Exception as e:
                logger.error(f"Failed to push to Hugging Face Hub: {str(e)}")
        
        logger.info(f"Fine-tuned model saved to: {model_path}")
        logger.info("To use the fine-tuned model, update the model_name parameter in PromptRetriever to:")
        logger.info(f"model_name = '{model_path}'")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise 
#!/usr/bin/env python3
"""
RAGAS-based Synthetic Data Generation, Model Finetuning, and Evaluation

This script:
1. Generates a synthetic test set using RAGAS's TestsetGenerator
2. Finetunes a sentence transformer model using the generated data
3. Evaluates both the base and finetuned models using simulated RAGAS metrics
4. Pushes the finetuned model to Hugging Face Hub
"""

import os
from dotenv import load_dotenv # Added for .env support
load_dotenv() # Added to load .env file

import json
import random
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Any
from pathlib import Path
from tqdm import tqdm

# RAGAS imports
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
# Note: We are not using ragas.metrics for evaluation in this script, 
# as we focus on retriever performance and simulate metrics based on similarity.
# from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall 

# LangChain imports
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# Sentence transformers for finetuning
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch

# Hugging Face Hub import
from huggingface_hub import HfApi

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_tags_and_prompt(text: str) -> Tuple[List[str], str]:
    """Extract tags and prompt text from a prompt entry."""
    import re
    tags = re.findall(r'<([^>]+)>', text)
    # Remove tag names that are the same as their content
    tags = [tag for tag in tags if not tag.endswith(f" </{tag}>")]
    # Clean up tags by removing closing tags
    tags = [tag for tag in tags if not tag.startswith('/')]
    
    # Extract the actual prompt text
    prompt_text = text.split('prompt ')[-1].strip() if 'prompt' in text else text.strip()
    return tags, prompt_text

def load_prompts(filename: str) -> List[str]:
    """Load prompts from a text file."""
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def prompts_to_documents(prompts: List[str]) -> List[Document]:
    """Convert prompts to LangChain documents with metadata."""
    documents = []
    
    for prompt in prompts:
        tags, content = extract_tags_and_prompt(prompt)
        
        # Extract genres and themes from tags
        genres = [tag for tag in tags if tag in ['Fantasy', 'Science Fiction', 'Horror', 'Mystery', 'Romance']]
        themes = [tag for tag in tags if tag not in genres and tag != 'prompt']
        
        # Create document with metadata
        doc = Document(
            page_content=content,
            metadata={
                "genres": genres,
                "themes": themes,
                "tags": tags,
                "original_text": prompt
            }
        )
        documents.append(doc)
    
    return documents

def generate_ragas_testset(
    documents: List[Document], 
    output_dir: str, 
    testset_size: int = 100,
    openai_api_key: str = None
) -> pd.DataFrame:
    """Generate a test set using a direct LLM call for queries."""
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    elif "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OpenAI API key must be provided either via --openai_api_key argument or OPENAI_API_KEY environment variable.")
        
    # Initialize LLM directly
    logger.info("Initializing Direct LLM for query generation...")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3) # Use moderate temperature

    # Define a prompt template for generating search queries
    query_generation_prompt_template = """
Given the following writing prompt text:
---------------------
{prompt_text}
---------------------

Generate 1 to 3 relevant, concise search queries or keywords that someone might use to find this specific prompt.
Focus on the key themes, objects, characters, or concepts. Do NOT make them overly long or conversational.
Output each query on a new line, without any numbering or bullet points.

Example Input Prompt:
<Fantasy> </Fantasy> <Magic> </Magic> <The Accidental Alchemist> </The Accidental Alchemist> prompt A clumsy apprentice baker accidentally creates a potion that grants temporary, unpredictable magical abilities instead of a perfect sourdough starter.

Example Output Queries:
accidental magic potion baker
clumsy baker magic potion
temporary magic baking accident

Your turn:
"""
    prompt = ChatPromptTemplate.from_template(query_generation_prompt_template)
    chain = prompt | llm 

    generated_data = []
    logger.info(f"Generating queries for {len(documents)} prompts...")

    for doc in tqdm(documents, desc="Generating Queries"):
        prompt_content = doc.page_content
        metadata = doc.metadata
        
        try:
            # Invoke the LLM to generate queries
            response = chain.invoke({"prompt_text": prompt_content})
            queries = [q.strip() for q in response.content.split('\n') if q.strip()]
            
            # Create entries for each generated query paired with the original context
            for query in queries:
                 if query: # Ensure query is not empty
                     generated_data.append({
                         "query": query,
                         "context": prompt_content, # Use original prompt as context
                         "metadata": metadata # Carry metadata along if needed later
                     })
        except Exception as e:
            logger.warning(f"Could not generate query for prompt: {prompt_content[:50]}... Error: {e}")
            continue # Skip this prompt if generation fails

    if not generated_data:
        logger.error("No query-context pairs were generated. Check LLM connection or prompt template.")
        return pd.DataFrame() # Return empty DataFrame

    # Convert generated data to DataFrame
    df = pd.DataFrame(generated_data)
    
    # Remove RAGAS-specific generation and processing
    # logger.info(f"Generating test set with {testset_size} examples using RAGAS...")
    # test_data_distribution = {"simple": 0.7, "reasoning": 0.3, "multi_context": 0.0} 
    # sdg_dataset = sdg_generator.generate_with_langchain_docs(
    #     documents, 
    #     testset_size=testset_size # Use testset_size and remove distributions
    # )
    # df = sdg_dataset.to_pandas() 
    # logger.info(f"Generated {len(df)} examples. Post-processing queries...")
    # df['query'] = df['question'].apply(lambda q: q.replace('?', '').strip()) 

    # Keep the general query cleaning steps if desired, though LLM might produce cleaner output now
    logger.info(f"Applying post-processing to {len(df)} generated queries...")
    question_starters = [
        "can you", "could you", "would you", "i need", "i want", 
        "i'm looking for", "show me", "find me", "give me",
        "what are", "what is", "how to", "where can", "tell me about",
        "provide", "generate", "recommend", "suggest" 
    ]
    
    # Remove question starters
    for starter in question_starters:
        # Case-insensitive check
        df['query'] = df['query'].apply(
            lambda q: q[len(starter):].strip() if q.lower().startswith(starter.lower()) else q
        )
    
    # Make first letter lowercase (search term style)
    df['query'] = df['query'].apply(
        lambda q: q[0].lower() + q[1:] if len(q) > 0 else q
    )

    # Rename 'ground_truth' to 'context' if needed, or ensure 'context' exists
    # These renames might not be necessary now as we explicitly create 'context'
    # if 'ground_truth' in df.columns and 'context' not in df.columns:
    #      df.rename(columns={'ground_truth': 'context'}, inplace=True)
    # elif 'answer' in df.columns and 'context' not in df.columns: # RAGAS might sometimes use 'answer'
    #      df.rename(columns={'answer': 'context'}, inplace=True)
         
    # Ensure 'context' column exists and handle potential missing values
    if 'context' not in df.columns:
         logger.error("Could not find 'context', 'ground_truth', or 'answer' column in generated dataset.")
         # Handle error or create a placeholder context if appropriate
         df['context'] = "" # Placeholder, adjust as needed
    else:
        # Ensure context is string and handle potential NaN/None values
        df['context'] = df['context'].astype(str).fillna('')


    logger.info(f"Post-processing complete. Final dataset size: {len(df)}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the processed dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"ragas_testset_processed_{timestamp}.csv")
    df.to_csv(output_file, index=False)
    
    # Save raw RAGAS dataset object if needed (it's already saved by generator potentially, check RAGAS docs)
    # Or save the initial DataFrame before processing
    # raw_json_file = os.path.join(output_dir, f"ragas_testset_raw_{timestamp}.json")
    # df_raw.to_json(raw_json_file, orient='records', indent=2) 
    
    logger.info(f"Processed test set saved to {output_file}")
    return df

def prepare_training_data(df: pd.DataFrame) -> List[InputExample]:
    """Prepare training data for sentence transformer finetuning."""
    examples = []
    
    # Get all unique contexts, ensuring they are strings
    all_contexts = df['context'].astype(str).unique().tolist()
    
    logger.info(f"Preparing training data from {len(df)} examples...")
    for _, row in df.iterrows():
        # Ensure query and context are strings
        query = str(row['query'])
        positive_context = str(row['context'])
        
        # Filter out empty positive contexts
        if not positive_context:
            continue

        # Generate negative examples (contexts that don't match the query)
        valid_negative_pool = [ctx for ctx in all_contexts if ctx != positive_context and ctx]
        
        # Ensure we have enough valid negative examples to sample from
        num_available_negatives = len(valid_negative_pool)
        num_negatives_to_sample = min(3, num_available_negatives)

        if num_negatives_to_sample > 0:
             negative_contexts = random.sample(
                 valid_negative_pool,
                 num_negatives_to_sample 
             )
        else:
             # Handle case where no valid negative contexts are available (e.g., only one unique context)
             # Option 1: Skip this example
             # continue 
             # Option 2: Use some default negative or repeat positive (less ideal)
             negative_contexts = [] # Or handle differently

        # Create an example only if we have a positive context
        examples.append(InputExample(
            texts=[query, positive_context] + negative_contexts,
            label=1.0  # Standard label for MultipleNegativesRankingLoss
        ))
        
    logger.info(f"Prepared {len(examples)} training examples.")
    if len(examples) == 0 and len(df) > 0:
        logger.warning("Prepared 0 training examples. Check input data and context content.")
        
    return examples

def finetune_model(
    base_model_name: str,
    training_data: List[InputExample],
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5
) -> str:
    """Finetune a sentence transformer model."""
    if not training_data:
        logger.error("Cannot finetune model: No training data provided.")
        return None
        
    # Load the model
    model = SentenceTransformer(base_model_name)
    logger.info(f"Loaded base model: {base_model_name}")
    
    # Create data loader
    train_dataloader = DataLoader(training_data, shuffle=True, batch_size=batch_size)
    
    # Use MultipleNegativesRankingLoss
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize model name for path
    sanitized_model_name = base_model_name.replace('/', '_') 
    model_output_path = os.path.join(output_dir, f"finetuned_{sanitized_model_name}_{timestamp}")
    os.makedirs(model_output_path, exist_ok=True)
    
    # Train the model
    logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=100,
        optimizer_params={'lr': learning_rate},
        output_path=model_output_path,
        show_progress_bar=True
    )
    
    logger.info(f"Training completed. Model saved locally to: {model_output_path}")
    
    # Add fine-tuning info to the model
    model_info = {
        'base_model': base_model_name,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'training_examples': len(training_data),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save model info
    with open(os.path.join(model_output_path, 'fine_tuning_info.json'), 'w') as f:
        json.dump(model_info, f, indent=2)
    
    return model_output_path

def push_to_huggingface(
    model_path: str, 
    hf_token: str, 
    repo_name: str, 
    organization: str = None
) -> str:
    """Push the finetuned model to Hugging Face Hub."""
    if not hf_token:
        logger.warning("Hugging Face token not provided. Skipping model push.")
        return None
        
    api = HfApi()
    
    # Create the full repository name
    if organization:
        repo_id = f"{organization}/{repo_name}"
    else:
        # Get username associated with the token
        try:
             user_info = api.whoami(token=hf_token)
             username = user_info['name']
             repo_id = f"{username}/{repo_name}"
        except Exception as e:
             logger.error(f"Could not automatically determine username from HF token: {e}")
             logger.error("Please provide organization or ensure token has read access to user info.")
             # Fallback: Use repo_name directly, assuming it's for the user's namespace
             repo_id = repo_name
             logger.warning(f"Attempting to push to repo: {repo_id}")

    try:
        # Create the repository
        logger.info(f"Creating or retrieving repository: {repo_id}")
        api.create_repo(
            repo_id=repo_id,
            token=hf_token,
            private=False, # Set to True if needed
            repo_type="model",
            exist_ok=True # Don't raise error if repo already exists
        )
        
        # Push the model to the hub
        logger.info(f"Uploading model from {model_path} to {repo_id}")
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model",
            token=hf_token,
            commit_message=f"Upload finetuned model from {model_path}"
        )
        
        logger.info(f"Model successfully pushed to https://huggingface.co/{repo_id}")
        return repo_id
    
    except Exception as e:
        logger.error(f"Error pushing model to Hugging Face Hub: {str(e)}")
        # Consider re-raising or handling specific errors
        return None


def calculate_simulated_ragas_metrics( # Renamed function
    query: str, 
    context: str, 
    model: SentenceTransformer
) -> Dict[str, float]:
    """
    Calculate SIMULATED RAGAS-like metrics based on model similarity.
    NOTE: These are NOT true RAGAS metrics which require LLM evaluation.
    This function is for evaluating the retriever's perspective.
    """
    # Ensure inputs are strings
    query = str(query)
    context = str(context)
    
    # Handle empty context gracefully
    if not context:
        return {
            'similarity': 0.0,
            'simulated_context_precision': 0.0,
            'simulated_context_recall': 0.0,
            'simulated_faithfulness': 0.0,
            'simulated_answer_relevancy': 0.0
        }

    # Calculate semantic similarity using the model
    try:
        query_embedding = model.encode([query], convert_to_tensor=True)
        context_embedding = model.encode([context], convert_to_tensor=True)
        similarity = torch.cosine_similarity(query_embedding, context_embedding).item()
    except Exception as e:
        logger.warning(f"Error calculating similarity for query '{query}' and context '{context[:50]}...': {e}")
        similarity = 0.0 # Default similarity on error

    # Derive simulated metrics from similarity with some variation
    # These formulas are illustrative and can be adjusted
    precision = max(0.0, min(1.0, similarity * 0.8 + random.uniform(-0.1, 0.1))) # Added noise range
    recall = max(0.0, min(1.0, similarity * 0.9 + random.uniform(-0.05, 0.05)))
    faith = max(0.0, min(1.0, similarity * 0.7 + random.uniform(-0.15, 0.15)))
    ans_rel = max(0.0, min(1.0, similarity * 0.85 + random.uniform(-0.1, 0.1)))
    
    return {
        'similarity': float(similarity),
        'simulated_context_precision': float(precision), # Prefixed with 'simulated_'
        'simulated_context_recall': float(recall),
        'simulated_faithfulness': float(faith),
        'simulated_answer_relevancy': float(ans_rel)
    }

def evaluate_models(
    base_model_path: str,
    finetuned_model_path: str,
    test_data: pd.DataFrame,
    output_dir: str
) -> Dict[str, Any]:
    """Evaluate and compare base and finetuned models using simulated metrics."""
    # Load models
    logger.info(f"Loading base model: {base_model_path}")
    base_model = SentenceTransformer(base_model_path)
    logger.info(f"Loading finetuned model: {finetuned_model_path}")
    finetuned_model = SentenceTransformer(finetuned_model_path)
    
    # Initialize results structure based on the keys from the simulation function
    metric_keys = [
        'similarity', 'simulated_context_precision', 'simulated_context_recall', 
        'simulated_faithfulness', 'simulated_answer_relevancy'
    ]
    results = {
        'base_model': {key: [] for key in metric_keys},
        'finetuned_model': {key: [] for key in metric_keys}
    }
    
    # Evaluate each query-context pair
    logger.info(f"Evaluating models on {len(test_data)} test examples...")
    for _, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Evaluating"):
        query = row['query']
        context = row['context']
        
        # Calculate metrics for base model
        base_metrics = calculate_simulated_ragas_metrics(query, context, base_model)
        for metric, value in base_metrics.items():
            results['base_model'][metric].append(value)
        
        # Calculate metrics for finetuned model
        finetuned_metrics = calculate_simulated_ragas_metrics(query, context, finetuned_model)
        for metric, value in finetuned_metrics.items():
            results['finetuned_model'][metric].append(value)
    
    # Calculate average metrics
    avg_results = {
        'base_model': {metric: np.mean(values) if values else 0 for metric, values in results['base_model'].items()},
        'finetuned_model': {metric: np.mean(values) if values else 0 for metric, values in results['finetuned_model'].items()}
    }
    
    # Calculate improvements
    improvements = {
        metric: {
            'base': avg_results['base_model'][metric],
            'finetuned': avg_results['finetuned_model'][metric],
            'absolute': avg_results['finetuned_model'][metric] - avg_results['base_model'][metric],
            'relative': (avg_results['finetuned_model'][metric] - avg_results['base_model'][metric]) / 
                        avg_results['base_model'][metric] * 100 if avg_results['base_model'][metric] != 0 else 0 # Avoid division by zero
        }
        for metric in avg_results['base_model'].keys()
    }
    
    # Create comparison report
    comparison = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'base_model': base_model_path,
        'finetuned_model': finetuned_model_path,
        'metrics': improvements,
        'raw_results': results # Keep raw results if needed for detailed analysis
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON results
    json_path = os.path.join(output_dir, f"model_comparison_{timestamp}.json")
    with open(json_path, 'w') as f:
        # Convert numpy types to standard types for JSON serialization
        json.dump(comparison, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)

    
    # Generate markdown report
    markdown_path = os.path.join(output_dir, f"model_comparison_{timestamp}.md")
    with open(markdown_path, 'w') as f:
        f.write('# Model Finetuning Comparison Report\n\n')
        f.write(f'Generated on: {comparison["timestamp"]}\n\n')
        f.write('## Model Details\n')
        f.write(f'- Base Model: `{base_model_path}`\n')
        f.write(f'- Finetuned Model: `{finetuned_model_path}`\n\n')
        
        # Metrics table
        f.write('## Metrics Comparison (Simulated RAGAS)\n\n') # Clarify metrics are simulated
        f.write('| Metric | Base Model | Finetuned Model | Absolute Improvement | Relative Improvement |\n')
        f.write('|--------|------------|-----------------|---------------------|--------------------|\n')
        
        for metric, values in improvements.items():
            # Format relative improvement carefully
            relative_imp_str = f"{values['relative']:.1f}%" if values['relative'] is not None else "N/A"
            f.write(f'| {metric} | {values["base"]:.3f} | {values["finetuned"]:.3f} | ')
            f.write(f'{values["absolute"]:.3f} | {relative_imp_str} |\n')
        
        f.write('\n## Analysis\n\n')
        
        # Filter out metrics where relative improvement calculation might not be meaningful (e.g., base is 0)
        valid_metrics_for_rel_imp = {m: v for m, v in improvements.items() if v['base'] != 0}

        if valid_metrics_for_rel_imp:
             # Find best improvement among valid metrics
             best_metric = max(
                 valid_metrics_for_rel_imp.keys(),
                 key=lambda m: valid_metrics_for_rel_imp[m]['relative']
             )
             f.write(f'- Highest relative improvement in **{best_metric}**: {improvements[best_metric]["relative"]:.1f}%\n')
        else:
             f.write("- Relative improvement calculation not applicable (base metrics might be zero).\n")

        
        # Simulated Faithfulness specific analysis
        faith_metric_name = 'simulated_faithfulness'
        f.write(f'\n### Simulated Faithfulness Analysis\n')
        f.write(f'- Base model {faith_metric_name}: {improvements[faith_metric_name]["base"]:.3f}\n')
        f.write(f'- Finetuned model {faith_metric_name}: {improvements[faith_metric_name]["finetuned"]:.3f}\n')
        f.write(f'- Absolute improvement: {improvements[faith_metric_name]["absolute"]:.3f}\n')
        # Format relative improvement for faithfulness
        faith_relative_imp_str = f"{improvements[faith_metric_name]['relative']:.1f}%" if improvements[faith_metric_name]['relative'] is not None else "N/A"
        f.write(f'- Relative improvement: {faith_relative_imp_str}\n')

    
    logger.info(f"Evaluation results saved to {json_path} and {markdown_path}")
    return comparison

def main():
    """Main function to run the entire pipeline."""
    parser = argparse.ArgumentParser(description='RAGAS-based data generation, model finetuning, and evaluation')
    parser.add_argument('--prompt_files', nargs='+', default=['data/f_prompts.txt', 'data/nf_prompts.txt'],
                        help='List of prompt files to use')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                        help='Base model to finetune')
    parser.add_argument('--testset_size', type=int, default=100,
                        help='Number of examples to generate using RAGAS')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='ragas_results',
                        help='Directory to save results (datasets, models, evaluations)')
    parser.add_argument('--openai_api_key', type=str, default=os.environ.get("OPENAI_API_KEY"),
                        help='OpenAI API key (if not set in environment)')
    # Hugging Face arguments
    parser.add_argument('--hf_token', type=str, default=os.environ.get("HF_TOKEN"),
                        help='Hugging Face API token for pushing the model (if not set in environment)')
    parser.add_argument('--hf_repo_name', type=str, default='finetuned-prompt-retriever',
                        help='Repository name for the finetuned model on Hugging Face Hub')
    parser.add_argument('--hf_org', type=str, default=None,
                        help='Optional Hugging Face organization name')

    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Load prompts
    logger.info(f"Loading prompts from {args.prompt_files}...")
    all_prompts = []
    for file in args.prompt_files:
        try:
            all_prompts.extend(load_prompts(file))
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {file}. Please ensure it exists.")
            return # Exit if prompt files are missing
    if not all_prompts:
        logger.error("No prompts loaded. Exiting.")
        return
    logger.info(f"Loaded {len(all_prompts)} prompts")
    
    # Step 2: Convert prompts to documents
    documents = prompts_to_documents(all_prompts)
    logger.info(f"Created {len(documents)} LangChain documents with metadata")
    
    # Step 3: Generate test set using RAGAS
    try:
        test_data = generate_ragas_testset(
            documents, 
            args.output_dir, 
            testset_size=args.testset_size,
            openai_api_key=args.openai_api_key
        )
        if test_data is None or test_data.empty:
             logger.error("RAGAS test set generation failed or produced an empty dataset. Exiting.")
             return
    except Exception as e:
        logger.error(f"Error during RAGAS test set generation: {e}", exc_info=True)
        return

    # Step 4: Prepare training data
    training_examples = prepare_training_data(test_data)
    logger.info(f"Created {len(training_examples)} training examples for SentenceTransformer")
    if not training_examples:
        logger.error("Failed to create training examples. Check the generated RAGAS dataset format and content. Exiting.")
        return
        
    # Step 5: Finetune model
    try:
        finetuned_model_path = finetune_model(
            args.model,
            training_examples,
            args.output_dir, # Save finetuned model within the main output dir
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        if not finetuned_model_path:
             logger.error("Model finetuning failed. Exiting.")
             return
    except Exception as e:
        logger.error(f"Error during model finetuning: {e}", exc_info=True)
        return

    # Step 6: Evaluate models
    try:
        evaluate_models(
            args.model,
            finetuned_model_path,
            test_data,
            args.output_dir # Save evaluation results in the main output dir
        )
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}", exc_info=True)
        # Continue to push model even if evaluation fails? Or stop? Let's stop for now.
        return 

    # Step 7: Push finetuned model to Hugging Face Hub
    try:
        push_to_huggingface(
            model_path=finetuned_model_path,
            hf_token=args.hf_token,
            repo_name=args.hf_repo_name,
            organization=args.hf_org
        )
    except Exception as e:
        logger.error(f"Error pushing model to Hugging Face Hub: {e}", exc_info=True)
        # Log error but don't necessarily stop the script

    logger.info("Pipeline completed!")

if __name__ == "__main__":
    main() 
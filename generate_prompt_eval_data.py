import re
import json
import random
from typing import List, Dict, Tuple
import pandas as pd

def extract_tags_and_prompt(text: str) -> Tuple[List[str], str]:
    """Extract tags and prompt text from a prompt entry."""
    tags = re.findall(r'<([^>]+)>', text)
    # Remove tag names that are the same as their content
    tags = [tag for tag in tags if not tag.endswith(f" </{tag}>")]
    # Clean up tags by removing closing tags
    tags = [tag for tag in tags if not tag.startswith('/')]
    
    # Extract the actual prompt text
    prompt_text = text.split('prompt ')[-1].strip()
    return tags, prompt_text

def generate_keyword_queries(tags: List[str], prompt_text: str) -> List[Dict]:
    """Generate different types of keyword queries for a prompt."""
    queries = []
    
    # 1. Single genre query
    genre_tags = [tag for tag in tags if tag in ['Fantasy', 'Science Fiction', 'Horror', 'Mystery', 'Romance']]
    if genre_tags:
        queries.append({
            'query': f"Find a {genre_tags[0]} writing prompt",
            'relevance': 1.0
        })
    
    # 2. Theme + genre query
    theme_tags = [tag for tag in tags if tag not in genre_tags and tag != 'prompt']
    if theme_tags and genre_tags:
        queries.append({
            'query': f"Find a {genre_tags[0]} prompt about {theme_tags[0].lower()}",
            'relevance': 1.0
        })
    
    # 3. Keyword-based queries
    words = prompt_text.split()
    key_nouns = [word for word in words if len(word) > 4 and word.isalpha()]
    if key_nouns:
        # Select 1-3 random keywords
        num_keywords = min(len(key_nouns), random.randint(1, 3))
        selected_keywords = random.sample(key_nouns, num_keywords)
        query = f"Find a writing prompt about {' and '.join(selected_keywords).lower()}"
        queries.append({
            'query': query,
            'relevance': 1.0
        })
    
    # 4. Generate some negative examples (queries that shouldn't match)
    opposite_genres = [g for g in ['Fantasy', 'Science Fiction', 'Horror', 'Mystery', 'Romance'] if g not in genre_tags]
    if opposite_genres:
        queries.append({
            'query': f"Find a {opposite_genres[0]} writing prompt",
            'relevance': 0.0
        })
    
    return queries

def load_prompts(filename: str) -> List[str]:
    """Load prompts from a text file."""
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def generate_evaluation_dataset(prompt_files: List[str], output_file: str):
    """Generate the evaluation dataset from prompt files."""
    all_prompts = []
    for file in prompt_files:
        all_prompts.extend(load_prompts(file))
    
    evaluation_data = []
    
    for prompt_text in all_prompts:
        tags, prompt_content = extract_tags_and_prompt(prompt_text)
        queries = generate_keyword_queries(tags, prompt_content)
        
        for query in queries:
            evaluation_data.append({
                'query': query['query'],
                'context': prompt_text,
                'relevance': query['relevance']
            })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(evaluation_data)
    df.to_csv(output_file, index=False)
    print(f"Generated {len(df)} evaluation examples")
    
    # Print some example queries
    print("\nExample queries:")
    for i, row in df.sample(min(5, len(df))).iterrows():
        print(f"Query: {row['query']}")
        print(f"Relevance: {row['relevance']}")
        print(f"Context: {row['context']}\n")

if __name__ == "__main__":
    prompt_files = ['data/f_prompts.txt', 'data/nf_prompts.txt']
    generate_evaluation_dataset(prompt_files, 'data/prompt_evaluation_dataset.csv') 
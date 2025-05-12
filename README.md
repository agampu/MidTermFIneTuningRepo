# Prompt Retrieval Model Fine-tuning

This repository contains the code and documentation for fine-tuning embedding models for creative writing prompt retrieval. The project includes evaluation scripts, fine-tuning code, and comprehensive documentation of the process and results.

## Project Structure

```
.
├── data/                      # Evaluation and training datasets
├── evaluation_results/        # Model evaluation results and metrics
├── generate_prompt_eval_data.py   # Script to generate evaluation dataset
├── prompt_evaluation.py       # Script for evaluating prompt retrieval
├── evaluate_embeddings.py     # Core embedding evaluation utilities
├── finetune_embeddings.py     # Fine-tuning script for embedding models
├── FineTuningStrategy.md      # Detailed fine-tuning strategy documentation
├── FineTuningEmbeddingModels.md  # Fine-tuning results and analysis
└── pyproject.toml            # Project dependencies and configuration
```

## Setup

1. Clone the repository:
```bash
git clone [your-repo-url]
cd [repo-name]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Generate evaluation dataset:
```bash
python generate_prompt_eval_data.py
```

2. Evaluate baseline models:
```bash
python prompt_evaluation.py
```

3. Fine-tune the model:
```bash
python finetune_embeddings.py
```

## Documentation

- [Fine-tuning Strategy](FineTuningStrategy.md) - Comprehensive guide on the fine-tuning approach
- [Fine-tuning Results](FineTuningEmbeddingModels.md) - Detailed analysis of the fine-tuning results

## License

MIT License 
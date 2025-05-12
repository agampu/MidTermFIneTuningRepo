# Prompt Retrieval Model Fine-tuning

This repository contains the code and documentation for fine-tuning embedding models for creative writing prompt retrieval. The project includes evaluation scripts, fine-tuning code, and comprehensive documentation of the process and results.

## Project Structure

```
.
├── data/                      # Evaluation and training datasets
├── evaluation_results/        # Model evaluation results and metrics
├── generate_prompt_eval_data.py   # Script to generate evaluation dataset
├── prompt_evaluation.py       # Script for evaluating prompt retrieval
├── finetune_embeddings.py     # Fine-tuning script for embedding models
├── FineTuningStrategy.md      # Detailed fine-tuning strategy documentation
├── FineTuningEmbeddingModels.md  # Fine-tuning results and analysis
└── pyproject.toml            # Project dependencies and configuration
```

## Workflow Overview

The project follows a clear three-step workflow:

1.  **Evaluation Dataset Creation (`generate_prompt_eval_data.py`):**
    *   The process begins with generating a "golden" evaluation dataset.
    *   This script creates `data/prompt_evaluation_dataset.csv`, which contains queries, relevant/irrelevant prompts (contexts), and their corresponding relevance scores. This dataset is foundational for both evaluating baseline models and for fine-tuning.

2.  **Baseline Model Evaluation (`prompt_evaluation.py`):**
    *   Once the evaluation dataset is ready, this script is used to assess the performance of various pre-trained (baseline) embedding models.
    *   It loads `data/prompt_evaluation_dataset.csv` and runs evaluations to identify the most promising model for the creative writing prompt retrieval task. The `all-MiniLM-L6-v2` model was selected based on these results.

3.  **Model Fine-tuning (`finetune_embeddings.py`):**
    *   The final step involves fine-tuning the selected baseline model (`all-MiniLM-L6-v2`).
    *   This script also utilizes `data/prompt_evaluation_dataset.csv`, using it (or deriving training examples from it) to further train and adapt the chosen embedding model specifically for improved performance on this particular dataset and task.

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

## Results Summary

This project involved a three-step process to improve prompt retrieval:

1.  **Evaluation Data Generation (`generate_prompt_eval_data.py`):**
    *   Successfully generated a dataset of 532 evaluation examples.
    *   This dataset includes a mix of positive (relevant query-prompt pairs) and negative (irrelevant query-prompt pairs) examples, crucial for robust model evaluation.
    *   Example entries from `data/prompt_evaluation_dataset.csv`:
        ```csv
        query,context,relevance
        Find a Fantasy writing prompt,"<Fantasy> </Fantasy> <Magic> </Magic> <The Accidental Alchemist> </The Accidental Alchemist> prompt A clumsy apprentice baker accidentally creates a potion that grants temporary, unpredictable magical abilities instead of a perfect sourdough starter.",1.0
        Find a Science Fiction writing prompt,"<Fantasy> </Fantasy> <Magic> </Magic> <The Accidental Alchemist> </The Accidental Alchemist> prompt A clumsy apprentice baker accidentally creates a potion that grants temporary, unpredictable magical abilities instead of a perfect sourdough starter.",0.0
        Find a Science Fiction prompt about alien encounter,"<Science Fiction> </Science Fiction> <Alien Encounter> </Alien Encounter> <Lost Translator> </Lost Translator> prompt An alien lands on Earth, but their universal translator is broken. They must communicate their urgent message using only gestures and drawings to a skeptical bystander.",1.0
        Find a Fantasy writing prompt,"<Science Fiction> </Science Fiction> <Alien Encounter> </Alien Encounter> <Lost Translator> </Lost Translator> prompt An alien lands on Earth, but their universal translator is broken. They must communicate their urgent message using only gestures and drawings to a skeptical bystander.",0.0
        ```

2.  **Baseline Model Evaluation (`prompt_evaluation.py`):**
    *   Evaluated several pre-trained embedding models (e.g., `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `multi-qa-mpnet-base-dot-v1`).
    *   The `all-MiniLM-L6-v2` model was selected as the best baseline, achieving an MRR of 0.193 and Precision@1 of 0.547 on the generated dataset. Full results are available in `evaluation_results/evaluation_report_20250511_232805.md`.

3.  **Model Fine-tuning (`finetune_embeddings.py`):**
    *   The selected `all-MiniLM-L6-v2` model was fine-tuned using the generated dataset.
    *   Fine-tuning significantly improved performance. For example, on a test set:
        *   **Precision@1 increased from 0.667 to 0.898 (34.6% relative improvement).**
        *   **MRR increased from 0.714 to 0.931 (30.5% relative improvement).**
        *   **NDCG@5 increased from 0.569 to 0.933 (63.9% relative improvement).**
    *   Detailed comparison report: `evaluation_results/finetuning_comparison/comparison_20250511_235754.md`.
    *   The fine-tuned model is saved locally and can be pushed to Hugging Face Hub.

## Documentation

- [Fine-tuning Strategy](FineTuningStrategy.md) - Comprehensive guide on the fine-tuning approach
- [Fine-tuning Results](FineTuningEmbeddingModels.md) - Detailed analysis of the fine-tuning results

## License

MIT License 
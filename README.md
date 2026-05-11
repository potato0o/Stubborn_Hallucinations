# From Flat Facts to Sharp Hallucinations: Detecting Stubborn Errors via Gradient Sensitivity [ICML 2026]

Official PyTorch Implementation of the paper [From Flat Facts to Sharp Hallucinations: Detecting Stubborn Errors via Gradient Sensitivity](https://arxiv.org/abs/2605.00939)

## Overview

Stubborn hallucinations occur when an LLM confidently and consistently generates incorrect information, even when the input context is slightly perturbed. This project implements a gradient-based curvature method - **Embedding-Perturbed Gradient Sensitivity (EPGS)** - to detect such hallucinations by analyzing the sensitivity of the model's internal representations (gradients) to input perturbations.

### Key Features
*   **Gradient-Based Detection**: measures the "curvature" of the loss landscape to identify stubborn errors.
*   **Input Perturbations**: supports template-based textual perturbations, neural paraphrasing (using Pegasus), embedding noise injection, and MC Dropout.
*   **Key Phrase Masking**: optionally focuses analysis on specific entities (NER-detected) in the answer.
*   **Flexible Model Support**: Supports Llama-2, Llama-3, Falcon, Mistral, and others via HuggingFace.

## Installation

1.  **Create Environment**
    It is recommended to use Conda.
    ```bash
    conda env create -f environment.yaml
    conda activate stubb_hallu
    ```
    *Note: Requires Python 3.11 and PyTorch with CUDA support.*

2.  **Setup Environment Variables**
    Ensure your HuggingFace token and other paths are set if necessary. You may need to adjust `run.py` or export variables or set up `wandb` API key.

## Usage

### Running an Experiment

The primary script is `run.py`. It runs the gradient-based curvature analysis to generate answers and compute curvature scores.

```bash
python run.py \
    --model_name Meta-Llama-3-8B \
    --gradient_target last_transformer_block \
    --use_embedding_noise_perturbation \
    --embedding_noise_epsilon 0.1 \
    --dataset squad \
    --num_perturbations 1 \
    --metric bertscore \
    --num_samples 400 \
    --use_key_phrase_masking \
    --no-get_training_set_generations
```

**Key Arguments:**
*   `--gradient_target`: layer to compute gradients on (e.g., `lm_head`, `last_transformer_block`).
*   `--use_embedding_noise_perturbation`: injects Gaussian noise into input embeddings instead of textual perturbations.
*   `--embedding_noise_epsilon`: sets standard deviation of the Gaussian noise (default: 0.1).
*   `--use_mc_dropout_perturbation`: uses MC Dropout (stochastic weight masking) as a perturbation method.
*   `--mc_dropout_rate`: dropout probability for MC Dropout perturbation (default: 0.1).
*   `--use_paraphrase_perturbation`: uses a paraphraser model instead of fixed templates.
*   `--use_key_phrase_masking`: masks non-entity tokens in the target to focus gradients on key phrases.
*   `--num_few_shot`: number of few-shot examples (default: 5).


### Stubborn Hallucination Analysis

#### 1. Generating the Stubborn Dataset
Use `stubb_dataset/generate_stubborn_dataset.py` to process a standard dataset and identify samples where the model produces consistent responses over multiple generations.

```bash
python stubb_dataset/generate_stubborn_dataset.py \
    --model_name Meta-Llama-3-8B \
    --dataset nq \
    --num_generations 5 \
    --temperature 0.7 \
    --output_file stubb_dataset/stubborn_nq.json
```

#### 2. Running Experiments on Stubborn Subset
Once generated, you can run the gradient-based curvature analysis directly on the stubborn dataset by passing the JSON file path to the `--dataset` argument.

```bash
python run.py \
    --model_name Meta-Llama-3-8B \
    --dataset stubb_dataset/stubborn_nq.json \
    --gradient_target last_transformer_block \
    --use_embedding_noise_perturbation \
    --num_perturbations 1 \
    --metric bertscore \
    --num_samples 50 \
    --no-get_training_set_generations
```


### Metrics & Evaluation

The system supports various metrics for assessing generation quality:
*   `squad`: SQuAD-based F1 (default).
*   `bertscore`: Semantic similarity using BERTScore.
*   `llm`: LLM-based correctness check (entailment).

Add `--metric bertscore` to your run command to use BERTScore.


## Citation

```bibtex
@misc{liew2026flatfactssharphallucinations,
      title={From Flat Facts to Sharp Hallucinations: Detecting Stubborn Errors via Gradient Sensitivity}, 
      author={Yee Zhing Liew and Andrew Huey Ping Tan and Anwar P. P Abdul Majeed},
      year={2026},
      eprint={2605.00939},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.00939}, 
}
```

